#!/usr/bin/env python3
"""Build, score, and analyze a reusable 5-model x 2-rollout candidate pool."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REBUTTAL_DIR = HERE.parent
DYNS_DIR = REBUTTAL_DIR.parent
LONG_CONTEXT_DIR = REBUTTAL_DIR / "long_context"
sys.path.insert(0, str(REBUTTAL_DIR))
sys.path.insert(0, str(LONG_CONTEXT_DIR))

from common import (  # noqa: E402
    ChatCompletionsClient,
    dump_json,
    dump_jsonl,
    extract_json_object,
    load_json_or_jsonl,
    resolve_token,
)
from evaluate_fixed_contexts import (  # noqa: E402
    DIMENSIONS,
    format_dialogue,
    load_models,
    load_prompt_constants,
)
from user_simulator_prompts import build_user_simulator_prompt, sanitize_user_response  # noqa: E402

DEFAULT_INPUT = DYNS_DIR / "test" / "test-construct" / "outputs" / "test_0424_1043" / "merged_dialogues.jsonl"
DEFAULT_MODELS = HERE / "models.json"
DEFAULT_JUDGE_PROMPTS = REBUTTAL_DIR / "base" / "dynsess_rubrics.py"
DEFAULT_ROLE_API_URL = "https://aigc-api.fuxi.netease.com/v1/chat/completions"
DEFAULT_USER_API_URL = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
USER_EXTRA_BODY = {
    "caching": {"type": "enabled"},
    "thinking": {"type": "disabled"},
}


def select_histories(input_path: Path, sample_size: int, seed: int, complete_only: bool) -> list[dict]:
    source = load_json_or_jsonl(input_path)
    max_messages = max(len(item.get("dialogue", [])) for item in source)
    eligible = [
        index for index, item in enumerate(source)
        if not complete_only or len(item.get("dialogue", [])) == max_messages
    ]
    if sample_size > len(eligible):
        raise ValueError(f"Cannot sample {sample_size} from {len(eligible)} eligible histories")
    indices = random.Random(seed).sample(eligible, sample_size)
    selected = []
    for rank, source_index in enumerate(indices):
        item = source[source_index]
        dialogue = item.get("dialogue", [])
        if not dialogue or dialogue[-1].get("role") != "assistant":
            raise ValueError(f"Source history {source_index} does not end with assistant")
        if any(
            dialogue[index].get("role") != ("user" if index % 2 == 0 else "assistant")
            for index in range(len(dialogue))
        ):
            raise ValueError(f"Source history {source_index} is not strictly alternating")
        selected.append({
            "history_id": f"h{source_index:03d}",
            "sample_rank": rank,
            "source_index": source_index,
            "history_turns": len(dialogue) // 2,
            "persona_info": item.get("persona_info", ""),
            "user_system_prompt": item.get("user_system_prompt", ""),
            "dialogue": dialogue,
        })
    return selected


def model_system_prompt(history: dict, model: dict) -> str:
    persona = history["persona_info"]
    prompt = persona if persona.startswith("请你扮演以下人设：") else f"请你扮演以下人设：{persona}"
    suffix = model.get("system_suffix", "").strip()
    return prompt + ("\n\n【重要】" + suffix if suffix else "")


def user_view(dialogue: list[dict]) -> list[dict]:
    return [
        {
            "role": "user" if message["role"] == "assistant" else "assistant",
            "content": message["content"],
        }
        for message in dialogue
    ]


def model_client(model: dict, args: argparse.Namespace) -> ChatCompletionsClient:
    token_env = model.get("token_env", "DYNS_ROLE_API_TOKEN")
    token = model.get("token") or os.environ.get(token_env, "") or resolve_token(args.role_api_token, token_env)
    if not token and not args.mock_api and model.get("requires_token", True):
        raise ValueError(f"Missing token for {model['name']}; set {token_env} or DYNS_API_TOKEN")
    return ChatCompletionsClient(
        model.get("api_url", args.role_api_url),
        token,
        timeout=args.timeout,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
        mock_api=args.mock_api,
    )


def user_client(args: argparse.Namespace) -> ChatCompletionsClient:
    token = (
        args.user_api_key
        or os.environ.get("ARK_API_KEY", "")
        or resolve_token(None, "DYNS_USER_API_TOKEN")
    )
    if not token and not args.mock_api:
        raise ValueError("Set ARK_API_KEY/DYNS_USER_API_TOKEN or pass --user-api-key")
    return ChatCompletionsClient(
        args.user_api_url,
        token,
        timeout=args.timeout,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
        mock_api=args.mock_api,
    )


def valid_text(text: str, max_chars: int) -> bool:
    return bool(text and text != "null" and text.strip() and len(text) <= max_chars)


def pool_key(history_id: str, model_name: str, replicate: int) -> str:
    return f"{history_id}|{model_name}|r{replicate}"


def build_fixed_anchor(history: dict, client: ChatCompletionsClient, args: argparse.Namespace) -> str:
    prompt = build_user_simulator_prompt(
        args.user_simulator_style,
        history["persona_info"],
        history["user_system_prompt"],
    )
    response = client.complete(
        model=args.user_model,
        messages=[{"role": "system", "content": prompt}] + user_view(history["dialogue"]),
        temperature=args.user_temperature,
        max_tokens=args.user_max_tokens,
        extra_body=USER_EXTRA_BODY,
    )
    response = sanitize_user_response(response)
    if not valid_text(response, args.max_user_chars):
        raise ValueError(f"Invalid fixed user anchor for {history['history_id']}: {response!r}")
    return response


def prepare_selection(args: argparse.Namespace, output_dir: Path) -> list[dict]:
    path = output_dir / "selected_histories.jsonl"
    if path.exists() and not args.overwrite_selection:
        selected = load_json_or_jsonl(path)
        if len(selected) != args.sample_size:
            raise ValueError(f"Existing selection has {len(selected)} records, expected {args.sample_size}")
        return selected
    selected = select_histories(Path(args.input), args.sample_size, args.seed, args.complete_histories_only)
    dump_jsonl(selected, path)
    return selected


def generate_pool(args: argparse.Namespace, histories: list[dict], models: list[dict], output_dir: Path) -> list[dict]:
    progress_path = output_dir / "rollout_progress.json"
    completed = {}
    if progress_path.exists() and not args.overwrite_rollouts:
        completed = json.loads(progress_path.read_text(encoding="utf-8"))

    uclient = user_client(args)
    model_map = {model["name"]: model for model in models}
    clients = {name: model_client(model, args) for name, model in model_map.items()}

    anchor_path = output_dir / "fixed_user_anchors.json"
    anchors = {}
    if anchor_path.exists() and not args.overwrite_rollouts:
        anchors = json.loads(anchor_path.read_text(encoding="utf-8"))
    for history in histories:
        if history["history_id"] not in anchors:
            anchors[history["history_id"]] = build_fixed_anchor(history, uclient, args)
            dump_json(anchors, anchor_path)

    tasks = [
        (history, model, replicate)
        for history in histories
        for model in models
        for replicate in range(1, args.replicates + 1)
    ]
    pending = [
        task for task in tasks
        if pool_key(task[0]["history_id"], task[1]["name"], task[2]) not in completed
    ]
    lock = threading.Lock()

    def worker(history: dict, model: dict, replicate: int) -> tuple[str, dict]:
        full_dialogue = list(history["dialogue"])
        continued = []
        prompt = build_user_simulator_prompt(
            args.user_simulator_style,
            history["persona_info"],
            history["user_system_prompt"],
        )
        for turn in range(1, args.eval_turns + 1):
            if turn == 1:
                user_response = anchors[history["history_id"]]
            else:
                user_response = uclient.complete(
                    model=args.user_model,
                    messages=[{"role": "system", "content": prompt}] + user_view(full_dialogue),
                    temperature=args.user_temperature,
                    max_tokens=args.user_max_tokens,
                    extra_body=USER_EXTRA_BODY,
                )
                user_response = sanitize_user_response(user_response)
            if not valid_text(user_response, args.max_user_chars):
                raise ValueError(f"Invalid user response at turn {turn}: {user_response!r}")
            user_message = {"role": "user", "content": user_response}
            full_dialogue.append(user_message)
            continued.append(user_message)

            role_response = clients[model["name"]].complete(
                model=model["model"],
                messages=[{"role": "system", "content": model_system_prompt(history, model)}, *full_dialogue],
                temperature=float(model.get("temperature", args.role_temperature)),
                max_tokens=int(model.get("max_tokens", args.role_max_tokens)),
                extra_body=model.get("extra_body"),
            )
            if not valid_text(role_response, args.max_role_chars):
                raise ValueError(f"Invalid role response at turn {turn}: {role_response!r}")
            role_message = {"role": "assistant", "content": role_response}
            full_dialogue.append(role_message)
            continued.append(role_message)

        key = pool_key(history["history_id"], model["name"], replicate)
        return key, {
            "candidate_id": key,
            "history_id": history["history_id"],
            "source_index": history["source_index"],
            "history_turns": history["history_turns"],
            "model_name": model["name"],
            "model_id": model["model"],
            "replicate": replicate,
            "eval_turns": args.eval_turns,
            "fixed_first_user_message": anchors[history["history_id"]],
            "continued_dialogue": continued,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        }

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(worker, *task): task for task in pending}
        for count, future in enumerate(as_completed(futures), 1):
            history, model, replicate = futures[future]
            key = pool_key(history["history_id"], model["name"], replicate)
            try:
                key, result = future.result()
            except Exception as exc:
                print(f"[rollout failed] {key}: {exc}", flush=True)
                continue
            with lock:
                completed[key] = result
                if count % args.save_every == 0:
                    dump_json(completed, progress_path)
            print(f"[rollout] {key}", flush=True)
    dump_json(completed, progress_path)

    keys = [pool_key(h["history_id"], m["name"], r) for h, m, r in tasks]
    missing = [key for key in keys if key not in completed]
    if missing:
        raise RuntimeError(f"Missing {len(missing)} rollouts; rerun to resume. First: {missing[:3]}")
    ordered = [completed[key] for key in keys]
    dump_json(ordered, output_dir / "rollout_pool.json")
    return ordered


def judge_client(args: argparse.Namespace) -> ChatCompletionsClient:
    token = resolve_token(args.judge_api_token, "DYNS_EVAL_API_TOKEN")
    if not token and not args.mock_api:
        raise ValueError("Set DYNS_EVAL_API_TOKEN/DYNS_API_TOKEN or pass --judge-api-token")
    return ChatCompletionsClient(
        args.judge_api_url,
        token,
        timeout=args.timeout,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
        mock_api=args.mock_api,
    )


def score_pool(
    args: argparse.Namespace,
    histories: list[dict],
    candidates: list[dict],
    output_dir: Path,
) -> list[dict]:
    prompts = load_prompt_constants(Path(args.judge_prompt_file))
    history_map = {history["history_id"]: history for history in histories}
    client = judge_client(args)
    progress_path = output_dir / "judge_progress.json"
    completed = {}
    if progress_path.exists() and not args.overwrite_judgments:
        completed = json.loads(progress_path.read_text(encoding="utf-8"))

    tasks = [(candidate, dimension) for candidate in candidates for dimension in DIMENSIONS]
    pending = [task for task in tasks if f"{task[0]['candidate_id']}|{task[1]}" not in completed]
    lock = threading.Lock()

    def worker(candidate: dict, dimension: str) -> tuple[str, dict]:
        history = history_map[candidate["history_id"]]
        prompt = prompts[DIMENSIONS[dimension]]
        prompt = (
            prompt.replace("{character_profile}", history["persona_info"])
            .replace("{dialogue_history}", format_dialogue(history["dialogue"]))
            .replace("{dialogue}", format_dialogue(candidate["continued_dialogue"]))
        )
        raw = client.complete(
            model=args.judge_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=args.judge_temperature,
            max_tokens=args.judge_max_tokens,
        )
        parsed = extract_json_object(raw)
        result = parsed.get(dimension) if parsed else None
        if not isinstance(result, dict):
            raise ValueError(f"Judge response lacks {dimension}: {raw[:500]}")
        score = result.get("score")
        if not isinstance(score, (int, float)) or not 1 <= int(score) <= 5:
            raise ValueError(f"Invalid score for {dimension}: {result}")
        key = f"{candidate['candidate_id']}|{dimension}"
        return key, {"score": int(score), "reason": str(result.get("reason", "")), "raw_response": raw}

    with ThreadPoolExecutor(max_workers=args.judge_workers) as executor:
        futures = {executor.submit(worker, *task): task for task in pending}
        for count, future in enumerate(as_completed(futures), 1):
            candidate, dimension = futures[future]
            key = f"{candidate['candidate_id']}|{dimension}"
            try:
                key, result = future.result()
            except Exception as exc:
                print(f"[judge failed] {key}: {exc}", flush=True)
                continue
            with lock:
                completed[key] = result
                if count % args.save_every == 0:
                    dump_json(completed, progress_path)
            print(f"[judge] {key}", flush=True)
            if args.judge_sleep:
                time.sleep(args.judge_sleep)
    dump_json(completed, progress_path)

    missing = [f"{c['candidate_id']}|{d}" for c, d in tasks if f"{c['candidate_id']}|{d}" not in completed]
    if missing:
        raise RuntimeError(f"Missing {len(missing)} judgments; rerun to resume. First: {missing[:3]}")
    scored = []
    for candidate in candidates:
        scores = {dimension: completed[f"{candidate['candidate_id']}|{dimension}"] for dimension in DIMENSIONS}
        scored.append({
            **candidate,
            "scores": scores,
            "overall": statistics.mean(value["score"] for value in scores.values()),
        })
    dump_json(scored, output_dir / "scored_rollout_pool.json")
    return scored


STRATEGY_BUDGETS = {
    "random_1": 1,
    "random_2_models": 2,
    "same_model_2": 2,
    "all_5_models": 5,
    "all_10_pool_diagnostic": 10,
}


def choose_best(candidates: list[dict]) -> dict:
    return max(candidates, key=lambda item: (item["overall"], item["candidate_id"]))


def analyze_pool(
    args: argparse.Namespace,
    histories: list[dict],
    models: list[dict],
    scored: list[dict],
    output_dir: Path,
) -> None:
    names = [model["name"] for model in models]
    lookup = {(item["history_id"], item["model_name"], item["replicate"]): item for item in scored}
    assignments = []
    selected_rows = []

    for history in histories:
        permutation = list(names)
        random.Random(args.analysis_seed + history["source_index"] * 1009).shuffle(permutation)
        first, second = permutation[:2]
        groups = {
            "random_1": [lookup[(history["history_id"], first, 1)]],
            "random_2_models": [
                lookup[(history["history_id"], first, 1)],
                lookup[(history["history_id"], second, 1)],
            ],
            "same_model_2": [
                lookup[(history["history_id"], first, 1)],
                lookup[(history["history_id"], first, 2)],
            ],
            "all_5_models": [lookup[(history["history_id"], name, 1)] for name in permutation],
            "all_10_pool_diagnostic": [
                lookup[(history["history_id"], name, replicate)]
                for name in permutation
                for replicate in (1, 2)
            ],
        }
        assignments.append({
            "history_id": history["history_id"],
            "source_index": history["source_index"],
            "model_permutation": permutation,
            "random_1_model": first,
            "random_2_models": [first, second],
        })
        for strategy, candidates in groups.items():
            winner = choose_best(candidates)
            selected_rows.append({
                "history_id": history["history_id"],
                "source_index": history["source_index"],
                "strategy": strategy,
                "rollout_budget": STRATEGY_BUDGETS[strategy],
                "candidate_ids": [item["candidate_id"] for item in candidates],
                "selected_candidate_id": winner["candidate_id"],
                "selected_model": winner["model_name"],
                "selected_replicate": winner["replicate"],
                "overall": winner["overall"],
                "scores": winner["scores"],
            })
    dump_json(assignments, output_dir / "strategy_assignments.json")
    dump_json(selected_rows, output_dir / "strategy_selected_results.json")

    summary = []
    for strategy, budget in STRATEGY_BUDGETS.items():
        rows = [row for row in selected_rows if row["strategy"] == strategy]
        record: dict[str, Any] = {
            "strategy": strategy,
            "rollout_budget": budget,
            "role_api_calls_per_sample": budget * args.eval_turns,
            "user_api_calls_per_sample": 1 + budget * (args.eval_turns - 1),
            "judge_api_calls_per_sample": budget * len(DIMENSIONS),
            "n": len(rows),
        }
        overall = [row["overall"] for row in rows]
        record["overall_mean"] = statistics.mean(overall)
        record["overall_std"] = statistics.stdev(overall) if len(overall) > 1 else 0.0
        record["overall_ci95"] = 1.96 * record["overall_std"] / math.sqrt(len(overall)) if overall else 0.0
        for dimension in DIMENSIONS:
            values = [row["scores"][dimension]["score"] for row in rows]
            record[f"{dimension}_mean"] = statistics.mean(values)
        summary.append(record)

    csv_path = output_dir / "strategy_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    lines = [
        "# Rollout Candidate Ablation",
        "",
        "| Strategy | Rollouts | Role calls | User calls | Judge calls | N | Overall | Human | Role | Context | Interactive |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['strategy']} | {row['rollout_budget']} | {row['role_api_calls_per_sample']} | "
            f"{row['user_api_calls_per_sample']} | {row['judge_api_calls_per_sample']} | {row['n']} | "
            f"{row['overall_mean']:.3f} +/- {row['overall_ci95']:.3f} | "
            f"{row['human_likeness_mean']:.3f} | {row['role_consistency_mean']:.3f} | "
            f"{row['context_consistency_mean']:.3f} | {row['interactive_ability_mean']:.3f} |"
        )
    (output_dir / "strategy_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a reusable rollout pool and analyze best-of-N strategies.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode", choices=("generate", "evaluate", "analyze", "all"), default="all")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--models-config", default=str(DEFAULT_MODELS))
    parser.add_argument("--output-dir", default=str(HERE / "outputs" / "test100_sample20_seed20260529"))
    parser.add_argument("--sample-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260529)
    parser.add_argument(
        "--complete-histories-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Sample only histories with the dataset's maximum message count.",
    )
    parser.add_argument("--analysis-seed", type=int, default=20260710)
    parser.add_argument("--eval-turns", type=int, default=10)
    parser.add_argument("--replicates", type=int, default=2)

    parser.add_argument("--role-api-url", default=DEFAULT_ROLE_API_URL)
    parser.add_argument("--role-api-token", default=None)
    parser.add_argument("--role-temperature", type=float, default=1.0)
    parser.add_argument("--role-max-tokens", type=int, default=2048)
    parser.add_argument("--max-role-chars", type=int, default=500)
    parser.add_argument("--user-api-url", default=DEFAULT_USER_API_URL)
    parser.add_argument("--user-api-key", default=None)
    parser.add_argument("--user-model", default="doubao-1-5-pro-32k-character-250715")
    parser.add_argument("--user-simulator-style", choices=("passive", "balanced", "proactive"), default="passive")
    parser.add_argument("--user-temperature", type=float, default=1.0)
    parser.add_argument("--user-max-tokens", type=int, default=200)
    parser.add_argument("--max-user-chars", type=int, default=150)

    parser.add_argument("--judge-prompt-file", default=str(DEFAULT_JUDGE_PROMPTS))
    parser.add_argument("--judge-api-url", default=DEFAULT_ROLE_API_URL)
    parser.add_argument("--judge-api-token", default=None)
    parser.add_argument("--judge-model", default="gemini-3-flash-preview")
    parser.add_argument("--judge-temperature", type=float, default=0.7)
    parser.add_argument("--judge-max-tokens", type=int, default=4096)

    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--judge-workers", type=int, default=12)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--judge-sleep", type=float, default=0.0)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=3.0)
    parser.add_argument("--overwrite-selection", action="store_true")
    parser.add_argument("--overwrite-rollouts", action="store_true")
    parser.add_argument("--overwrite-judgments", action="store_true")
    parser.add_argument("--mock-api", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.replicates < 2:
        raise ValueError("--replicates must be at least 2 for same_model_2")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    histories = prepare_selection(args, output_dir)
    models = load_models(Path(args.models_config))
    if len(models) != 5:
        raise ValueError(f"This ablation requires exactly 5 models, got {len(models)}")

    dump_json({
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input": str(Path(args.input).resolve()),
        "models_config": str(Path(args.models_config).resolve()),
        "sample_size": args.sample_size,
        "complete_histories_only": args.complete_histories_only,
        "seed": args.seed,
        "analysis_seed": args.analysis_seed,
        "eval_turns": args.eval_turns,
        "replicates": args.replicates,
        "user_model": args.user_model,
        "user_simulator_style": args.user_simulator_style,
        "judge_model": args.judge_model,
    }, output_dir / "run_config.json")

    pool_path = output_dir / "rollout_pool.json"
    scored_path = output_dir / "scored_rollout_pool.json"
    if args.mode in {"generate", "all"}:
        candidates = generate_pool(args, histories, models, output_dir)
    else:
        if not pool_path.exists():
            raise FileNotFoundError(f"Run --mode generate first: {pool_path}")
        candidates = json.loads(pool_path.read_text(encoding="utf-8"))

    if args.mode in {"evaluate", "all"}:
        scored = score_pool(args, histories, candidates, output_dir)
    elif args.mode == "analyze":
        if not scored_path.exists():
            raise FileNotFoundError(f"Run --mode evaluate first: {scored_path}")
        scored = json.loads(scored_path.read_text(encoding="utf-8"))
    else:
        scored = []

    if args.mode in {"analyze", "all"}:
        analyze_pool(args, histories, models, scored, output_dir)
        print(f"Ablation complete: {output_dir / 'strategy_summary.md'}")
    elif args.mode == "generate":
        print(f"Rollout generation complete: {pool_path}")
    else:
        print(f"Pool evaluation complete: {scored_path}")


if __name__ == "__main__":
    main()
