#!/usr/bin/env python3
"""Generate and judge model replies at fixed Doubao-context checkpoints."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import os
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

from common import (
    ChatCompletionsClient,
    dump_json,
    extract_json_object,
    load_json_or_jsonl,
    resolve_token,
    slugify,
)

HERE = Path(__file__).resolve().parent
REBUTTAL_DIR = HERE.parent
DYNS_DIR = REBUTTAL_DIR.parent
sys.path.insert(0, str(REBUTTAL_DIR))

from user_simulator_prompts import build_user_simulator_prompt, sanitize_user_response  # noqa: E402

DEFAULT_JUDGE_PROMPTS = REBUTTAL_DIR / "base" / "dynsess_rubrics.py"
DEFAULT_MODELS = HERE / "models.example.json"
DEFAULT_API_URL = "https://aigc-api.fuxi.netease.com/v1/chat/completions"
DEFAULT_USER_API_URL = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
DIMENSIONS = {
    "human_likeness": "multi_turn_eval_score_human_likeness",
    "role_consistency": "multi_turn_eval_score_role_consistency",
    "context_consistency": "multi_turn_eval_score_context_consistency",
    "interactive_ability": "multi_turn_eval_score_interactivity",
}


def parse_int_list(value: str) -> tuple[int, ...]:
    values = tuple(sorted({int(part.strip()) for part in value.split(",") if part.strip()}))
    if not values or values[0] < 1:
        raise argparse.ArgumentTypeError("Expected comma-separated positive integers")
    return values


def load_models(path: Path) -> list[dict]:
    models = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(models, list) or not models:
        raise ValueError("Model config must be a non-empty JSON list")
    names = set()
    for model in models:
        if not model.get("name") or not model.get("model"):
            raise ValueError("Every model requires non-empty 'name' and 'model' fields")
        if model["name"] in names:
            raise ValueError(f"Duplicate model name: {model['name']}")
        names.add(model["name"])
    return models


def load_prompt_constants(path: Path) -> dict[str, str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    wanted = set(DIMENSIONS.values())
    prompts = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id in wanted:
            value = ast.literal_eval(node.value)
            if isinstance(value, str):
                prompts[target.id] = value
    missing = wanted - set(prompts)
    if missing:
        raise ValueError(f"Missing judge prompts in {path}: {sorted(missing)}")
    return prompts


def validate_context(record: dict, checkpoints: tuple[int, ...]) -> None:
    dialogue = record.get("dialogue", [])
    if len(dialogue) % 2:
        raise ValueError(f"{record.get('persona_id')} has odd message count")
    pairs = len(dialogue) // 2
    if max(checkpoints) > pairs:
        raise ValueError(
            f"{record.get('persona_id')} has {pairs} turns, below checkpoint {max(checkpoints)}"
        )
    for index in range(0, len(dialogue), 2):
        if dialogue[index].get("role") != "user" or dialogue[index + 1].get("role") != "assistant":
            raise ValueError(f"{record.get('persona_id')} is not user/assistant paired at message {index}")
    anchors = record.get("eval_anchors", {})
    missing = [checkpoint for checkpoint in checkpoints if str(checkpoint) not in anchors]
    if missing:
        raise ValueError(
            f"{record.get('persona_id')} lacks fixed eval anchors at {missing}; rerun context preparation"
        )


def checkpoint_history(record: dict, checkpoint: int) -> list[dict]:
    history = record["dialogue"][: 2 * checkpoint]
    if not history or history[-1].get("role") != "assistant":
        raise ValueError(f"Checkpoint {checkpoint} does not contain complete pairs")
    return history


def model_system_prompt(record: dict, model: dict) -> str:
    persona = record.get("model_persona") or record.get("persona_info", "")
    prompt = persona if persona.startswith("请你扮演以下人设：") else f"请你扮演以下人设：{persona}"
    suffix = model.get("system_suffix", "").strip()
    return prompt + ("\n\n" + suffix if suffix else "")


def client_for_model(model: dict, args: argparse.Namespace) -> ChatCompletionsClient:
    token_env = model.get("token_env", "DYNS_ROLE_API_TOKEN")
    token = model.get("token") or os.environ.get(token_env, "") or resolve_token(args.api_token, token_env)
    if not token and not args.mock_api and model.get("requires_token", True):
        raise ValueError(f"Missing token for {model['name']}; set {token_env} or DYNS_API_TOKEN")
    return ChatCompletionsClient(
        model.get("api_url", args.api_url),
        token,
        timeout=args.timeout,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
        mock_api=args.mock_api,
    )


def candidate_key(persona_id: str, checkpoint: int, model_name: str) -> str:
    return f"{persona_id}|t{checkpoint}|{model_name}"


def make_candidate_task(record: dict, checkpoint: int, model: dict) -> dict:
    history = checkpoint_history(record, checkpoint)
    return {
        "key": candidate_key(record["persona_id"], checkpoint, model["name"]),
        "persona_id": record["persona_id"],
        "source_index": record.get("source_index"),
        "persona_title": record.get("persona_title"),
        "model_name": model["name"],
        "model_id": model["model"],
        "context_turns": checkpoint,
        "completed_context_pairs": checkpoint,
        "context_message_count": len(history),
        "history": history,
        "fixed_first_user_message": record["eval_anchors"][str(checkpoint)]["user_message"],
        "model_persona": record.get("model_persona") or record.get("persona_info", ""),
        "user_persona": record.get("user_persona") or record.get("user_system_prompt", ""),
        "user_simulator_style": record.get("meta", {}).get("user_simulator_style", "balanced"),
    }


def user_view(dialogue: list[dict]) -> list[dict]:
    return [
        {
            "role": "user" if message["role"] == "assistant" else "assistant",
            "content": message["content"],
        }
        for message in dialogue
    ]


def generate_candidates(
    args: argparse.Namespace,
    records: list[dict],
    models: list[dict],
    output_dir: Path,
) -> list[dict]:
    progress_path = output_dir / "candidate_progress.json"
    completed = {}
    if progress_path.exists() and not args.overwrite_candidates:
        completed = json.loads(progress_path.read_text(encoding="utf-8"))

    model_map = {model["name"]: model for model in models}
    clients = {name: client_for_model(model, args) for name, model in model_map.items()}
    user_token = (
        args.user_api_key
        or os.environ.get("ARK_API_KEY", "")
        or resolve_token(None, "DYNS_USER_API_TOKEN")
    )
    if not user_token and not args.mock_api:
        raise ValueError("Set ARK_API_KEY/DYNS_USER_API_TOKEN or pass --user-api-key")
    user_client = ChatCompletionsClient(
        args.user_api_url,
        user_token,
        timeout=args.timeout,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
        mock_api=args.mock_api,
    )
    tasks = [
        make_candidate_task(record, checkpoint, model)
        for record in records
        for checkpoint in args.checkpoints
        for model in models
    ]
    pending = [task for task in tasks if task["key"] not in completed]
    lock = threading.Lock()

    def worker(task: dict) -> tuple[str, dict]:
        model = model_map[task["model_name"]]
        record = next(item for item in records if item["persona_id"] == task["persona_id"])
        full_dialogue = list(task["history"])
        continued_dialogue = []
        user_prompt = build_user_simulator_prompt(
            task["user_simulator_style"],
            task["model_persona"],
            task["user_persona"],
        )

        for eval_turn in range(1, args.eval_turns + 1):
            if eval_turn == 1:
                user_response = task["fixed_first_user_message"]
            else:
                user_response = user_client.complete(
                    model=args.user_model,
                    messages=[{"role": "system", "content": user_prompt}] + user_view(full_dialogue),
                    temperature=args.user_temperature,
                    max_tokens=args.user_max_tokens,
                )
                user_response = sanitize_user_response(user_response)
            if not user_response or user_response == "null" or len(user_response) > args.max_user_chars:
                raise ValueError(f"Invalid simulated-user response at eval turn {eval_turn}: {user_response!r}")
            user_message = {"role": "user", "content": user_response}
            full_dialogue.append(user_message)
            continued_dialogue.append(user_message)

            response = clients[task["model_name"]].complete(
                model=model["model"],
                messages=[
                    {"role": "system", "content": model_system_prompt(record, model)},
                    *full_dialogue,
                ],
                temperature=float(model.get("temperature", args.candidate_temperature)),
                max_tokens=int(model.get("max_tokens", args.candidate_max_tokens)),
                extra_body=model.get("extra_body"),
            )
            if not response or response == "null" or len(response) > args.max_response_chars:
                raise ValueError(f"Invalid candidate response at eval turn {eval_turn}: {response!r}")
            assistant_message = {"role": "assistant", "content": response}
            full_dialogue.append(assistant_message)
            continued_dialogue.append(assistant_message)

        result = {
            **task,
            "eval_turns": args.eval_turns,
            "continued_dialogue": continued_dialogue,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        }
        result.pop("key", None)
        return task["key"], result

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(worker, task): task for task in pending}
        for count, future in enumerate(as_completed(futures), 1):
            task = futures[future]
            try:
                key, result = future.result()
            except Exception as exc:
                print(f"[candidate failed] {task['key']}: {exc}", flush=True)
                continue
            with lock:
                completed[key] = result
                if count % args.save_every == 0:
                    dump_json(completed, progress_path)
            print(f"[candidate] {key}", flush=True)
    dump_json(completed, progress_path)

    missing = [task["key"] for task in tasks if task["key"] not in completed]
    if missing:
        raise RuntimeError(f"Missing {len(missing)} candidate replies; rerun to resume. First: {missing[:3]}")
    ordered = [completed[task["key"]] for task in tasks]
    dump_json(ordered, output_dir / "candidate_responses.json")
    return ordered


def format_dialogue(history: list[dict]) -> str:
    lines = []
    for index, message in enumerate(history, 1):
        role = "用户" if message.get("role") == "user" else "角色"
        lines.append(f"{index}. {role}: {message.get('content', '')}")
    return "\n".join(lines)


def judge_key(candidate: dict, dimension: str) -> str:
    return f"{candidate_key(candidate['persona_id'], candidate['context_turns'], candidate['model_name'])}|{dimension}"


def evaluate_candidates(
    args: argparse.Namespace,
    candidates: list[dict],
    prompts: dict[str, str],
    output_dir: Path,
) -> list[dict]:
    progress_path = output_dir / "judge_progress.json"
    completed = {}
    if progress_path.exists() and not args.overwrite_judgments:
        completed = json.loads(progress_path.read_text(encoding="utf-8"))

    token = resolve_token(args.judge_api_token, "DYNS_EVAL_API_TOKEN")
    if not token and not args.mock_api:
        raise ValueError("Set DYNS_EVAL_API_TOKEN/DYNS_API_TOKEN or pass --judge-api-token")
    client = ChatCompletionsClient(
        args.judge_api_url,
        token,
        timeout=args.timeout,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
        mock_api=args.mock_api,
    )

    tasks = [(candidate, dimension) for candidate in candidates for dimension in DIMENSIONS]
    pending = [(candidate, dim) for candidate, dim in tasks if judge_key(candidate, dim) not in completed]
    lock = threading.Lock()

    def worker(candidate: dict, dimension: str) -> tuple[str, dict]:
        prompt = prompts[DIMENSIONS[dimension]]
        prompt = (
            prompt.replace("{character_profile}", candidate["model_persona"])
            .replace("{dialogue_history}", format_dialogue(candidate["history"]))
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
            raise ValueError(f"Invalid judge score for {dimension}: {result}")
        return judge_key(candidate, dimension), {
            "score": int(score),
            "reason": str(result.get("reason", "")),
            "raw_response": raw,
        }

    with ThreadPoolExecutor(max_workers=args.judge_workers) as executor:
        futures = {executor.submit(worker, candidate, dim): (candidate, dim) for candidate, dim in pending}
        for count, future in enumerate(as_completed(futures), 1):
            candidate, dimension = futures[future]
            key = judge_key(candidate, dimension)
            try:
                _, result = future.result()
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

    missing = [judge_key(candidate, dim) for candidate, dim in tasks if judge_key(candidate, dim) not in completed]
    if missing:
        raise RuntimeError(f"Missing {len(missing)} judgments; rerun to resume. First: {missing[:3]}")

    results = []
    for candidate in candidates:
        scores = {dim: completed[judge_key(candidate, dim)] for dim in DIMENSIONS}
        overall = statistics.mean(result["score"] for result in scores.values())
        results.append({**candidate, "scores": scores, "overall": overall})
    dump_json(results, output_dir / "evaluated_responses.json")
    return results


def summarize(results: list[dict], output_dir: Path) -> None:
    groups: dict[tuple[str, int], list[dict]] = {}
    for result in results:
        groups.setdefault((result["model_name"], result["context_turns"]), []).append(result)

    rows = []
    for (model_name, checkpoint), items in sorted(groups.items(), key=lambda x: (x[0][1], x[0][0])):
        row: dict[str, Any] = {
            "model": model_name,
            "context_turns": checkpoint,
            "n": len(items),
        }
        for dimension in DIMENSIONS:
            values = [item["scores"][dimension]["score"] for item in items]
            row[f"{dimension}_mean"] = statistics.mean(values)
            row[f"{dimension}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        overall_values = [item["overall"] for item in items]
        row["overall_mean"] = statistics.mean(overall_values)
        row["overall_std"] = statistics.stdev(overall_values) if len(overall_values) > 1 else 0.0
        row["overall_ci95"] = 1.96 * row["overall_std"] / math.sqrt(len(items)) if items else 0.0
        rows.append(row)

    csv_path = output_dir / "summary_by_context.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    eval_turn_values = sorted({int(result["eval_turns"]) for result in results})
    eval_turn_label = ",".join(str(value) for value in eval_turn_values)
    lines = [
        "# Fixed-Doubao Long-Context Evaluation",
        "",
        "Each point uses the same N-turn fixed Doubao history and the same first evaluation user message",
        f"for every candidate model. Each model then interacts with the same Doubao user simulator for {eval_turn_label}",
        "complete user/assistant turns; that continuation is scored with the multi-turn judge prompts.",
        "",
        "| Context turns | Model | N | Overall | Human | Role | Context | Interactive |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['context_turns']} | {row['model']} | {row['n']} | "
            f"{row['overall_mean']:.3f} +/- {row['overall_ci95']:.3f} | "
            f"{row['human_likeness_mean']:.3f} | {row['role_consistency_mean']:.3f} | "
            f"{row['context_consistency_mean']:.3f} | {row['interactive_ability_mean']:.3f} |"
        )
    (output_dir / "summary_by_context.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate candidate models on identical fixed Doubao long contexts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--context-dir", required=True)
    parser.add_argument("--models-config", default=str(DEFAULT_MODELS))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--mode", choices=("generate", "evaluate", "all"), default="all")
    parser.add_argument("--checkpoints", type=parse_int_list, default=(50, 60, 70, 80, 90, 100))
    parser.add_argument("--eval-turns", type=int, default=10, help="Complete user+assistant pairs generated after each fixed context.")

    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--api-token", default=None)
    parser.add_argument("--candidate-temperature", type=float, default=1.0)
    parser.add_argument("--candidate-max-tokens", type=int, default=2048)
    parser.add_argument("--max-response-chars", type=int, default=500)
    parser.add_argument("--user-api-url", default=DEFAULT_USER_API_URL)
    parser.add_argument("--user-api-key", default=None)
    parser.add_argument("--user-model", default="doubao-1-5-pro-32k-character-250715")
    parser.add_argument("--user-temperature", type=float, default=1.0)
    parser.add_argument("--user-max-tokens", type=int, default=200)
    parser.add_argument("--max-user-chars", type=int, default=150)

    parser.add_argument("--judge-prompt-file", default=str(DEFAULT_JUDGE_PROMPTS))
    parser.add_argument("--judge-api-url", default=DEFAULT_API_URL)
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
    parser.add_argument("--overwrite-candidates", action="store_true")
    parser.add_argument("--overwrite-judgments", action="store_true")
    parser.add_argument("--mock-api", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    context_dir = Path(args.context_dir)
    context_file = context_dir / "context_dialogues.jsonl"
    records = load_json_or_jsonl(context_file)
    if not records:
        raise ValueError(f"No contexts in {context_file}")
    for record in records:
        validate_context(record, args.checkpoints)

    models = load_models(Path(args.models_config))
    output_dir = Path(args.output_dir) if args.output_dir else context_dir / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    dump_json({
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "context_dir": str(context_dir.resolve()),
        "models_config": str(Path(args.models_config).resolve()),
        "models": models,
        "checkpoints": list(args.checkpoints),
        "eval_turns": args.eval_turns,
        "protocol": f"fixed Doubao history and fixed first user message; each candidate generates a {args.eval_turns}-turn interactive continuation",
        "turn_definition": "one turn = one user message followed by one assistant message",
        "user_model": args.user_model,
        "judge_model": args.judge_model,
    }, output_dir / "eval_config.json")

    candidate_path = output_dir / "candidate_responses.json"
    if args.mode in {"generate", "all"}:
        candidates = generate_candidates(args, records, models, output_dir)
    else:
        if not candidate_path.exists():
            raise FileNotFoundError(f"Run --mode generate first: {candidate_path}")
        candidates = json.loads(candidate_path.read_text(encoding="utf-8"))

    if args.mode in {"evaluate", "all"}:
        prompts = load_prompt_constants(Path(args.judge_prompt_file))
        results = evaluate_candidates(args, candidates, prompts, output_dir)
        summarize(results, output_dir)
        print(f"Evaluation complete: {output_dir / 'summary_by_context.md'}")
    else:
        print(f"Candidate generation complete: {candidate_path}")


if __name__ == "__main__":
    main()
