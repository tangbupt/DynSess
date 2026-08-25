#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Measure role_consistency as a function of dialogue-history length.

Data comes from ``prepare_doubao_contexts.py`` (a fixed long Doubao history per
persona). This script follows the ``base/run_dynsess_eval.py`` method
(continue-generate -> format -> judge) but:

- swaps in the prepared long histories, truncated at several lengths;
- only scores the ``role_consistency`` dimension;
- aggregates the score grouped by history length so we can see whether the
  role drifts as the preceding context grows.

For each persona and each requested history length L we take the first L turns
(2*L messages) of that persona's fixed history, let the candidate role model
keep chatting with the passive Doubao user simulator for ``--eval-turns`` more
turns, and judge only the continuation for role consistency.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
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
sys.path.insert(0, str(REBUTTAL_DIR))
sys.path.insert(0, str(REBUTTAL_DIR / "different_user_simulation"))

from user_sim_prompts import build_user_simulator_prompt, sanitize_user_response  # noqa: E402

from common import (  # noqa: E402
    ChatCompletionsClient,
    dump_json,
    extract_json_object,
    load_json_or_jsonl,
    resolve_token,
)

# ── Only this dimension is evaluated. ──
DIMENSION = "role_consistency"
JUDGE_PROMPT_CONSTANT = "multi_turn_eval_score_role_consistency"

# ── Endpoints/keys hardcoded from different_user_simulation/run_dynsess_eval.py. ──
# Role model under test = the local vLLM model (ASSISTANT_MODEL="local").
DEFAULT_ROLE_API_URL = "http://localhost:88/v1/chat/completions"
DEFAULT_ROLE_MODEL = "persona_general"
# User simulator = 豆包 via ARK.
DEFAULT_USER_API_URL = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
DEFAULT_USER_MODEL = "doubao-1-5-pro-32k-character-250715"
ARK_API_KEY = os.environ.get("ARK_API_KEY", "")
# Judge = 伏羲 gemini-3-flash-preview.
DEFAULT_JUDGE_API_URL = "https://aigc-api.fuxi.netease.com/v1/chat/completions"
DEFAULT_JUDGE_MODEL = "gemini-3-flash-preview"
EVAL_API_BEARER_TOKEN = os.environ.get("DYNS_EVAL_API_TOKEN", "")
DEFAULT_JUDGE_PROMPTS = REBUTTAL_DIR / "base" / "dynsess_rubrics.py"


def parse_int_list(value: str) -> tuple[int, ...]:
    return tuple(sorted({int(part.strip()) for part in value.split(",") if part.strip()}))


def role_system_prompt(persona: str, suffix: str) -> str:
    processed = persona.removeprefix("请你扮演以下人设：")
    prompt = f"你是一个角色扮演AI助手。请严格按照以下人设进行角色扮演：\n\n{processed}\n\n"
    if suffix.strip():
        prompt = f"{suffix.strip()}\n\n{prompt}"
    return prompt


def role_view(history: list[dict]) -> list[dict]:
    """History as the role model sees it (user = the human)."""
    return [dict(m) for m in history]


def user_view(history: list[dict]) -> list[dict]:
    """History as the user simulator sees it: roles are flipped."""
    flipped = []
    for message in history:
        flipped.append({
            "role": "user" if message["role"] == "assistant" else "assistant",
            "content": message["content"],
        })
    return flipped


def valid_response(text: str, max_chars: int) -> bool:
    return bool(text and text != "null" and text.strip() and len(text) <= max_chars)


def format_dialogue(dialogue: list[dict]) -> str:
    lines = []
    for i, turn in enumerate(dialogue):
        label = "用户" if turn.get("role") == "user" else "角色"
        lines.append(f"{i + 1}. {label}: {turn.get('content', '')}")
    return "\n".join(lines)


def load_role_consistency_prompt(path: Path) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id == JUDGE_PROMPT_CONSTANT:
                value = ast.literal_eval(node.value)
                if isinstance(value, str):
                    return value
    raise ValueError(f"Prompt {JUDGE_PROMPT_CONSTANT} not found in {path}")


# ─────────────────────── stage 1: continuation ───────────────────────

def continue_dialogue(
    args: argparse.Namespace,
    persona_info: str,
    user_persona: str,
    history: list[dict],
    role_client: ChatCompletionsClient,
    user_client: ChatCompletionsClient,
) -> list[dict]:
    """Continue the truncated history for --eval-turns full turns.

    Mirrors base/run_dynsess_eval.py generate_dialogue_continue: alternate
    the user simulator and the role model. history ends on an assistant turn,
    so the user simulator speaks first.
    """
    role_messages = role_view(history)
    role_system = role_system_prompt(persona_info, args.role_system_suffix)
    user_system = build_user_simulator_prompt(args.user_simulator_style, persona_info, user_persona)

    continued: list[dict] = []
    # Each full turn is one user message followed by one assistant message.
    for _ in range(args.eval_turns):
        # 1) user simulator turn
        user_msgs = [{"role": "system", "content": user_system}] + user_view(history + continued)
        user_text = sanitize_user_response(
            user_client.complete(
                model=args.user_model,
                messages=user_msgs,
                temperature=args.user_temperature,
                max_tokens=args.user_max_tokens,
            )
        )
        if not valid_response(user_text, args.max_user_chars):
            break
        continued.append({"role": "user", "content": user_text})

        # 2) role model turn
        role_messages.append({"role": "user", "content": user_text})
        role_text = role_client.complete(
            model=args.role_model,
            messages=[{"role": "system", "content": role_system}] + role_messages,
            temperature=args.role_temperature,
            max_tokens=args.role_max_tokens,
            extra_body=args.role_extra_body,
        )
        if not valid_response(role_text, args.max_role_chars):
            continued.pop()  # drop the dangling user message
            role_messages.pop()
            break
        role_messages.append({"role": "assistant", "content": role_text})
        continued.append({"role": "assistant", "content": role_text})

    return continued


def build_tasks(records: list[dict], lengths: tuple[int, ...], full_history: bool) -> list[dict]:
    tasks = []
    for record in records:
        persona_info = record.get("model_persona") or record.get("persona_info", "")
        user_persona = record.get("user_persona") or record.get("user_system_prompt", "")
        dialogue = record.get("dialogue", [])
        available_turns = len(dialogue) // 2
        persona_id = record.get("persona_id") or record.get("source_index") or "unknown"
        # full_history: evaluate each persona at its own (already varied) length.
        chosen = [available_turns] if full_history else [l for l in lengths if l <= available_turns]
        for length in lengths:
            if not full_history and length > available_turns:
                print(f"[skip] {persona_id} length={length} > available {available_turns}", flush=True)
        for length in chosen:
            tasks.append({
                "persona_id": str(persona_id),
                "history_turns": length,
                "persona_info": persona_info,
                "user_persona": user_persona,
                "history": dialogue[: 2 * length],
            })
    return tasks


def task_key(task: dict) -> str:
    return f"{task['persona_id']}|L{task['history_turns']}"


def generate_candidates(args, tasks, output_dir: Path) -> list[dict]:
    progress_path = output_dir / "generation_progress.json"
    completed: dict[str, dict] = {}
    if progress_path.exists():
        completed = json.loads(progress_path.read_text(encoding="utf-8"))

    role_token = "" if args.mock_api else resolve_token(args.role_api_token, args.role_token_env)
    user_token = args.user_api_key or os.environ.get("ARK_API_KEY", "")
    if not role_token and not args.mock_api and args.role_requires_token:
        raise SystemExit(f"Role model token missing (env {args.role_token_env}).")
    if not user_token and not args.mock_api:
        raise SystemExit("User simulator token missing (env ARK_API_KEY).")

    role_client = ChatCompletionsClient(args.role_api_url, role_token, args.timeout,
                                        args.max_retries, args.retry_delay, args.mock_api)
    user_client = ChatCompletionsClient(args.user_api_url, user_token, args.timeout,
                                        args.max_retries, args.retry_delay, args.mock_api)

    pending = [t for t in tasks if task_key(t) not in completed]
    lock = threading.Lock()

    def worker(task: dict) -> tuple[str, dict]:
        continued = continue_dialogue(args, task["persona_info"], task["user_persona"],
                                      task["history"], role_client, user_client)
        return task_key(task), {
            "persona_id": task["persona_id"],
            "history_turns": task["history_turns"],
            "persona_info": task["persona_info"],
            "history": task["history"],
            "continued_dialogue": continued,
            "continued_turns": len(continued) // 2,
        }

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(worker, t): t for t in pending}
        for count, future in enumerate(as_completed(futures), 1):
            task = futures[future]
            key = task_key(task)
            try:
                _, result = future.result()
            except Exception as exc:  # keep going; unfinished keys just re-run next time
                print(f"[gen failed] {key}: {exc}", flush=True)
                continue
            with lock:
                completed[key] = result
                if count % args.save_every == 0:
                    dump_json(completed, progress_path)
            print(f"[gen] {key} (+{result['continued_turns']} turns)", flush=True)
    dump_json(completed, progress_path)

    candidates = [completed[task_key(t)] for t in tasks if task_key(t) in completed]
    dump_json(candidates, output_dir / "candidate_continuations.json")
    return candidates


# ─────────────────────── stage 2/3: judge role_consistency ───────────────────────

def judge_candidates(args, candidates, prompt_template: str, output_dir: Path) -> list[dict]:
    progress_path = output_dir / "judge_progress.json"
    completed: dict[str, dict] = {}
    if progress_path.exists():
        completed = json.loads(progress_path.read_text(encoding="utf-8"))

    judge_token = "" if args.mock_api else resolve_token(args.judge_api_token, args.judge_token_env)
    if not judge_token and not args.mock_api:
        raise SystemExit(f"Judge token missing (env {args.judge_token_env}).")
    judge_client = ChatCompletionsClient(args.judge_api_url, judge_token, args.timeout,
                                         args.max_retries, args.retry_delay, args.mock_api)

    pending = [c for c in candidates if task_key(c) not in completed]
    lock = threading.Lock()

    def worker(candidate: dict) -> tuple[str, dict]:
        prompt = (
            prompt_template
            .replace("{character_profile}", candidate["persona_info"])
            .replace("{dialogue_history}", format_dialogue(candidate["history"]))
            .replace("{dialogue}", format_dialogue(candidate["continued_dialogue"]))
        )
        raw = judge_client.complete(
            model=args.judge_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=args.judge_temperature,
            max_tokens=args.judge_max_tokens,
        )
        parsed = extract_json_object(raw)
        result = parsed.get(DIMENSION) if parsed else None
        if not isinstance(result, dict):
            raise ValueError(f"Judge response lacks {DIMENSION}: {raw[:300]}")
        score = result.get("score")
        if not isinstance(score, (int, float)) or not 1 <= int(score) <= 5:
            raise ValueError(f"Invalid score: {result}")
        return task_key(candidate), {"score": int(score), "reason": str(result.get("reason", ""))}

    with ThreadPoolExecutor(max_workers=args.judge_workers) as executor:
        futures = {executor.submit(worker, c): c for c in pending}
        for count, future in enumerate(as_completed(futures), 1):
            candidate = futures[future]
            key = task_key(candidate)
            try:
                _, result = future.result()
            except Exception as exc:
                print(f"[judge failed] {key}: {exc}", flush=True)
                continue
            with lock:
                completed[key] = result
                if count % args.save_every == 0:
                    dump_json(completed, progress_path)
            print(f"[judge] {key} score={result['score']}", flush=True)
    dump_json(completed, progress_path)

    results = []
    for candidate in candidates:
        key = task_key(candidate)
        if key not in completed:
            continue
        results.append({
            "persona_id": candidate["persona_id"],
            "history_turns": candidate["history_turns"],
            "continued_turns": candidate["continued_turns"],
            "score": completed[key]["score"],
            "reason": completed[key]["reason"],
        })
    dump_json(results, output_dir / "role_consistency_by_length.json")
    return results


def summarize(results: list[dict], output_dir: Path, bucket_size: int) -> None:
    # Bucket by dialogue-history length, e.g. bucket_size=10 -> 50-59, 60-69, ...
    def bucket_lo(h: int) -> int:
        return (h // bucket_size) * bucket_size

    groups: dict[int, list[int]] = {}
    for row in results:
        groups.setdefault(bucket_lo(row["history_turns"]), []).append(row["score"])

    summary = []
    for lo in sorted(groups):
        scores = groups[lo]
        summary.append({
            "history_bucket": f"{lo}-{lo + bucket_size - 1}",
            "bucket_lo": lo,
            "n": len(scores),
            "role_consistency_mean": round(statistics.mean(scores), 4),
            "role_consistency_std": round(statistics.stdev(scores), 4) if len(scores) > 1 else 0.0,
            "min": min(scores),
            "max": max(scores),
        })

    with (output_dir / "summary_by_length.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0].keys()) if summary else [])
        writer.writeheader()
        writer.writerows(summary)

    lines = [
        "# Role Consistency vs. Dialogue-History Length",
        "",
        f"Each persona keeps its own (varied) history length; only `role_consistency` is judged on the",
        f"continuation. Results bucketed by history length in groups of {bucket_size} turns.",
        "",
        "| History turns (bucket) | N | Role consistency (mean) | Std | Min | Max |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['history_bucket']} | {row['n']} | {row['role_consistency_mean']:.3f} "
            f"| {row['role_consistency_std']:.3f} | {row['min']} | {row['max']} |"
        )
    (output_dir / "summary_by_length.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Role-consistency vs. history-length using the base pipeline method.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--context-dir", required=True,
                   help="Output dir of prepare_doubao_contexts.py (contains context_dialogues.jsonl).")
    p.add_argument("--context-file", default=None,
                   help="Explicit context jsonl; defaults to <context-dir>/context_dialogues.jsonl.")
    p.add_argument("--output-dir", default=None,
                   help="Defaults to <context-dir>/role_consistency_eval.")
    p.add_argument("--lengths", type=parse_int_list, default=(50, 60, 70, 80, 90, 100),
                   help="History lengths (turns) to truncate to; ignored when --full-history.")
    p.add_argument("--full-history", action="store_true", default=True,
                   help="Use each persona's own (already varied) history length as-is.")
    p.add_argument("--no-full-history", dest="full_history", action="store_false")
    p.add_argument("--bucket-size", type=int, default=10,
                   help="Bucket width (turns) for the summary, e.g. 10 -> 50-59, 60-69, ...")
    p.add_argument("--eval-turns", type=int, default=10, help="Continuation turns per task.")
    p.add_argument("--mode", choices=["generate", "evaluate", "all"], default="all")

    p.add_argument("--role-model", default=DEFAULT_ROLE_MODEL)
    p.add_argument("--role-api-url", default=DEFAULT_ROLE_API_URL)
    p.add_argument("--role-api-token", default=None)
    p.add_argument("--role-token-env", default="DYNS_API_TOKEN")
    p.add_argument("--role-requires-token", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--role-temperature", type=float, default=0.7)
    p.add_argument("--role-max-tokens", type=int, default=1024)
    p.add_argument("--role-system-suffix", default="")
    p.add_argument("--role-extra-body", type=json.loads,
                   default={"top_p": 0.95, "chat_template_kwargs": {"enable_thinking": False}})
    p.add_argument("--max-role-chars", type=int, default=500)

    p.add_argument("--user-model", default=DEFAULT_USER_MODEL)
    p.add_argument("--user-api-url", default=DEFAULT_USER_API_URL)
    p.add_argument("--user-api-key", default=ARK_API_KEY)
    p.add_argument("--user-simulator-style", default="passive")
    p.add_argument("--user-temperature", type=float, default=1.0)
    p.add_argument("--user-max-tokens", type=int, default=200)
    p.add_argument("--max-user-chars", type=int, default=150)

    p.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    p.add_argument("--judge-api-url", default=DEFAULT_JUDGE_API_URL)
    p.add_argument("--judge-api-token", default=EVAL_API_BEARER_TOKEN)
    p.add_argument("--judge-token-env", default="DYNS_API_TOKEN")
    p.add_argument("--judge-prompt-file", default=str(DEFAULT_JUDGE_PROMPTS))
    p.add_argument("--judge-temperature", type=float, default=0.7)
    p.add_argument("--judge-max-tokens", type=int, default=4096)

    p.add_argument("--max-workers", type=int, default=8)
    p.add_argument("--judge-workers", type=int, default=8)
    p.add_argument("--save-every", type=int, default=5)
    p.add_argument("--timeout", type=int, default=180)
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--retry-delay", type=float, default=3.0)
    p.add_argument("--mock-api", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    context_dir = Path(args.context_dir)
    context_file = Path(args.context_file) if args.context_file else context_dir / "context_dialogues.jsonl"
    if not context_file.exists():
        raise SystemExit(f"Context file not found: {context_file}")
    output_dir = Path(args.output_dir) if args.output_dir else context_dir / "role_consistency_eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_json_or_jsonl(context_file)
    tasks = build_tasks(records, args.lengths, args.full_history)
    if not tasks:
        raise SystemExit("No (persona, length) tasks; check --lengths against history length.")
    mode_desc = "full per-persona history" if args.full_history else f"lengths {list(args.lengths)}"
    print(f"Prepared {len(tasks)} tasks ({mode_desc}).", flush=True)

    dump_json({
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "context_file": str(context_file.resolve()),
        "lengths": list(args.lengths),
        "eval_turns": args.eval_turns,
        "role_model": args.role_model,
        "user_model": args.user_model,
        "judge_model": args.judge_model,
        "dimension": DIMENSION,
    }, output_dir / "eval_config.json")

    candidate_path = output_dir / "candidate_continuations.json"
    if args.mode in {"generate", "all"}:
        candidates = generate_candidates(args, tasks, output_dir)
    else:
        candidates = json.loads(candidate_path.read_text(encoding="utf-8"))

    if args.mode in {"evaluate", "all"}:
        prompt_template = load_role_consistency_prompt(Path(args.judge_prompt_file))
        results = judge_candidates(args, candidates, prompt_template, output_dir)
        summarize(results, output_dir, args.bucket_size)
        print(f"\nDone: {output_dir / 'summary_by_length.md'}")
    else:
        print(f"Generation complete: {candidate_path}")


if __name__ == "__main__":
    main()
