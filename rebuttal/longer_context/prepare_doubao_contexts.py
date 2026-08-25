#!/usr/bin/env python3
"""Sample personas and construct fixed 100-turn contexts with Doubao."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
REBUTTAL_DIR = HERE.parent
DYNS_DIR = REBUTTAL_DIR.parent
sys.path.insert(0, str(REBUTTAL_DIR))

from user_simulator_prompts import (  # noqa: E402
    available_styles,
    build_user_simulator_prompt,
    sanitize_user_response,
)

from common import (  # noqa: E402
    ChatCompletionsClient,
    dump_json,
    dump_jsonl,
    load_json_or_jsonl,
    persona_title,
    resolve_token,
    seeded_greeting,
)

DEFAULT_INPUT = HERE / "personas_100_selected.jsonl"
DEFAULT_API_URL = "https://aigc-api.fuxi.netease.com/v1/chat/completions"
DEFAULT_USER_API_URL = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
DEFAULT_CONTEXT_MODEL = "Doubao-1.5-pro-32k-character-250715"
DEFAULT_USER_MODEL = "doubao-1-5-pro-32k-character-250715"
DEFAULT_CHECKPOINTS = (50, 60, 70, 80, 90, 100)


def parse_int_list(value: str) -> tuple[int, ...]:
    values = tuple(sorted({int(part.strip()) for part in value.split(",") if part.strip()}))
    if not values or values[0] < 1:
        raise argparse.ArgumentTypeError("Expected comma-separated positive integers")
    return values


def select_personas(input_path: Path, sample_size: int, seed: int) -> list[dict]:
    source = load_json_or_jsonl(input_path)
    if sample_size > len(source):
        raise ValueError(f"Cannot sample {sample_size} records from {len(source)} personas")
    indices = random.Random(seed).sample(range(len(source)), sample_size)
    selected = []
    for rank, source_index in enumerate(indices):
        item = source[source_index]
        role_persona = item.get("model_persona") or item.get("persona_info")
        user_persona = item.get("user_persona") or item.get("user_system_prompt")
        if not role_persona or not user_persona:
            raise ValueError(f"Persona {source_index} lacks role or user persona fields")
        selected.append({
            "persona_id": f"p{source_index:03d}",
            "sample_rank": rank,
            "source_index": source_index,
            "persona_title": persona_title(role_persona),
            "model_persona": role_persona,
            "user_persona": user_persona,
        })
    return selected


def role_system_prompt(persona: str, suffix: str) -> str:
    prompt = persona if persona.startswith("请你扮演以下人设：") else f"请你扮演以下人设：{persona}"
    if suffix.strip():
        prompt += "\n\n" + suffix.strip()
    return prompt


def user_messages_from_dialogue(dialogue: list[dict]) -> list[dict]:
    # The simulated user sees the role model as its user, matching the SFT code.
    converted = []
    for message in dialogue:
        converted.append({
            "role": "user" if message["role"] == "assistant" else "assistant",
            "content": message["content"],
        })
    return converted


def valid_response(text: str, max_chars: int) -> bool:
    return bool(text and text != "null" and text.strip() and len(text) <= max_chars)


def make_user_message(
    client: ChatCompletionsClient,
    args: argparse.Namespace,
    item: dict,
    dialogue: list[dict],
    turn_index: int,
    memory_probe: bool = False,
) -> str:
    prompt = build_user_simulator_prompt(
        args.user_simulator_style,
        item["model_persona"],
        item["user_persona"],
    )
    if memory_probe:
        prompt += (
            "\n\n本轮是长上下文评测的起始消息。请像真实用户一样，自然地追问或呼应至少20轮以前"
            "出现过的一个具体信息、约定、事件或态度。不要说明你在测试记忆，也不要引用最近两轮即可回答的内容。"
        )
    response = client.complete(
        model=args.user_model,
        messages=[{"role": "system", "content": prompt}] + user_messages_from_dialogue(dialogue),
        temperature=args.user_temperature,
        max_tokens=args.user_max_tokens,
    )
    response = sanitize_user_response(response)
    if not valid_response(response, args.max_user_chars):
        raise ValueError(f"Invalid user response at turn {turn_index}: {response!r}")
    return response


def generate_one(
    args: argparse.Namespace,
    item: dict,
    output_dir: Path,
    context_client: ChatCompletionsClient,
    user_client: ChatCompletionsClient,
) -> dict:
    record_path = output_dir / "records" / f"{item['persona_id']}.json"
    if record_path.exists() and not args.overwrite:
        record = json.loads(record_path.read_text(encoding="utf-8"))
        anchors = record.get("eval_anchors", {})
        has_anchors = all(str(checkpoint) in anchors for checkpoint in args.checkpoints)
        if (
            record.get("meta", {}).get("complete")
            and record.get("meta", {}).get("target_turns") == args.target_turns
            and has_anchors
        ):
            return record
    else:
        record = {}

    dialogue = list(record.get("dialogue", []))
    if len(dialogue) % 2:
        raise ValueError(f"Cannot resume odd message count in {record_path}")
    completed_turns = len(dialogue) // 2
    system_prompt = role_system_prompt(item["model_persona"], args.role_system_suffix)

    meta = {
        "complete": False,
        "turn_definition": "one turn = one user message followed by one assistant message",
        "target_turns": args.target_turns,
        "completed_turns": completed_turns,
        "context_model": args.context_model,
        "user_model": args.user_model,
        "user_simulator_style": args.user_simulator_style,
        "memory_probe": args.memory_probe,
        "probe_checkpoints": list(args.checkpoints) if args.memory_probe else [],
        "seed": args.seed,
    }

    for turn_index in range(completed_turns + 1, args.target_turns + 1):
        if turn_index == 1:
            user_text = seeded_greeting(args.seed, item["source_index"])
        else:
            user_text = make_user_message(user_client, args, item, dialogue, turn_index)
        dialogue.append({"role": "user", "content": user_text})

        assistant_text = context_client.complete(
            model=args.context_model,
            messages=[{"role": "system", "content": system_prompt}] + dialogue,
            temperature=args.context_temperature,
            max_tokens=args.context_max_tokens,
            extra_body=args.context_extra_body,
        )
        if not valid_response(assistant_text, args.max_role_chars):
            dialogue.pop()
            raise ValueError(f"Invalid context-model response at turn {turn_index}: {assistant_text!r}")
        dialogue.append({"role": "assistant", "content": assistant_text})

        meta["completed_turns"] = turn_index
        if turn_index % args.save_every == 0 or turn_index == args.target_turns:
            dump_json({**item, "meta": meta, "dialogue": dialogue}, record_path)

    eval_anchors = dict(record.get("eval_anchors", {}))
    for checkpoint in args.checkpoints:
        key = str(checkpoint)
        if key in eval_anchors and not args.overwrite:
            continue
        prefix = dialogue[: 2 * checkpoint]
        anchor = make_user_message(
            user_client,
            args,
            item,
            prefix,
            checkpoint + 1,
            memory_probe=args.memory_probe,
        )
        eval_anchors[key] = {
            "history_turns": checkpoint,
            "history_message_count": len(prefix),
            "user_message": anchor,
            "memory_probe": args.memory_probe,
        }
        dump_json(
            {**item, "meta": meta, "dialogue": dialogue, "eval_anchors": eval_anchors},
            record_path,
        )

    meta["complete"] = True
    record = {**item, "meta": meta, "dialogue": dialogue, "eval_anchors": eval_anchors}
    dump_json(record, record_path)
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build fixed long contexts with Doubao for controlled model comparison.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--sample-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260529)
    parser.add_argument("--target-turns", type=int, default=100)
    parser.add_argument("--checkpoints", type=parse_int_list, default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--user-simulator-style", choices=available_styles(), default="passive")
    parser.add_argument("--memory-probe", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--context-api-url", default=DEFAULT_API_URL)
    parser.add_argument("--context-api-token", default=None)
    parser.add_argument("--context-model", default=DEFAULT_CONTEXT_MODEL)
    parser.add_argument("--user-api-url", default=DEFAULT_USER_API_URL)
    parser.add_argument("--user-api-key", default=None)
    parser.add_argument("--user-model", default=DEFAULT_USER_MODEL)
    parser.add_argument("--context-temperature", type=float, default=1.0)
    parser.add_argument("--user-temperature", type=float, default=1.0)
    parser.add_argument("--context-max-tokens", type=int, default=2048)
    parser.add_argument("--user-max-tokens", type=int, default=200)
    parser.add_argument("--max-role-chars", type=int, default=500)
    parser.add_argument("--max-user-chars", type=int, default=150)
    parser.add_argument("--role-system-suffix", default="")
    parser.add_argument("--context-extra-body", type=json.loads, default={})

    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=3.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--mock-api", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if max(args.checkpoints) > args.target_turns:
        raise ValueError("All checkpoints must be <= --target-turns")
    context_token = resolve_token(args.context_api_token, "DYNS_CONTEXT_API_TOKEN")
    user_token = (
        args.user_api_key
        or os.environ.get("ARK_API_KEY", "")
        or resolve_token(None, "DYNS_USER_API_TOKEN")
    )
    if not context_token and not args.mock_api:
        raise ValueError("Set DYNS_API_TOKEN/DYNS_CONTEXT_API_TOKEN or pass --context-api-token")
    if not user_token and not args.mock_api:
        raise ValueError("Set ARK_API_KEY/DYNS_USER_API_TOKEN or pass --user-api-key")

    run_name = f"doubao_fixed_{args.sample_size}p_{args.target_turns}t_seed{args.seed}"
    output_dir = Path(args.output_dir) if args.output_dir else HERE / "outputs" / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    selection_path = output_dir / "selected_personas.jsonl"
    if selection_path.exists() and not args.overwrite:
        selected = load_json_or_jsonl(selection_path)
        if len(selected) != args.sample_size:
            raise ValueError(
                f"Existing selection has {len(selected)} personas, expected {args.sample_size}; "
                "use a new --output-dir or --overwrite"
            )
    else:
        selected = select_personas(Path(args.input), args.sample_size, args.seed)
        dump_jsonl(selected, selection_path)

    dump_json({
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input": str(Path(args.input).resolve()),
        "sample_size": args.sample_size,
        "seed": args.seed,
        "target_turns": args.target_turns,
        "checkpoints": list(args.checkpoints),
        "context_model": args.context_model,
        "user_model": args.user_model,
        "user_simulator_style": args.user_simulator_style,
        "memory_probe": args.memory_probe,
        "turn_definition": "one turn = one user message followed by one assistant message",
    }, output_dir / "run_config.json")

    context_client = ChatCompletionsClient(
        args.context_api_url,
        context_token,
        timeout=args.timeout,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
        mock_api=args.mock_api,
    )
    user_client = ChatCompletionsClient(
        args.user_api_url,
        user_token,
        timeout=args.timeout,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
        mock_api=args.mock_api,
    )
    results, failures = {}, {}
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(generate_one, args, item, output_dir, context_client, user_client): item
            for item in selected
        }
        for future in as_completed(futures):
            item = futures[future]
            try:
                record = future.result()
                results[item["persona_id"]] = record
                print(f"[done] {item['persona_id']} {item['persona_title']} ({len(record['dialogue']) // 2} turns)", flush=True)
            except Exception as exc:
                failures[item["persona_id"]] = str(exc)
                print(f"[failed] {item['persona_id']} {item['persona_title']}: {exc}", flush=True)

    ordered = [results[item["persona_id"]] for item in selected if item["persona_id"] in results]
    dump_jsonl(ordered, output_dir / "context_dialogues.jsonl")
    dump_json({"completed": len(ordered), "failed": failures}, output_dir / "status.json")
    print(f"Completed {len(ordered)}/{len(selected)} contexts. Output: {output_dir}")
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
