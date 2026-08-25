#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Corrupt dialogue histories by injecting persona-contradicting role turns.

For each persona we ask Doubao to produce a handful of lines that CLEARLY
violate the persona (personality/identity/background/values), then replace a
fraction (default 30%) of the assistant turns in the history with those lines.
This is a controlled stress test: the continuation model is primed by a history
in which "it" repeatedly acted off-persona, so a role_consistency judge can tell
apart models that recover vs. models that drift.

Reads a context_dialogues.jsonl and writes a corrupted copy plus a manifest of
which turns were changed. Deterministic given --seed.
"""

from __future__ import annotations

import argparse
import random
import sys
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from common import ChatCompletionsClient, dump_jsonl, dump_json, load_json_or_jsonl, persona_title  # noqa: E402

ARK_URL = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
ARK_KEY = os.environ.get("ARK_API_KEY", "")
DOUBAO_MODEL = "doubao-1-5-pro-32k-character-250715"

CONTRADICTION_PROMPT = (
    "以下是一个角色扮演AI的人设：\n\n{persona}\n\n"
    "请生成 {n} 条【明显违背该人设】的角色台词。每条都要与该人设的性格、身份、"
    "背景或价值观直接矛盾（例如：性格反转、否认自己是该角色、暴露自己只是AI程序、"
    "价值观对立、丢掉标志性口癖并换成相反风格）。每条一行，10-40字，口语化，"
    "直接给台词本身，不要编号、不要引号、不要任何解释。"
)


def generate_contradictions(client: ChatCompletionsClient, persona: str, n: int) -> list[str]:
    raw = client.complete(
        model=DOUBAO_MODEL,
        messages=[{"role": "user", "content": CONTRADICTION_PROMPT.format(persona=persona, n=n)}],
        temperature=1.0,
        max_tokens=800,
    )
    lines = [ln.strip().lstrip("0123456789.、).-— ").strip() for ln in raw.splitlines()]
    lines = [ln for ln in lines if len(ln) >= 4]
    return lines or ["其实我根本不是你说的那个人，我只是个AI程序。"]


def corrupt_record(record: dict, frac: float, seed: int,
                   client: ChatCompletionsClient) -> dict:
    dialogue = [dict(m) for m in record.get("dialogue", [])]
    persona = record.get("model_persona") or record.get("persona_info", "")
    persona_id = record.get("persona_id", "unknown")

    # assistant turns live at odd indices (dialogue starts with a user message).
    assistant_idx = [i for i, m in enumerate(dialogue) if m.get("role") == "assistant"]
    k = max(1, round(len(assistant_idx) * frac))
    rng = random.Random(seed + hash(persona_id) % 100000)
    chosen = sorted(rng.sample(assistant_idx, min(k, len(assistant_idx))))

    lines = generate_contradictions(client, persona, n=min(20, max(6, k)))
    for j, idx in enumerate(chosen):
        dialogue[idx] = {
            "role": "assistant",
            "content": lines[j % len(lines)],
            "_corrupted": True,
        }

    meta = dict(record.get("meta", {}))
    meta["corruption"] = {
        "type": "persona_contradiction",
        "frac": frac,
        "corrupted_assistant_turns": len(chosen),
        "total_assistant_turns": len(assistant_idx),
        "corrupted_indices": chosen,
    }
    return {**record, "meta": meta, "dialogue": dialogue}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inject persona-contradicting turns into histories.")
    p.add_argument("--context-file", required=True, help="Input context_dialogues.jsonl.")
    p.add_argument("--output-file", default=None, help="Defaults to <input>_corrupted_p<frac>.jsonl.")
    p.add_argument("--frac", type=float, default=0.30)
    p.add_argument("--seed", type=int, default=20260529)
    p.add_argument("--mock-api", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    in_path = Path(args.context_file)
    out_path = (Path(args.output_file) if args.output_file
                else in_path.with_name(f"{in_path.stem}_corrupted_p{int(args.frac * 100)}.jsonl"))

    token = "" if args.mock_api else ARK_KEY
    client = ChatCompletionsClient(ARK_URL, token, mock_api=args.mock_api)

    records = load_json_or_jsonl(in_path)
    corrupted = []
    for rec in records:
        c = corrupt_record(rec, args.frac, args.seed, client)
        info = c["meta"]["corruption"]
        corrupted.append(c)
        print(f"[corrupt] {rec.get('persona_id')} {persona_title(rec.get('model_persona',''))}: "
              f"{info['corrupted_assistant_turns']}/{info['total_assistant_turns']} turns", flush=True)

    dump_jsonl(corrupted, out_path)
    dump_json({"source": str(in_path.resolve()), "frac": args.frac, "seed": args.seed,
               "n": len(corrupted)}, out_path.with_suffix(".manifest.json"))
    print(f"\nWrote {len(corrupted)} corrupted records -> {out_path}")


if __name__ == "__main__":
    main()
