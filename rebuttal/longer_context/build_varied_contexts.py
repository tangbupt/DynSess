#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build varied-length Doubao histories (no config; values hardcoded).

Two sets, both constructed with Doubao (via ARK) and the passive user simulator,
reusing prepare_doubao_contexts.generate_one:

  Set A: the 20 sampled personas, each a RANDOM 50-100 turns.
  Set B: a SEPARATE 10 personas, each 200 turns.

No checkpoints/anchors are produced (that experiment was dropped). Run it and it
writes context_dialogues.jsonl for each set under outputs/.
"""

from __future__ import annotations

import copy
import random
import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import prepare_doubao_contexts as P  # noqa: E402
from common import ChatCompletionsClient, dump_json, dump_jsonl, load_json_or_jsonl, persona_title  # noqa: E402

# ── Hardcoded endpoints/keys (Doubao via ARK, from different_user_simulation). ──
ARK_URL = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
ARK_KEY = os.environ.get("ARK_API_KEY", "")
DOUBAO_MODEL = "doubao-1-5-pro-32k-character-250715"
INPUT_FILE = HERE / "personas_100_selected.jsonl"

SEED = 20260529
SET_A_SIZE = 100           # random 50-100 turns each (full persona pool)
SET_A_MIN_TURNS = 50
SET_A_MAX_TURNS = 100
SET_B_SIZE = 10            # fixed 200 turns each (extreme reference; may overlap Set A personas)
SET_B_TURNS = 200

# generous per-reply cap so a single long Doubao reply never fails a whole
# 200-turn build; keep replies bounded for cost.
MAX_ROLE_CHARS = 2000
CONTEXT_MAX_TOKENS = 512
MAX_WORKERS = 10           # personas built concurrently (turns within a persona are serial)


def base_args(mock: bool):
    """Reuse prepare_doubao_contexts' arg defaults, forced onto the ARK Doubao."""
    argv = [
        "build_varied_contexts.py",
        "--input", str(INPUT_FILE),
        "--context-api-url", ARK_URL,
        "--context-api-token", ARK_KEY,
        "--context-model", DOUBAO_MODEL,
        "--user-api-url", ARK_URL,
        "--user-api-key", ARK_KEY,
        "--user-model", DOUBAO_MODEL,
        "--user-simulator-style", "passive",
        "--context-max-tokens", str(CONTEXT_MAX_TOKENS),
        "--max-role-chars", str(MAX_ROLE_CHARS),
        "--seed", str(SEED),
    ]
    if mock:
        argv.append("--mock-api")
    saved = sys.argv
    sys.argv = argv
    try:
        return P.parse_args()
    finally:
        sys.argv = saved


def make_item(rows, index: int, rank: int) -> dict:
    row = rows[index]
    model_persona = row.get("model_persona") or row.get("persona_info")
    user_persona = row.get("user_persona") or row.get("user_system_prompt")
    return {
        "persona_id": f"p{index:03d}",
        "sample_rank": rank,
        "source_index": index,
        "persona_title": persona_title(model_persona),
        "model_persona": model_persona,
        "user_persona": user_persona,
    }


def build_set(name, items, output_dir, args, context_client, user_client):
    output_dir.mkdir(parents=True, exist_ok=True)
    dump_jsonl(items, output_dir / "selected_personas.jsonl")
    print(f"\n=== building {name}: {len(items)} personas -> {output_dir} ===", flush=True)

    results, failures = [], {}

    def worker(item):
        per_args = copy.copy(args)
        per_args.target_turns = item["target_turns"]
        per_args.checkpoints = ()  # no anchors
        return item, P.generate_one(per_args, item, output_dir, context_client, user_client)

    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(items))) as executor:
        futures = {executor.submit(worker, item): item for item in items}
        for future in as_completed(futures):
            item = futures[future]
            try:
                _, record = future.result()
                results.append(record)
                print(f"[done] {item['persona_id']} {item['persona_title']} "
                      f"({len(record['dialogue']) // 2}/{item['target_turns']} turns)", flush=True)
            except Exception as exc:
                failures[item["persona_id"]] = str(exc)
                print(f"[failed] {item['persona_id']} {item['persona_title']}: {exc}", flush=True)

    results.sort(key=lambda r: r.get("sample_rank", 0))

    dump_jsonl(results, output_dir / "context_dialogues.jsonl")
    dump_json({"completed": len(results), "failed": failures,
               "turns_per_persona": {r["persona_id"]: len(r["dialogue"]) // 2 for r in results}},
              output_dir / "status.json")
    return results, failures


def main() -> None:
    mock = "--mock-api" in sys.argv
    args = base_args(mock)

    token = "" if mock else ARK_KEY
    context_client = ChatCompletionsClient(ARK_URL, token, args.timeout, args.max_retries,
                                           args.retry_delay, mock)
    user_client = ChatCompletionsClient(ARK_URL, token, args.timeout, args.max_retries,
                                         args.retry_delay, mock)

    rows = load_json_or_jsonl(INPUT_FILE)
    n = len(rows)

    # Stable selection: keep the original base-20 and Set B unchanged; extend
    # Set A up to SET_A_SIZE (== full pool -> all personas).
    base20 = random.Random(SEED).sample(range(n), 20)
    remaining_b = [i for i in range(n) if i not in set(base20)]
    idx_b = random.Random(SEED + 777).sample(remaining_b, SET_B_SIZE)  # unchanged Set B
    if SET_A_SIZE >= n:
        idx_a = base20 + [i for i in range(n) if i not in set(base20)]  # all personas
    else:
        remaining_a = [i for i in remaining_b if i not in set(idx_b)]
        extra = random.Random(SEED + 555).sample(remaining_a, SET_A_SIZE - 20)
        idx_a = base20 + extra

    # Set A: personas each a random 50-100 turns.
    set_a = []
    for rank, i in enumerate(idx_a):
        item = make_item(rows, i, rank)
        item["target_turns"] = random.Random(SEED + i * 1009).randint(SET_A_MIN_TURNS, SET_A_MAX_TURNS)
        set_a.append(item)

    # Set B: 10 personas, each 200 turns (may overlap Set A when Set A is the full pool).
    set_b = []
    for rank, i in enumerate(idx_b):
        item = make_item(rows, i, rank)
        item["target_turns"] = SET_B_TURNS
        set_b.append(item)

    out_root = HERE / "outputs"
    a_dir = out_root / f"varied_{SET_A_SIZE}p_{SET_A_MIN_TURNS}to{SET_A_MAX_TURNS}t_seed{SEED}"
    b_dir = out_root / f"varied_{SET_B_SIZE}p_{SET_B_TURNS}t_seed{SEED}"

    res_a, fail_a = build_set("SET A (20p x random 50-100 turns)", set_a, a_dir, args,
                              context_client, user_client)
    res_b, fail_b = build_set("SET B (10p x 200 turns)", set_b, b_dir, args,
                              context_client, user_client)

    print("\n" + "=" * 60)
    print(f"SET A: {len(res_a)}/{len(set_a)} done -> {a_dir}/context_dialogues.jsonl")
    for r in res_a:
        print(f"   {r['persona_id']} {r['persona_title']}: {len(r['dialogue']) // 2} turns")
    print(f"SET B: {len(res_b)}/{len(set_b)} done -> {b_dir}/context_dialogues.jsonl")
    for r in res_b:
        print(f"   {r['persona_id']} {r['persona_title']}: {len(r['dialogue']) // 2} turns")
    if fail_a or fail_b:
        print(f"failures: A={fail_a} B={fail_b}")


if __name__ == "__main__":
    main()
