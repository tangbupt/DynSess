#!/usr/bin/env python3
"""Segmented best-of-N rollout ablation.

Unlike ``run_ablation.py`` (which pre-generates a static candidate pool and
combines whole trajectories offline), this script does *online segmented
branching*: the 50-turn evaluation is split into 5 segments of 10 turns; at
each segment the competing models continue from the shared best-so-far prefix,
a 4-dimension judge scores each segment, and the winning segment becomes the
new shared prefix.

Four strategies are compared:
  1. random_1        : one model runs the whole 50 turns straight through.
  2. same_model_2    : that model runs the whole 50 turns twice, keep the
                       higher-overall full trajectory.
  3. random_2_seg    : 2 random models, segment-by-segment branching.
  4. all_5_seg       : all 5 models, segment-by-segment branching.

Every strategy yields one final 50-turn trajectory that is re-scored as a whole
so the strategies share a comparable headline metric.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import statistics
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

HERE = Path(__file__).resolve().parent
REBUTTAL_DIR = HERE.parent
DYNS_DIR = REBUTTAL_DIR.parent
LONGER_CONTEXT_DIR = REBUTTAL_DIR / "longer_context"
sys.path.insert(0, str(REBUTTAL_DIR))
sys.path.insert(0, str(LONGER_CONTEXT_DIR))

from common import (  # noqa: E402
    dump_json,
    dump_jsonl,
    extract_json_object,
    load_json_or_jsonl,
)
from evaluate_fixed_contexts import (  # noqa: E402
    DIMENSIONS,
    format_dialogue,
    load_models,
    load_prompt_constants,
)
from user_simulator_prompts import build_user_simulator_prompt, sanitize_user_response  # noqa: E402
from model_api import llm_call_eval, llm_call_role, llm_call_user  # noqa: E402

DEFAULT_INPUT = LONGER_CONTEXT_DIR / "outputs" / "varied_20p_50to100t_seed20260529" / "context_dialogues.jsonl"
DEFAULT_MODELS = HERE / "models.json"
DEFAULT_JUDGE_PROMPTS = REBUTTAL_DIR / "base" / "dynsess_rubrics.py"


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #
def _persona_fields(item: dict) -> tuple[str, str]:
    """Return (role_persona, user_system_prompt) from a context record."""
    persona = item.get("model_persona") or item.get("persona_info") or ""
    user_prompt = item.get("user_persona") or item.get("user_system_prompt") or ""
    return persona, user_prompt


def select_histories(input_path: Path, sample_size: int, seed: int) -> list[dict]:
    source = load_json_or_jsonl(input_path)
    eligible = []
    for index, item in enumerate(source):
        dialogue = item.get("dialogue", [])
        if not dialogue or dialogue[-1].get("role") != "assistant":
            continue
        if any(
            dialogue[i].get("role") != ("user" if i % 2 == 0 else "assistant")
            for i in range(len(dialogue))
        ):
            continue
        eligible.append(index)
    if sample_size > len(eligible):
        raise ValueError(f"Cannot sample {sample_size} from {len(eligible)} eligible histories")
    indices = random.Random(seed).sample(eligible, sample_size)
    selected = []
    for rank, source_index in enumerate(indices):
        item = source[source_index]
        persona, user_prompt = _persona_fields(item)
        selected.append({
            "history_id": item.get("persona_id") or f"h{source_index:03d}",
            "sample_rank": rank,
            "source_index": source_index,
            "history_turns": len(item["dialogue"]) // 2,
            "persona_info": persona,
            "user_system_prompt": user_prompt,
            "dialogue": item["dialogue"],
        })
    return selected


def prepare_selection(args: argparse.Namespace, output_dir: Path) -> list[dict]:
    path = output_dir / "selected_histories.jsonl"
    if path.exists() and not args.overwrite_selection:
        selected = load_json_or_jsonl(path)
        if len(selected) != args.sample_size:
            raise ValueError(f"Existing selection has {len(selected)} records, expected {args.sample_size}")
        return selected
    selected = select_histories(Path(args.input), args.sample_size, args.seed)
    dump_jsonl(selected, path)
    return selected


# --------------------------------------------------------------------------- #
# Clients / prompts
# --------------------------------------------------------------------------- #
def model_system_prompt(history: dict, model: dict) -> str:
    persona = history["persona_info"]
    prompt = persona if persona.startswith("请你扮演以下人设：") else f"请你扮演以下人设：{persona}"
    suffix = model.get("system_suffix", "").strip()
    return prompt + ("\n\n【重要】" + suffix if suffix else "")


def user_view(dialogue: list[dict]) -> list[dict]:
    return [
        {"role": "user" if m["role"] == "assistant" else "assistant", "content": m["content"]}
        for m in dialogue
    ]


def valid_text(text: str, max_chars: int) -> bool:
    return bool(text and text != "null" and text.strip() and len(text) <= max_chars)


# --------------------------------------------------------------------------- #
# Core rollout / judge
# --------------------------------------------------------------------------- #
class Engine:
    """Bundles the three call paths plus judge prompts and per-history context."""

    def __init__(self, args: argparse.Namespace, models: list[dict]):
        self.args = args
        self.judge_prompts = load_prompt_constants(Path(args.judge_prompt_file))

    # ---- thin wrappers around the copied model_api calls (mock-aware) ---- #
    def _user_call(self, messages: list[dict]) -> str:
        if self.args.mock_api:
            return "那你还记得我们之前说过的细节吗？"
        return llm_call_user(
            messages, temperature=self.args.user_temperature, max_tokens=self.args.user_max_tokens
        )

    def _role_call(self, messages: list[dict], model: dict) -> str:
        if self.args.mock_api:
            return f"我记得。我们接着聊你刚才提到的事情吧。({model['name']})"
        return llm_call_role(
            messages,
            model_name=model["model"],
            temperature=float(model.get("temperature", self.args.role_temperature)),
            max_tokens=int(model.get("max_tokens", self.args.role_max_tokens)),
            extra_body=model.get("extra_body"),
        )

    def _eval_call(self, messages: list[dict], dimension: str) -> str:
        if self.args.mock_api:
            return json.dumps({dimension: {"reason": "mock result", "score": 4}})
        return llm_call_eval(
            messages, model_name=self.args.judge_model,
            temperature=self.args.judge_temperature, max_tokens=self.args.judge_max_tokens,
        )

    # ---- user anchor (fixed shared first eval message) ---- #
    def build_anchor(self, history: dict) -> str:
        prompt = build_user_simulator_prompt(
            self.args.user_simulator_style, history["persona_info"], history["user_system_prompt"]
        )
        resp = self._user_call([{"role": "system", "content": prompt}] + user_view(history["dialogue"]))
        resp = sanitize_user_response(resp)
        if not valid_text(resp, self.args.max_user_chars):
            raise ValueError(f"Invalid fixed anchor for {history['history_id']}: {resp!r}")
        return resp

    # ---- continue a dialogue for num_turns turns with one model ---- #
    def continue_segment(
        self, history: dict, model: dict, prefix: list[dict], num_turns: int, first_user: Optional[str]
    ) -> list[dict]:
        args = self.args
        sim_prompt = build_user_simulator_prompt(
            args.user_simulator_style, history["persona_info"], history["user_system_prompt"]
        )
        sys_prompt = model_system_prompt(history, model)
        working = history["dialogue"] + prefix
        new_msgs: list[dict] = []
        for turn in range(num_turns):
            if turn == 0 and first_user is not None:
                user_resp = first_user
            else:
                user_resp = self._user_call(
                    [{"role": "system", "content": sim_prompt}] + user_view(working)
                )
                user_resp = sanitize_user_response(user_resp)
            if not valid_text(user_resp, args.max_user_chars):
                raise ValueError(f"Invalid user response ({history['history_id']}): {user_resp!r}")
            um = {"role": "user", "content": user_resp}
            working.append(um)
            new_msgs.append(um)

            role_resp = self._role_call([{"role": "system", "content": sys_prompt}, *working], model)
            if not valid_text(role_resp, args.max_role_chars):
                raise ValueError(f"Invalid role response ({model['name']}): {role_resp!r}")
            rm = {"role": "assistant", "content": role_resp}
            working.append(rm)
            new_msgs.append(rm)
        return new_msgs

    # ---- judge a stretch of continued dialogue (dimensions run in parallel) ---- #
    def judge(self, history: dict, context_prefix: list[dict], target: list[dict]) -> dict:
        dialogue_history = format_dialogue(history["dialogue"] + context_prefix)
        dialogue = format_dialogue(target)

        def score_dimension(item: tuple[str, str]) -> tuple[str, dict]:
            dimension, const = item
            prompt = (
                self.judge_prompts[const]
                .replace("{character_profile}", history["persona_info"])
                .replace("{dialogue_history}", dialogue_history)
                .replace("{dialogue}", dialogue)
            )
            raw = self._eval_call([{"role": "user", "content": prompt}], dimension)
            parsed = extract_json_object(raw)
            result = parsed.get(dimension) if parsed else None
            if not isinstance(result, dict):
                raise ValueError(f"Judge response lacks {dimension}: {raw[:300]}")
            score = result.get("score")
            if not isinstance(score, (int, float)) or not 1 <= int(score) <= 5:
                raise ValueError(f"Invalid score for {dimension}: {result}")
            return dimension, {"score": int(score), "reason": str(result.get("reason", ""))}

        scores: dict[str, dict] = {}
        with ThreadPoolExecutor(max_workers=len(DIMENSIONS)) as executor:
            for dimension, value in executor.map(score_dimension, DIMENSIONS.items()):
                scores[dimension] = value
        overall = statistics.mean(v["score"] for v in scores.values())
        return {"scores": scores, "overall": overall}


# --------------------------------------------------------------------------- #
# Strategies
# --------------------------------------------------------------------------- #
def run_full(engine: Engine, history: dict, model: dict, anchor: str, num_segments: int, seg_turns: int) -> list[dict]:
    """Run one uninterrupted full trajectory with a single model."""
    trajectory: list[dict] = []
    for seg in range(num_segments):
        first_user = anchor if seg == 0 else None
        seg_msgs = engine.continue_segment(history, model, trajectory, seg_turns, first_user)
        trajectory += seg_msgs
    return trajectory


def run_segmented(
    engine: Engine, history: dict, competitors: list[dict], anchor: str,
    num_segments: int, seg_turns: int, workers: int = 5,
) -> tuple[list[dict], list[dict]]:
    """Segment-by-segment branching. Returns (final trajectory, per-segment log).

    Segments are sequential (segment N+1 branches from the segment-N winner), but
    the competing models within a segment run their rollout + judge concurrently.
    """
    shared: list[dict] = []
    seg_log: list[dict] = []
    for seg in range(num_segments):
        first_user = anchor if seg == 0 else None

        def evaluate(model: dict) -> dict:
            seg_msgs = engine.continue_segment(history, model, shared, seg_turns, first_user)
            verdict = engine.judge(history, shared, seg_msgs)
            return {"model": model, "segment": seg_msgs, "verdict": verdict}

        with ThreadPoolExecutor(max_workers=min(workers, len(competitors))) as executor:
            candidates = list(executor.map(evaluate, competitors))
        winner = max(candidates, key=lambda c: (c["verdict"]["overall"], c["model"]["name"]))
        shared += winner["segment"]
        scores_str = ", ".join(
            f"{c['model']['name']}={c['verdict']['overall']:.2f}" for c in candidates
        )
        print(f"[seg] {history['history_id']} seg{seg + 1}/{num_segments} "
              f"winner={winner['model']['name']} ({scores_str})", flush=True)
        seg_log.append({
            "segment_index": seg,
            "winner_model": winner["model"]["name"],
            "scores": {c["model"]["name"]: c["verdict"]["overall"] for c in candidates},
        })
    return shared, seg_log


STRATEGIES = ("random_1", "same_model_2", "random_2_seg", "all_5_seg")


def strategy_budget(strategy: str, num_segments: int, seg_turns: int, n_models: int) -> dict:
    """Rollout budget bookkeeping (role/user/judge calls per sample)."""
    total_turns = num_segments * seg_turns
    if strategy == "random_1":
        role = total_turns
        judge = len(DIMENSIONS)  # one final full-trajectory scoring
        rollouts = 1
    elif strategy == "same_model_2":
        role = 2 * total_turns
        judge = 2 * len(DIMENSIONS)  # score both full trajectories
        rollouts = 2
    elif strategy == "random_2_seg":
        k = 2
        role = k * total_turns
        judge = k * num_segments * len(DIMENSIONS) + len(DIMENSIONS)  # segment picks + final score
        rollouts = k
    else:  # all_5_seg
        k = n_models
        role = k * total_turns
        judge = k * num_segments * len(DIMENSIONS) + len(DIMENSIONS)
        rollouts = k
    return {
        "rollout_budget": rollouts,
        "role_api_calls_per_sample": role,
        "user_api_calls_per_sample": role,  # one user turn precedes each role turn
        "judge_api_calls_per_sample": judge,
    }


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def process_history(
    engine: Engine, history: dict, models: list[dict], model_map: dict, names: list[str],
    anchor: str, num_segments: int, args: argparse.Namespace, pending_strategies: list[str],
) -> list[dict]:
    """Run all pending strategies for one persona; returns result rows."""
    hid = history["history_id"]
    permutation = list(names)
    random.Random(args.analysis_seed + history["source_index"] * 1009).shuffle(permutation)
    m1 = model_map[permutation[0]]
    # random_2_seg = M1 (same_model) + one random partner drawn from the other models.
    # --random2-reseed changes the partner draw (M1 stays fixed) so an unlucky pair can be re-rolled.
    others = [n for n in names if n != permutation[0]]
    partner = random.Random(
        args.analysis_seed + history["source_index"] * 7919 + args.random2_reseed
    ).choice(others)
    m1_m2 = [m1, model_map[partner]]
    cw = args.candidate_workers
    rows = []

    for strategy in pending_strategies:
        print(f"[run] {hid} :: {strategy}", flush=True)
        seg_log: list[dict] = []
        if strategy == "random_1":
            traj = run_full(engine, history, m1, anchor, num_segments, args.segment_turns)
            final = engine.judge(history, [], traj)
            selected_model = m1["name"]
        elif strategy == "same_model_2":
            def one_run(_: int) -> tuple[list[dict], dict]:
                t = run_full(engine, history, m1, anchor, num_segments, args.segment_turns)
                return t, engine.judge(history, [], t)
            with ThreadPoolExecutor(max_workers=2) as executor:
                runs = list(executor.map(one_run, range(2)))
            traj, final = max(runs, key=lambda x: x[1]["overall"])
            selected_model = m1["name"]
        elif strategy == "random_2_seg":
            traj, seg_log = run_segmented(engine, history, m1_m2, anchor, num_segments, args.segment_turns, cw)
            final = engine.judge(history, [], traj)
            selected_model = "|".join(s["winner_model"] for s in seg_log)
        else:  # all_5_seg
            traj, seg_log = run_segmented(engine, history, models, anchor, num_segments, args.segment_turns, cw)
            final = engine.judge(history, [], traj)
            selected_model = "|".join(s["winner_model"] for s in seg_log)

        rows.append({
            "history_id": hid,
            "source_index": history["source_index"],
            "strategy": strategy,
            "model_permutation": permutation,
            "selected_model": selected_model,
            "overall": final["overall"],
            "scores": final["scores"],
            "segment_log": seg_log,
            "final_trajectory": traj,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        })
        print(f"[score] {hid} :: {strategy:14s} overall={final['overall']:.3f} "
              f"selected={selected_model}", flush=True)
    return rows


def run(args: argparse.Namespace, histories: list[dict], models: list[dict], output_dir: Path) -> None:
    engine = Engine(args, models)
    names = [m["name"] for m in models]
    model_map = {m["name"]: m for m in models}
    num_segments = args.total_turns // args.segment_turns
    if args.total_turns % args.segment_turns:
        raise ValueError("--total-turns must be divisible by --segment-turns")

    anchor_path = output_dir / "fixed_user_anchors.json"
    anchors = json.loads(anchor_path.read_text(encoding="utf-8")) if anchor_path.exists() and not args.overwrite_rollouts else {}

    results_path = output_dir / "strategy_results.json"
    results = json.loads(results_path.read_text(encoding="utf-8")) if results_path.exists() and not args.overwrite_rollouts else []
    rerun = {s.strip() for s in args.rerun_strategies.split(",") if s.strip()}
    if rerun:
        before = len(results)
        results = [r for r in results if r["strategy"] not in rerun]
        print(f"[run] rerun strategies {sorted(rerun)}: dropped {before - len(results)} existing rows", flush=True)
    done = {(r["history_id"], r["strategy"]) for r in results}
    if results:
        print(f"[run] resuming: {len(results)} (history,strategy) rows already done", flush=True)

    # Build any missing anchors first (in parallel) so persona workers can start clean.
    missing = [h for h in histories if h["history_id"] not in anchors]
    if missing:
        print(f"[run] building {len(missing)} fixed user anchors ...", flush=True)
        with ThreadPoolExecutor(max_workers=min(args.persona_workers, len(missing))) as executor:
            for hid, anchor in executor.map(lambda h: (h["history_id"], engine.build_anchor(h)), missing):
                anchors[hid] = anchor
                print(f"[anchor] {hid}: {anchor[:40]}...", flush=True)
        dump_json(anchors, anchor_path)
    else:
        print(f"[run] all {len(anchors)} anchors already present", flush=True)

    todo = [h for h in histories if any((h["history_id"], s) not in done for s in STRATEGIES)]
    lock = threading.Lock()
    print(f"[run] {len(todo)} personas have pending strategies", flush=True)

    def worker(history: dict) -> list[dict]:
        pending = [s for s in STRATEGIES if (history["history_id"], s) not in done]
        return process_history(
            engine, history, models, model_map, names,
            anchors[history["history_id"]], num_segments, args, pending,
        )

    if todo:
        with ThreadPoolExecutor(max_workers=min(args.persona_workers, len(todo))) as executor:
            futures = {executor.submit(worker, h): h for h in todo}
            for future in as_completed(futures):
                hid = futures[future]["history_id"]
                try:
                    rows = future.result()
                except Exception as exc:
                    print(f"[history failed] {hid}: {exc}", flush=True)
                    continue
                with lock:
                    results.extend(rows)
                    dump_json(results, results_path)
                print(f"[done] {hid} ({len(rows)} strategies)", flush=True)

    summarize(args, results, models, num_segments, output_dir)


def summarize(args: argparse.Namespace, results: list[dict], models: list[dict], num_segments: int, output_dir: Path) -> None:
    summary = []
    for strategy in STRATEGIES:
        rows = [r for r in results if r["strategy"] == strategy]
        if not rows:
            continue
        budget = strategy_budget(strategy, num_segments, args.segment_turns, len(models))
        overall = [r["overall"] for r in rows]
        record: dict[str, Any] = {"strategy": strategy, **budget, "n": len(rows)}
        record["overall_mean"] = statistics.mean(overall)
        record["overall_std"] = statistics.stdev(overall) if len(overall) > 1 else 0.0
        for dimension in DIMENSIONS:
            record[f"{dimension}_mean"] = statistics.mean(r["scores"][dimension]["score"] for r in rows)
        summary.append(record)

    if not summary:
        return
    csv_path = output_dir / "strategy_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    lines = [
        "# Segmented Rollout Ablation",
        "",
        "| Strategy | Rollouts | Role calls | User calls | Judge calls | N | Overall | Human | Role | Context | Interactive |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['strategy']} | {row['rollout_budget']} | {row['role_api_calls_per_sample']} | "
            f"{row['user_api_calls_per_sample']} | {row['judge_api_calls_per_sample']} | {row['n']} | "
            f"{row['overall_mean']:.3f} | {row['human_likeness_mean']:.3f} | {row['role_consistency_mean']:.3f} | "
            f"{row['context_consistency_mean']:.3f} | {row['interactive_ability_mean']:.3f} |"
        )
    (output_dir / "strategy_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Ablation complete: {output_dir / 'strategy_summary.md'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Segmented best-of-N rollout ablation (2-persona smoke by default).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--models-config", default=str(DEFAULT_MODELS))
    parser.add_argument("--output-dir", default=str(HERE / "outputs" / "segmented_sample2"))
    parser.add_argument("--sample-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260529)
    parser.add_argument("--analysis-seed", type=int, default=20260710)
    parser.add_argument("--total-turns", type=int, default=50)
    parser.add_argument("--segment-turns", type=int, default=10)

    parser.add_argument("--role-temperature", type=float, default=1.0)
    parser.add_argument("--role-max-tokens", type=int, default=2048)
    parser.add_argument("--max-role-chars", type=int, default=500)
    parser.add_argument("--user-simulator-style", choices=("passive", "balanced", "proactive"), default="passive")
    parser.add_argument("--user-temperature", type=float, default=1.0)
    parser.add_argument("--user-max-tokens", type=int, default=200)
    parser.add_argument("--max-user-chars", type=int, default=150)

    parser.add_argument("--judge-prompt-file", default=str(DEFAULT_JUDGE_PROMPTS))
    parser.add_argument("--judge-model", default="gemini-3-flash-preview")
    parser.add_argument("--judge-temperature", type=float, default=0.7)
    parser.add_argument("--judge-max-tokens", type=int, default=4096)

    parser.add_argument("--persona-workers", type=int, default=10,
                        help="Number of personas processed concurrently.")
    parser.add_argument("--candidate-workers", type=int, default=5,
                        help="Concurrent candidate rollouts within one segment.")
    parser.add_argument("--overwrite-selection", action="store_true")
    parser.add_argument("--overwrite-rollouts", action="store_true")
    parser.add_argument("--random2-reseed", type=int, default=0,
                        help="Re-draw the random_2_seg partner model (M1 stays fixed) with this offset.")
    parser.add_argument("--rerun-strategies", default="",
                        help="Comma-separated strategies to recompute (their existing rows are dropped).")
    parser.add_argument("--mock-api", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 70, flush=True)
    print(f"[main] Segmented rollout ablation  (mock={args.mock_api})", flush=True)
    print(f"[main] output_dir     : {output_dir}", flush=True)
    print(f"[main] sample_size    : {args.sample_size}", flush=True)
    print(f"[main] total/seg turns: {args.total_turns} / {args.segment_turns} "
          f"({args.total_turns // args.segment_turns} segments)", flush=True)
    print(f"[main] judge_model    : {args.judge_model}", flush=True)
    print(f"[main] workers        : persona={args.persona_workers}, candidate={args.candidate_workers}", flush=True)
    print("=" * 70, flush=True)
    histories = prepare_selection(args, output_dir)
    print(f"[main] selected {len(histories)} histories: "
          f"{', '.join(h['history_id'] for h in histories)}", flush=True)
    models = load_models(Path(args.models_config))
    if len(models) < 2:
        raise ValueError(f"Need at least 2 models, got {len(models)}")
    print(f"[main] {len(models)} models: {', '.join(m['name'] for m in models)}", flush=True)

    dump_json({
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input": str(Path(args.input).resolve()),
        "models_config": str(Path(args.models_config).resolve()),
        "sample_size": args.sample_size,
        "seed": args.seed,
        "analysis_seed": args.analysis_seed,
        "total_turns": args.total_turns,
        "segment_turns": args.segment_turns,
        "num_segments": args.total_turns // args.segment_turns,
        "user_model": "doubao-1-5-pro-32k-character-250715",
        "user_simulator_style": args.user_simulator_style,
        "judge_model": args.judge_model,
        "strategies": list(STRATEGIES),
    }, output_dir / "run_config.json")

    run(args, histories, models, output_dir)


if __name__ == "__main__":
    main()
