#!/usr/bin/env bash
# 只重跑 random_2_seg:M1(same_model)固定不变,第二个搭档模型用新种子重抽一个。
# random_1 / same_model_2 / all_5_seg 的已有结果保留、不重算。
set -euo pipefail

cd "$(dirname "$0")"

OUT=outputs/segmented_sample10

# 先备份当前 20 人设结果(含旧的 random_2_seg),便于对比。
cp "$OUT/strategy_results.json" "$OUT/strategy_results.before_reseed.json"
cp "$OUT/strategy_summary.md"   "$OUT/strategy_summary.before_reseed.md"

python3 run_segmented_ablation.py \
  --sample-size 20 --total-turns 50 --segment-turns 10 \
  --persona-workers 10 --candidate-workers 5 \
  --rerun-strategies random_2_seg \
  --random2-reseed 1 \
  --output-dir "$OUT" \
  2>&1 | tee -a "$OUT.log"
