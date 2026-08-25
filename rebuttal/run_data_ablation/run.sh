#!/usr/bin/env bash
# 分段 best-of-N rollout 消融：10 人设、50 turn、5 段（每段 10 turn）。
set -euo pipefail

cd "$(dirname "$0")"

# sample-size 20 = 数据集全部 20 条人设；已跑完的 10 条按 history_id 自动跳过，
# 只补跑剩余 10 条，最终汇总为 20 人设结果。首次跑需 --overwrite-selection 扩充选样。
python3 run_segmented_ablation.py \
  --sample-size 20 --total-turns 50 --segment-turns 10 \
  --persona-workers 10 --candidate-workers 5 \
  --overwrite-selection \
  --output-dir outputs/segmented_sample10 \
  2>&1 | tee -a outputs/segmented_sample10.log
