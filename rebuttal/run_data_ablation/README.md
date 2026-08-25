# Rollout Candidate Ablation

这套代码从现有 SFT test 对话中固定抽取 20 条完整历史，为每条历史生成一个可复用的
`5 models x 2 samples` 候选池，然后只在本地分数上组合不同 rollout budget。

## 实验定义

默认只从 98 条完整的 51-turn 历史中抽样，排除两条中途终止的数据。一个 turn 定义为
一条 user 消息和一条 assistant 消息。

每条历史先生成一条固定的豆包用户消息，所有候选从相同的历史和相同的第一条用户消息开始。
随后每个候选模型独立生成两条轨迹，每条轨迹包含 10 个完整 turns。四维 multi-turn judge
对候选池中的每条轨迹独立评分。

主分析使用固定随机排列 `M1...M5`：

- `random_1`: `M1` 的第 1 条轨迹。
- `random_2_models`: `M1`、`M2` 的第 1 条轨迹，选择 overall 较高者。
- `same_model_2`: `M1` 的两条独立轨迹，选择 overall 较高者。
- `all_5_models`: 五个模型的第 1 条轨迹，选择 overall 最高者。
- `all_10_pool_diagnostic`: 十条候选全部参与，仅作为额外诊断，不属于主表。

候选选择使用四个维度的平均分。选中一条真实轨迹后，报告该轨迹自己的四维分数，不进行
逐维度取最大值。

## 模型配置

[models.json](./models.json) 默认包含：

- GPT-5.4
- Claude Sonnet 4.6
- Doubao 1.5 Character
- Gemini 3 Pro
- Qwen Character Plus

如果某个模型由本地 vLLM 提供，在对应项增加：

```json
{
  "api_url": "http://localhost:PORT/v1/chat/completions",
  "requires_token": false
}
```

## 正式运行

```bash
cd DynSess  # clone 后进入仓库根目录

export DYNS_API_TOKEN='...'
export ARK_API_KEY='...'

python rebuttal_eval/rollout_candidate_ablation/run_ablation.py \
  --mode all \
  --sample-size 20 \
  --seed 20260529 \
  --analysis-seed 20260710 \
  --eval-turns 10 \
  --replicates 2 \
  --user-simulator-style passive \
  --max-workers 8 \
  --judge-workers 12
```

也可以分阶段运行，命令中断后使用相同参数重跑会读取 progress 文件：

```bash
python rebuttal_eval/rollout_candidate_ablation/run_ablation.py --mode generate
python rebuttal_eval/rollout_candidate_ablation/run_ablation.py --mode evaluate
python rebuttal_eval/rollout_candidate_ablation/run_ablation.py --mode analyze
```

更换 `--analysis-seed` 后只运行 `--mode analyze`，可以重新随机选择 `M1/M2`，不会重新生成
轨迹或调用 judge。

## 输出

默认输出目录：

```text
rebuttal_eval/rollout_candidate_ablation/outputs/test100_sample20_seed20260529/
```

主要文件：

- `selected_histories.jsonl`: 固定抽取的 20 条完整历史。
- `fixed_user_anchors.json`: 每条历史共用的第一条 eval 用户消息。
- `rollout_pool.json`: 200 条十轮轨迹。
- `scored_rollout_pool.json`: 200 条轨迹及其四维分数。
- `strategy_assignments.json`: 每条样本的随机模型排列。
- `strategy_selected_results.json`: 每种策略的候选集合和获胜轨迹。
- `strategy_summary.csv` / `strategy_summary.md`: rebuttal 主表数据和 API 成本。

完整实验产生 200 条轨迹，需要约 2000 次角色模型调用、1820 次豆包用户模拟器调用和
800 次 judge 调用。建议先用 `--mock-api --sample-size 1 --eval-turns 2` 检查流程，再进行
一个真实 persona 的内网 smoke test。
