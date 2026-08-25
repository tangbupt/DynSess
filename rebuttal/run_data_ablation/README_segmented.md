# Segmented Rollout Ablation

在线**分段分叉**版本的 best-of-N 消融，与同目录的 `run_ablation.py`（静态候选池、
整条轨迹离线组合）不同：50 个 eval turn 切成 5 段、每段 10 个 turn，每一段让候选模型
从「共享的当前最优前缀」各自续写，四维 judge 对该段打分，保留最优段作为新前缀继续。

## 四种策略

| 策略 | 说明 | Rollout 预算 |
|---|---|---|
| `random_1` | 随机模型 M1 一条跑到底（50 turn），不分叉 | 1 |
| `same_model_2` | M1 独立跑两条完整 50-turn 轨迹，judge 整条打分取优 | 2 |
| `random_2_seg` | 随机 2 模型，**每段**从共享前缀各续 10 turn，judge 打分留最优段 | 2 |
| `all_5_seg` | 5 模型每段竞争，逻辑同上 | 5 |

每条历史用固定随机排列 `M1…M5`；`random_1`/`same_model_2` 用 `M1`，`random_2_seg`
用 `M1/M2`。所有策略共享同一条固定的豆包首句用户消息（anchor），起点一致。

每个策略最终产出一条 50-turn 轨迹，再用四维 judge **对整条重新打分**作为可比主指标；
分段策略的段级打分只用于选段。四维为 human_likeness / role_consistency /
context_consistency / interactive_ability，overall 取四维均值。

## 输入 / 模型

- 默认输入：`../longer_context/outputs/varied_20p_50to100t_seed20260529/context_dialogues.jsonl`
  （字段 `model_persona` → 角色人设，`user_persona` → 用户模拟器系统提示，`dialogue` → 历史）。
- 模型来自 [models.json](./models.json)（5 个）。judge prompt 默认取
  `../base/dynsess_rubrics.py`。

## 运行

先用 mock 跑通流程（不调用真实 API）：

```bash
python run_segmented_ablation.py --mock-api --sample-size 2 \
  --total-turns 20 --segment-turns 10 --output-dir outputs/mock_smoke
```

真实 2 人设 smoke：

```bash
export DYNS_API_TOKEN='...'
export ARK_API_KEY='...'

python run_segmented_ablation.py \
  --sample-size 2 --total-turns 50 --segment-turns 10 \
  --seed 20260529 --analysis-seed 20260710 \
  --user-simulator-style passive
```

中断后用相同参数重跑会读取 `strategy_results.json` / `fixed_user_anchors.json`
按 (history, strategy) 断点续跑。加 `--overwrite-rollouts` 从头重算。

## 输出（默认 `outputs/segmented_sample2/`）

- `selected_histories.jsonl`：抽取的历史。
- `fixed_user_anchors.json`：每条历史共享的首句用户消息。
- `strategy_results.json`：每条 (history, strategy) 的最终轨迹、段级选段日志、四维分数。
- `strategy_summary.csv` / `strategy_summary.md`：主表（各策略 overall/四维均值 + API 成本）。

## 成本（每条历史，50 turn / 5 段）

| 策略 | Role calls | User calls | Judge 维度调用 |
|---|---:|---:|---:|
| random_1 | 50 | 50 | 4 |
| same_model_2 | 100 | 100 | 8 |
| random_2_seg | 100 | 100 | 2×5×4 + 4 = 44 |
| all_5_seg | 250 | 250 | 5×5×4 + 4 = 104 |

2 条人设合计约 1000 次角色调用、1000 次用户调用、320 次 judge 维度调用。
