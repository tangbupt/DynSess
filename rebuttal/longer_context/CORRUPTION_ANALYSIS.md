# 破坏(人设矛盾注入) A/B 分析 — role_consistency

口径：proactive 用户模拟器 + 每条续写 20 轮；破坏=把历史中约30%的角色回合替换成明显违背人设的台词。judge 仅评 role_consistency 维度，读取 {人设}+{历史}+{续写}。

## 1. 汇总

| 数据集 | 历史 | 条件 | N | 均值 | Std | 分布 |
|---|---|---|---:|---:|---:|---|
| Set A | 50-100 turns | 无破坏 | 20 | 4.95 | 0.22 | {4: 1, 5: 19} |
| Set A | 50-100 turns | 破坏30% | 20 | 4.55 | 0.97 | {2: 2, 3: 1, 4: 1, 5: 16} |
| Set B | 200 turns | 无破坏 | 10 | 4.60 | 0.92 | {2: 1, 4: 1, 5: 8} |
| Set B | 200 turns | 破坏30% | 10 | 4.10 | 1.14 | {2: 2, 4: 3, 5: 5} |

## 2. 逐 persona (baseline -> corrupted, Δ)

### Set A (50-100 turns)
| persona | turns | baseline | corrupted | Δ |
|---|---:|---:|---:|---:|
| p026 | 58 | 5 | 5 | 0 |
| p024 | 61 | 5 | 5 | 0 |
| p023 | 61 | 4 | 2 | -2 |
| p094 | 63 | 5 | 5 | 0 |
| p068 | 65 | 5 | 4 | -1 |
| p067 | 66 | 5 | 5 | 0 |
| p080 | 68 | 5 | 5 | 0 |
| p048 | 69 | 5 | 5 | 0 |
| p058 | 70 | 5 | 5 | 0 |
| p071 | 73 | 5 | 5 | 0 |
| p073 | 74 | 5 | 5 | 0 |
| p020 | 82 | 5 | 5 | 0 |
| p018 | 85 | 5 | 5 | 0 |
| p089 | 86 | 5 | 5 | 0 |
| p078 | 87 | 5 | 2 | -3 |
| p082 | 88 | 5 | 5 | 0 |
| p060 | 89 | 5 | 5 | 0 |
| p066 | 92 | 5 | 3 | -2 |
| p037 | 96 | 5 | 5 | 0 |
| p002 | 96 | 5 | 5 | 0 |

### Set B (200 turns)
| persona | turns | baseline | corrupted | Δ |
|---|---:|---:|---:|---:|
| p036 | 200 | 2 | 4 | 2 |
| p077 | 200 | 5 | 4 | -1 |
| p049 | 200 | 5 | 2 | -3 |
| p087 | 200 | 5 | 5 | 0 |
| p038 | 200 | 5 | 5 | 0 |
| p098 | 200 | 4 | 4 | 0 |
| p047 | 200 | 5 | 5 | 0 |
| p014 | 200 | 5 | 5 | 0 |
| p050 | 200 | 5 | 5 | 0 |
| p041 | 200 | 5 | 2 | -3 |

## 3. 掉分的两种机制（重要）

- **机制1 · judge 因历史矛盾扣分，但续写未漂**：如 Set A p078(甄嬛) 5→2，judge 评语称续写"堪称惊艳"，但因历史里的OOC判整体不稳。→ 这部分降分反映的是"judge 把污染历史计入"，不是模型自身漂移。
- **机制2 · 续写被污染真漂移**：如 Set A p023(April) 4→2、Set B p049(张曼玉) 5→2、p041 5→2，续写本身丢掉人设核心（价值观/棱角），退化为通用/顺从口吻。→ 这才是模型在被污染长上下文下的真实人设稳定性。

## 4. 结论

1. 注入人设矛盾能有效打破天花板：两套均值降 0.4~0.5，Std 从~0升到~1，2/3分出现。
2. 长历史更脆弱：无破坏时 200轮(4.60) 已低于 50-100轮(4.95)；破坏后 200轮进一步跌到 4.10。
3. 注意区分两种降分机制；若要纯测"模型自身是否漂移"，需让 judge 只对续写评分、忽略历史OOC（当前 prompt 会把历史算进去）。

## 5. 数据位置

- Set A: `longer_context/outputs/varied_20p_50to100t_seed20260529/rc_proactive_baseline/` 与 `longer_context/outputs/varied_20p_50to100t_seed20260529/rc_proactive_corrupted/`；破坏历史 `longer_context/outputs/varied_20p_50to100t_seed20260529/context_dialogues_corrupted_p30.jsonl`
- Set B: `longer_context/outputs/varied_10p_200t_seed20260529/rc_proactive_baseline/` 与 `longer_context/outputs/varied_10p_200t_seed20260529/rc_proactive_corrupted/`；破坏历史 `longer_context/outputs/varied_10p_200t_seed20260529/context_dialogues_corrupted_p30.jsonl`
