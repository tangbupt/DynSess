# DynSess Fixed-Context Long-Context Evaluation

这套代码用于 rebuttal 的长上下文实验，不修改原始 `test/`、`data/` 或训练代码。

## 实验口径

- 从现有 100 条 eval persona 中用固定 seed 随机抽取 20 条，并保存抽样清单。
- 所有长历史统一由 `Doubao-1.5-pro-32k-character-250715` 构造，避免不同候选模型形成不同历史。
- 一个 turn 严格定义为一条 user 消息加一条 assistant 消息。100 turns 对应 200 条消息。
- 在 50/60/70/80/90/100 turns 处截取固定豆包历史。
- 每个截点的第一条 eval 用户消息只生成一次，五个模型共用；之后每个模型继续与同一个豆包用户模拟器交互。
- 每个模型在每个截点生成 10 个完整 turns，共 20 条新消息。
- 使用原 `dynsess_rubrics.py` 的四个 multi-turn prompt 分别评分。

默认使用论文原 eval 对应的 `passive` 用户模拟器。主动用户对照实验应使用上一级
`rebuttal_eval/run_rebuttal_eval.py`，不要在同一张长上下文主表里同时改变两个变量。

## 1. 构造固定豆包历史

在公司内网设置 token 后运行：

```bash
cd DynSess  # clone 后进入仓库根目录
export DYNS_API_TOKEN='...'
export ARK_API_KEY='...'

python rebuttal_eval/long_context/prepare_doubao_contexts.py \
  --sample-size 20 \
  --seed 20260529 \
  --target-turns 100 \
  --checkpoints 50,60,70,80,90,100 \
  --context-model Doubao-1.5-pro-32k-character-250715 \
  --user-model doubao-1-5-pro-32k-character-250715 \
  --user-simulator-style passive \
  --max-workers 4
```

默认输出目录：

```text
rebuttal_eval/long_context/outputs/doubao_fixed_20p_100t_seed20260529/
```

其中 `selected_personas.jsonl` 是可复现的 20 条抽样清单，`records/` 保存逐 persona
断点，`context_dialogues.jsonl` 是完成后的合并数据。命令中断后直接用相同参数重跑即可续跑。

如果希望首个 eval 用户消息明确检查早期记忆，可额外加 `--memory-probe`。主实验建议先不加，
保持与原 eval user simulator 一致；该选项更适合作为额外的 memory-probe 分析。

## 2. 配置五个候选模型

按公司环境修改新建的 `models.example.json` 中的模型名或 endpoint。它与原始实验代码独立，
也可以通过 `--models-config` 指向另一个新配置文件。

每项支持：

- `name`: 表格中的模型名。
- `model`: chat-completions 请求里的模型名。
- `api_url`: 可选；不写则使用统一的 Fuxi API 地址。
- `token_env`: 可选；该模型单独使用的 token 环境变量。
- `requires_token`: 本地 vLLM 可设为 `false`。
- `temperature`、`max_tokens`、`extra_body`、`system_suffix`: 可选模型参数。

例如本地模型可写成：

```json
{
  "name": "multi-dpo",
  "model": "qwen3-32B-multi-dpo",
  "api_url": "http://localhost:11125/v1/chat/completions",
  "requires_token": false
}
```

示例配置已列出 multi-DPO、multi-GRPO、GPT-5.4、Claude Sonnet 4.6 和 Gemini 3 Pro。
三个闭源模型可按最终 rebuttal 方案替换。

## 3. 生成十轮续写并评分

```bash
python rebuttal_eval/long_context/evaluate_fixed_contexts.py \
  --context-dir rebuttal_eval/long_context/outputs/doubao_fixed_20p_100t_seed20260529 \
  --models-config rebuttal_eval/long_context/models.example.json \
  --checkpoints 50,60,70,80,90,100 \
  --eval-turns 10 \
  --user-model doubao-1-5-pro-32k-character-250715 \
  --judge-model gemini-3-flash-preview \
  --max-workers 8 \
  --judge-workers 12 \
  --mode all
```

也可以分两步运行：

```bash
# 先生成五个模型的十轮续写
python rebuttal_eval/long_context/evaluate_fixed_contexts.py \
  --context-dir rebuttal_eval/long_context/outputs/doubao_fixed_20p_100t_seed20260529 \
  --models-config rebuttal_eval/long_context/models.example.json \
  --mode generate

# 再调用 judge
python rebuttal_eval/long_context/evaluate_fixed_contexts.py \
  --context-dir rebuttal_eval/long_context/outputs/doubao_fixed_20p_100t_seed20260529 \
  --models-config rebuttal_eval/long_context/models.example.json \
  --mode evaluate
```

所有阶段都有 progress 文件，可直接重跑续传。主要结果为：

```text
evaluation/candidate_responses.json
evaluation/evaluated_responses.json
evaluation/summary_by_context.csv
evaluation/summary_by_context.md
```

完整正式实验预计产生 600 段候选模型续写和 2400 次 judge 调用。建议先用
`--sample-size 1 --target-turns 2 --checkpoints 2 --mock-api` 做离线流程检查，再在内网先跑
1 个 persona 的真实 API smoke test。

## 上下文长度注意事项

100 turns 加 persona system prompt 可能超过旧的 8K vLLM 配置。正式运行前需要确认两个本地模型
server 的 `--max-model-len` 足以容纳最长历史和 10-turn 续写；否则 API 会在长 checkpoint 失败，
这属于部署截断而不是模型效果。代码不会静默截断历史。
