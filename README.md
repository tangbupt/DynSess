# DynSess
[![license](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![EMNLP](https://img.shields.io/badge/EMNLP-2026-%23f1592a?labelColor=%23003973&color=%23be1c1a)](https://2026.emnlp.org/)
 - [*DynSess: Dynamic Session-Level Evaluation and Optimization Framework for Role-Playing Agents*](https://arxiv.org/)

> Official code for **"DynSess: Dynamic Session-Level Evaluation and Optimization Framework for Role-Playing Agents"** (EMNLP 2026). Role-playing with LLMs is a **session-level** task: an agent must sustain character identity and interaction quality across extended, multi-turn conversations. **DynSess** couples evaluation and optimization through a shared session-level reward.

## 🔔 News
- **`2026-08`** We release the [[Repo](https://github.com/tangbupt/DynSess)] for our **`EMNLP 2026`** paper.
- **`2026-05`** Our paper: *DynSess: Dynamic Session-Level Evaluation and Optimization Framework for Role-Playing Agents* was accepted by **`EMNLP 2026`**.

## 🌈 Framework
![Framework](./assets/figures/framework.png)

> **DynSess-Eval** (left) simulates multi-turn user–agent sessions and scores them across four dimensions via a rubric-anchored judge. **DynSess-Character** (right) leverages these session-level rewards to construct trajectories through multi-turn lookahead search for SFT, and aligns the agent via off-policy **DSPO** or on-policy **GSRPO**.

## 📊 Results

**Alignment with human judgments** — DynSess-Eval (Session-Level) leads on Rank Accuracy and Normalized MAE across all four dimensions.

| Eval Method | Inter. Rank↑ | Role Rank↑ | Context MAE↓ |
|---|---|---|---|
| CharacterJudge (Turn) | 0.27 | 0.53 | 0.54 |
| RMTBench (Pseudo-Session) | 0.33 | 0.50 | 0.50 |
| CharacterArena (Trajectory) | 0.53 | 0.57 | — |
| **DynSess-Eval (Session)** | **0.83** | **0.67** | **0.22** |

**Role-playing model performance** (human eval, 10-turn) — DynSess-Character-32B matches the ~200B proprietary baseline at ~6× parameter efficiency.

| Model | Params | Avg | Role |
|---|---|---|---|
| GPT-5.4 | — | 3.31 | 3.38 |
| Doubao-1.5-pro-character | ~200B | **3.38** | 3.47 |
| **DynSess-Character-32B (DSPO)** | 32B | 3.37 | **3.56** |
| **DynSess-Character-32B (GSRPO)** | 32B | 3.35 | 3.46 |

<details>
<summary><b>Case study & session-length analysis</b></summary>

<p align="center"><img src="./assets/figures/intro.png" width="60%"></p>

<p align="center"><img src="./assets/figures/case.png" width="70%"></p>

<p align="center"><img src="./assets/figures/session-length.png" width="45%"></p>

</details>

## 📦 Data Download
- **Personas.** 2,100 character personas (2,000 train / 100 held-out test), spanning celebrities, literary/media, game, social, and non-human characters.
- **Seed sessions.** 100 seed persona+history records in [`data/test_dialogue_0424.jsonl`](./data/test_dialogue_0424.jsonl) (~841 KB, `continue`-mode input).
- **Training-format samples.** One 2-line final-format example each for SFT and DSPO (session-level prefix-chain merge) in [`data/train_samples/`](./data/train_samples) — each record carries a non-trainable persona block plus a `dialogues` list with per-turn `trainable` flags, so you can mirror the schema for your own corpora.

> The full SFT/DPO corpora (~512 MB) and the `persona_general` checkpoint are **not** bundled (size). Prepare your own data following the [`data/train_samples/`](./data/train_samples) schema. The shipped auto-eval stats across 89 runs are in [`results/eval_summary.json`](./results/eval_summary.json) (overall mean 4.18).

## 📕 Code Path

#### Code Structures
There are three parts in the code.
- **`eval/`**: the 3-stage eval pipeline — `run_dynsess_eval.py` (entry/orchestrator) + `config.py` (shared config) + `llm.py` (unified `LLMClient`: one `model_name` + `chat(messages)` API, all auth/model selection inside) + one module per stage (`stage1_generate.py` / `stage2_format.py` / `stage3_evaluate.py`). One-click entry: **`bash run_eval.sh`** at the repo root.
- **`prompt/`**: the judge rubrics, split by level — `session_level.py` (multi-turn, the pipeline default) and `turn_level.py` (single-turn) — plus `user_sim_prompt.py`, the user-simulator prompt templates (`passive` / `balanced` / `proactive` styles, isolated from stage1; select via `USER_SIM_STYLE`).
- **`data/train_samples/`**: one 2-line final-format example each for SFT and DSPO — a non-trainable persona block (`dynamic_text`) plus a `dialogues` list with per-turn `trainable` flags (session-level prefix-chain merge for DSPO). Mirror this schema to build your own corpora.

<details>
<summary><b>Full tree</b></summary>

```
DynSess/
├── run_eval.sh                    # one-click entry: bash run_eval.sh
├── eval/
│   ├── run_dynsess_eval.py        # entry: orchestrates the 3 stages
│   ├── config.py                  # shared config (model / paths / LLMClient instances)
│   ├── llm.py                     # unified LLMClient (auth + model selection, one chat() API)
│   ├── stage1_generate.py         # stage 1: dialogue generation
│   ├── stage2_format.py           # stage 2: format conversion
│   └── stage3_evaluate.py         # stage 3: rubric-anchored judge
├── prompt/
│   ├── session_level.py           # multi-turn (session-level) judge rubrics
│   ├── turn_level.py              # single-turn (turn-level) judge rubrics
│   └── user_sim_prompt.py         # user-simulator prompt templates (passive/balanced/proactive)
├── data/
│   ├── test_dialogue_0424.jsonl              # 100 seed sessions
│   └── train_samples/                        # final-format SFT/DSPO samples (2-line each)
├── results/                       # aggregate eval stats + example result
├── assets/figures/                # figures rendered from the paper
├── requirements.txt
├── .env.example
├── LICENSE
└── README.md
```
</details>

## 🔬 Dependencies

- ```Python 3.8+```
- ```openai```, `requests`, `numpy`, `tqdm```` (the only runtime deps)
- Install:
```bash
pip install -r requirements.txt
```

DynSess loads **all credentials from environment variables** — no keys are stored in the repo. Copy [`.env.example`](./.env.example) to `.env` (or `export` in your shell):

```bash
export ARK_API_KEY=<volcengine-ark-key>            # user simulator (Doubao)
export LOCAL_VLLM_URL=http://localhost:88/v1/chat/completions   # role-playing model
export EVAL_API_URL=<your-openai-compatible-judge-endpoint>
export DYNS_EVAL_API_TOKEN=<your-judge-token>
```

## 🚀 Train & Eval

### Evaluate a role-playing model
Edit the config block at the top of [`eval/run_dynsess_eval.py`](./eval/run_dynsess_eval.py) to point `LOCAL_VLLM_URL` / `LOCAL_MODEL_NAME` at your model, then:
```shell
bash run_eval.sh        # or: python eval/run_dynsess_eval.py
```
Outputs land under `./evaluate/` (generated sessions → merged formats → per-record scores + statistics). The pipeline is **resumable** — re-running continues from `progress.json`.

### Training-data format
The conversion scripts and full corpora are not shipped (size). Each record in [`data/train_samples/`](./data/train_samples) is one final-format example — a non-trainable persona block (`dynamic_text`) plus a `dialogues` list where every turn carries a `trainable` flag (assistant turns train, user/history turns don't); DSPO uses the session-level prefix-chain merge. Mirror this schema to prepare your own SFT/DSPO corpora.

### [Parameter](#content)
```
[--GENERATE_MODE {continue,scratch}] [--USER_SIM_STYLE {passive,balanced,proactive}] [--BATCH_SIZE] [--STAGE1_MAX_WORKERS] [--MAX_WORKERS] [--SKIP_STAGE1]
[--LOCAL_MODEL_NAME] [--ASSISTANT_MODEL {local,api}] [--LOCAL_VLLM_URL] [--EVAL_API_URL]
```
**Note**: edit the config block at the top of `eval/run_dynsess_eval.py` for <a href="#Parameter">parameter</a> modification.

## 🤝 Cite
Please consider citing this paper if you use the ```code``` or ```data``` from our work. Thanks a lot :)

```bibtex
@inproceedings{dynsess2026,
  title     = {DynSess: Dynamic Session-Level Evaluation and Optimization Framework for Role-Playing Agents},
  author    = {Zhang, Rongsheng and Tang, Jiji and Ren, Junnan and Bao, Zuyi and Chen, Weijie and Hu, Ruofan and Lv, Tangjie and Zhao, Zhou and Zhang, Yan},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year      = {2026}
}
```

## 📄 License
Released under the [MIT License](./LICENSE).

## 🙏 Acknowledgements
The default judge was developed against an internal NetEase Fuxi LLM gateway; the user simulator defaults to Volcengine Doubao-1.5-pro-32k-character.
