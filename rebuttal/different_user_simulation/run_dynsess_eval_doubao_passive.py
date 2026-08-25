#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
新入口：用户模拟器 = 豆包(doubao, 直连 ARK)，风格 = passive（原始/论文基线被动人设）；
        assistant 仍为本地 vllm(persona_general)。

复用 run_dynsess_eval.py 的全部流程，只覆盖：
  1. USER_SIM_STYLE = "passive"
用户模拟器保持原始的豆包直连（base.user_simulate 已用 llm_call_user），无需改动。
用法: python run_pipeline_doubao_passive.py
"""

import os
from datetime import datetime

import run_dynsess_eval as base

# ─────────────── 覆盖配置 ───────────────
base.USER_SIM_STYLE = "passive"          # 原始被动人设

# 让输出路径带上区分标签，避免和其他版本结果混在一起
_RUN_TAG = f"doubao_passive_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
base.GENERATE_OUTPUT_FILE = os.path.join(base.OUTPUT_ROOT, "generate", f"dialogues_{_RUN_TAG}.json")
base.FORMATTED_FILE = os.path.join(base.OUTPUT_ROOT, "format", f"dialogues_{_RUN_TAG}_format.json")
base.EVAL_OUTPUT_DIR = os.path.join(base.OUTPUT_ROOT, "eval_result", _RUN_TAG)
base.EVAL_PROGRESS_FILE = os.path.join(base.EVAL_OUTPUT_DIR, "progress.json")
base.EVAL_FINAL_FILE = os.path.join(base.EVAL_OUTPUT_DIR, f"eval_{_RUN_TAG}_final.json")
base.EVAL_LOG_FILE = os.path.join(base.EVAL_OUTPUT_DIR, f"eval_{_RUN_TAG}.log")


if __name__ == "__main__":
    print(f"★ 用户模拟器: {base.USER_MODEL_NAME} (豆包直连) | 风格: {base.USER_SIM_STYLE} | assistant: {base.ASSISTANT_MODEL}")
    base.main()
