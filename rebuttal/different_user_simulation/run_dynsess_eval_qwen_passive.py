#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
新入口：用户模拟器 = qwen-plus-character（走伏羲 V2 接口，即 assistant 的非豆包分支），
        风格 = passive（消极/被动人设）；assistant 仍为本地 vllm(persona_general)。

复用 run_dynsess_eval.py 的全部流程，只覆盖：
  1. USER_SIM_STYLE = "passive"
  2. user_simulate() 改用 qwen-plus-character（llm_call_assistant_api_v2）而非豆包直连
用法: python run_pipeline_qwen_passive.py
"""

import re
import os
from datetime import datetime

import run_dynsess_eval as base

# ─────────────── 覆盖配置 ───────────────
base.USER_SIM_STYLE = "passive"          # 消极/被动人设
USER_SIM_MODEL = "qwen-plus-character"   # 用户模拟器模型（走伏羲 V2，非豆包直连）

# 让输出路径带上区分标签，避免和豆包版结果混在一起
_RUN_TAG = f"qwen_passive_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
base.GENERATE_OUTPUT_FILE = os.path.join(base.OUTPUT_ROOT, "generate", f"dialogues_{_RUN_TAG}.json")
base.FORMATTED_FILE = os.path.join(base.OUTPUT_ROOT, "format", f"dialogues_{_RUN_TAG}_format.json")
base.EVAL_OUTPUT_DIR = os.path.join(base.OUTPUT_ROOT, "eval_result", _RUN_TAG)
base.EVAL_PROGRESS_FILE = os.path.join(base.EVAL_OUTPUT_DIR, "progress.json")
base.EVAL_FINAL_FILE = os.path.join(base.EVAL_OUTPUT_DIR, f"eval_{_RUN_TAG}_final.json")
base.EVAL_LOG_FILE = os.path.join(base.EVAL_OUTPUT_DIR, f"eval_{_RUN_TAG}.log")


def user_simulate(messages, persona_info, user_system_prompt, debug=False):
    """用户模拟器（qwen-plus-character，经伏羲 V2 接口）"""
    full_system_prompt = base.build_user_simulator_prompt(
        base.USER_SIM_STYLE, persona_info, user_system_prompt
    )
    api_messages = [{"role": "system", "content": full_system_prompt}]
    for msg in messages[1:]:
        api_messages.append(msg)

    response = base.llm_call_assistant_api_v2(
        api_messages, temperature=1.0, max_tokens=200, model_name=USER_SIM_MODEL
    )
    if response and response != "null":
        response = re.sub(r'【.*?】', '', response).strip()
        response = re.sub(r'（.*?）', '', response).strip()
    if debug:
        print(f"[DEBUG] 用户回复(qwen-passive): {response}")
    return response


# 覆盖模块内的 user_simulate：generate_dialogue* 在调用时按模块全局查找，故此替换生效
base.user_simulate = user_simulate


if __name__ == "__main__":
    print(f"★ 用户模拟器: {USER_SIM_MODEL} (伏羲V2) | 风格: {base.USER_SIM_STYLE} | assistant: {base.ASSISTANT_MODEL}")
    base.main()
