#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""一键启动脚本：对话生成 + 格式转换 + 自动评估
Usage: bash run_eval.sh  (或 python eval/run_dynsess_eval.py)

本文件为入口编排器，三个阶段分别在 stage1_generate / stage2_format /
stage3_evaluate 子模块中实现，共享配置见 config.py。
"""
import os
import sys
import time

# 确保能 import 同目录下的 config / stage* 模块（按脚本方式运行时）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    ASSISTANT_MODEL, ASSISTANT_API_BASE_URL, ASSISTANT_API_MODEL_NAME,
    LOCAL_VLLM_URL, LOCAL_MODEL_NAME, GENERATE_MODE, SKIP_STAGE1,
    PRE_GENERATE_OUTPUT_FILE, GENERATE_OUTPUT_FILE, FORMATTED_FILE, EVAL_FINAL_FILE,
)
import stage1_generate
import stage2_format
import stage3_evaluate


def main():
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║          一键启动 Pipeline: 生成 → 转换 → 评估              ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"Assistant模型类型: {ASSISTANT_MODEL}")
    if ASSISTANT_MODEL == "api":
        print(f"  ├─ API地址: {ASSISTANT_API_BASE_URL}")
        print(f"  └─ 模型名称: {ASSISTANT_API_MODEL_NAME}")
    else:
        print(f"  ├─ VLLM地址: {LOCAL_VLLM_URL}")
        print(f"  └─ 模型名称: {LOCAL_MODEL_NAME}")
    print(f"生成模式: {GENERATE_MODE}")
    print(f"跳过Stage1: {SKIP_STAGE1}")
    if SKIP_STAGE1:
        print(f"预生成文件: {PRE_GENERATE_OUTPUT_FILE}")
    else:
        print(f"生成输出: {GENERATE_OUTPUT_FILE}")
    print(f"格式转换: {FORMATTED_FILE}")
    print(f"评估结果: {EVAL_FINAL_FILE}")
    print()

    start_time = time.time()

    # 阶段1：对话生成
    if SKIP_STAGE1:
        print("\n" + "=" * 60)
        print("阶段1：跳过（使用预生成文件）")
        print("=" * 60)
        if not PRE_GENERATE_OUTPUT_FILE:
            raise ValueError("SKIP_STAGE1=True 时，必须设置 PRE_GENERATE_OUTPUT_FILE")
        if not os.path.exists(PRE_GENERATE_OUTPUT_FILE):
            raise FileNotFoundError(f"预生成文件不存在: {PRE_GENERATE_OUTPUT_FILE}")
        gen_file = PRE_GENERATE_OUTPUT_FILE
        print(f"✓ 使用预生成文件: {gen_file}")
    else:
        gen_file = stage1_generate.stage1_generate()

    # 阶段2：格式转换
    fmt_file = stage2_format.stage2_format(gen_file)

    # 阶段3：自动评估
    result_file = stage3_evaluate.stage3_evaluate(fmt_file)

    elapsed = time.time() - start_time
    print(f"\n🎉 Pipeline 全部完成！耗时 {elapsed/60:.1f} 分钟")
    if SKIP_STAGE1:
        print(f"   预生成文件: {PRE_GENERATE_OUTPUT_FILE}")
    else:
        print(f"   生成文件: {GENERATE_OUTPUT_FILE}")
    print(f"   转换文件: {FORMATTED_FILE}")
    print(f"   评估结果: {EVAL_FINAL_FILE}")


if __name__ == "__main__":
    main()
