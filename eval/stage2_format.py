#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""阶段2：将阶段1的生成结果转换为评估所需的统一格式。"""
import json
import os

from config import FORMATTED_FILE

def stage2_format(input_file):
    """
    阶段2：将阶段1的输出转换为评估脚本所需的格式
    
    转换逻辑：
    - 续写模式: original_dialogue + continued_dialogue → 合并为 dialogue, 记录 original_turns
    - 从零模式: dialogue 直接保留, original_turns = 0
    """
    print("\n" + "=" * 60)
    print("阶段2：格式转换")
    print("=" * 60)

    with open(input_file, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    formatted = []
    for i, record in enumerate(raw_data):
        if "original_dialogue" in record and "continued_dialogue" in record:
            # 续写模式数据
            orig = record.get("original_dialogue", [])
            cont = record.get("continued_dialogue", [])
            merged = orig + cont
            formatted.append({
                "persona_info": record.get("persona_info", ""),
                "dialogue": merged,
                "original_turns": len(orig),
                "continued_turns": len(cont),
            })
        elif "dialogue" in record:
            # 从零开始模式数据
            formatted.append({
                "persona_info": record.get("persona_info", ""),
                "dialogue": record.get("dialogue", []),
                "original_turns": 0,
                "continued_turns": len(record.get("dialogue", [])),
            })
        else:
            print(f"  警告: 第 {i} 条记录格式未知，跳过")

    output_dir = os.path.dirname(FORMATTED_FILE)
    os.makedirs(output_dir, exist_ok=True)
    with open(FORMATTED_FILE, 'w', encoding='utf-8') as f:
        json.dump(formatted, f, ensure_ascii=False, indent=2)

    print(f"✓ 阶段2完成，转换 {len(formatted)} 条记录，保存到: {FORMATTED_FILE}")
