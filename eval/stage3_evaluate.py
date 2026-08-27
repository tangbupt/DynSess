#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""阶段3：rubric-anchored 多维度自动评判（session-level）。"""
import json
import os
import re
import time
from datetime import datetime
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from threading import Lock

from config import (
    eval_client, JUDGE_PROMPT_FILE,
    MAX_WORKERS, MAX_RETRIES, RETRY_DELAY, SAVE_INTERVAL,
    EVAL_OUTPUT_DIR, EVAL_PROGRESS_FILE, EVAL_FINAL_FILE, EVAL_LOG_FILE,
)

def write_log(msg):
    """写日志"""
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{ts}] {msg}"
    print(line)
    os.makedirs(EVAL_OUTPUT_DIR, exist_ok=True)
    with open(EVAL_LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(line + "\n")


# 评判客户端失败时走 write_log（保留原"评估API请求失败"日志）
eval_client.error_log = write_log


def extract_json_from_response(response):
    """从LLM响应中提取JSON"""
    if not isinstance(response, str):
        return None
    try:
        return json.loads(response)
    except Exception:
        pass

    json_candidates = []
    for pattern in [r'```json\s*\n?(.*?)\n?```', r'```\s*\n?(.*?)\n?```']:
        m = re.search(pattern, response, re.DOTALL)
        if m:
            json_candidates.append(m.group(1).strip())

    first = response.find('{')
    last = response.rfind('}')
    if first != -1 and last > first:
        json_candidates.append(response[first:last + 1])

    for s in json_candidates:
        try:
            return json.loads(s)
        except Exception:
            pass
        if '\\n' in s or '\\"' in s:
            try:
                return json.loads(s.encode().decode('unicode_escape'))
            except Exception:
                pass

    # 正则兜底
    result = {}
    for pat in [r'"score"\s*:\s*(\d+)', r'score["\s:]+(\d+)']:
        m = re.search(pat, response)
        if m:
            try:
                s = int(m.group(1))
                if 1 <= s <= 5:
                    result['score'] = s
                    break
            except Exception:
                pass
    for pat in [r'"reason"\s*:\s*"([^"]+)"', r'reason["\s:]+([^,}]+)']:
        m = re.search(pat, response, re.DOTALL)
        if m:
            result['reason'] = m.group(1).strip()
            break
    return result if result else None


def load_eval_prompts():
    """加载评判Prompt"""
    with open(JUDGE_PROMPT_FILE, 'r', encoding='utf-8') as f:
        content = f.read()

    prompts = {}
    names = [
        'multi_turn_eval_score_human_likeness',
        'multi_turn_eval_score_role_consistency',
        'multi_turn_eval_score_context_consistency',
        'multi_turn_eval_score_interactivity',
    ]
    for name in names:
        m = re.search(rf'{name}\s*=\s*"{{2,5}}(.*?)"{{2,5}}', content, re.DOTALL)
        if not m:
            m = re.search(rf'{name}\s*=\s*["]+\n(.*?)(?=\n\w+\s*=\s*["{{]|\Z)', content, re.DOTALL)
        if m:
            prompts[name] = m.group(1).strip().strip('"')
        else:
            write_log(f"⚠️ 未找到 prompt: {name}")
    return prompts


def format_dialogue(dialogue_list):
    """格式化对话列表为编号字符串"""
    lines = []
    for i, turn in enumerate(dialogue_list):
        role_label = "用户" if turn.get("role") == "user" else "角色"
        lines.append(f"{i + 1}. {role_label}: {turn.get('content', '')}")
    return "\n".join(lines)


DIMENSION_MAP = {
    'human_likeness':      'multi_turn_eval_score_human_likeness',
    'role_consistency':    'multi_turn_eval_score_role_consistency',
    'context_consistency': 'multi_turn_eval_score_context_consistency',
    'interactive_ability': 'multi_turn_eval_score_interactivity',
}


def evaluate_dimension(persona_info, dialogue_history_str, eval_dialogue_str,
                       prompt_template, dim_key, record_idx):
    """单维度评分"""
    prompt = prompt_template \
        .replace("{character_profile}", persona_info) \
        .replace("{dialogue_history}", dialogue_history_str) \
        .replace("{dialogue}", eval_dialogue_str)

    for attempt in range(MAX_RETRIES):
        resp = eval_client.chat([{"role": "user", "content": prompt}], max_tokens=4096)
        if resp and resp != "null":
            parsed = extract_json_from_response(resp)
            if parsed and dim_key in parsed:
                dim_result = parsed[dim_key]
                if "score" in dim_result and "reason" in dim_result:
                    score = dim_result["score"]
                    if isinstance(score, (int, float)) and 1 <= int(score) <= 5:
                        dim_result["score"] = int(score)
                        return dim_result
            write_log(f"  [记录{record_idx}] 维度 {dim_key} 解析失败，第{attempt+1}次重试")
        if attempt < MAX_RETRIES - 1:
            time.sleep(RETRY_DELAY)

    write_log(f"  [记录{record_idx}] 维度 {dim_key} 全部重试失败，使用默认值")
    return {"score": 3, "reason": "API调用失败或解析失败"}


def evaluate_record(record, prompts, record_idx):
    """单条记录全维度评分"""
    persona_info = record.get("persona_info", "")
    all_dialogue = record.get("dialogue", [])
    original_turns = record.get("original_turns", 0)

    history = all_dialogue[:original_turns]
    eval_dialogue = all_dialogue[original_turns:]

    if not eval_dialogue:
        write_log(f"  [记录{record_idx}] 无续写内容，对完整对话打分")
        history = []
        eval_dialogue = all_dialogue

    history_str = format_dialogue(history)
    eval_str = format_dialogue(eval_dialogue)

    scores = {}
    for dim_key, prompt_key in DIMENSION_MAP.items():
        if prompt_key not in prompts:
            write_log(f"  [记录{record_idx}] 缺少 prompt: {prompt_key}")
            continue
        scores[dim_key] = evaluate_dimension(
            persona_info, history_str, eval_str,
            prompts[prompt_key], dim_key, record_idx
        )
        time.sleep(0.3)

    return {
        "record_index": record_idx,
        "persona_info": persona_info,
        "eval_turns": len(eval_dialogue),
        "history_turns": len(history),
        "scores": scores,
    }


def load_progress():
    """加载评估进度"""
    if os.path.exists(EVAL_PROGRESS_FILE):
        try:
            with open(EVAL_PROGRESS_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
            write_log(f"✓ 加载已有进度，已完成 {len(data)} 条")
            return {int(k): v for k, v in data.items()}
        except Exception as e:
            write_log(f"⚠️ 进度文件损坏: {e}")
    return {}


def save_progress(completed):
    """保存评估进度"""
    with open(EVAL_PROGRESS_FILE, 'w', encoding='utf-8') as f:
        json.dump({str(k): v for k, v in completed.items()}, f, ensure_ascii=False, indent=2)


def calculate_statistics(results):
    """计算评估统计"""
    dims = list(DIMENSION_MAP.keys())
    stats = {"total": len(results), "dimensions": {}}
    for dim in dims:
        scores = [r["scores"][dim]["score"] for r in results
                  if r.get("scores") and dim in r["scores"]
                  and isinstance(r["scores"][dim].get("score"), (int, float))]
        if scores:
            stats["dimensions"][dim] = {
                "mean": round(float(np.mean(scores)), 4),
                "std": round(float(np.std(scores)), 4),
                "min": int(np.min(scores)),
                "max": int(np.max(scores)),
                "count": len(scores),
            }
    all_means = [v["mean"] for v in stats["dimensions"].values() if v["count"] > 0]
    stats["overall_mean"] = round(float(np.mean(all_means)), 4) if all_means else 0.0
    return stats


def stage3_evaluate(input_file):
    """阶段3：评估"""
    print("\n" + "=" * 60)
    print("阶段3：自动评估")
    print("=" * 60)

    os.makedirs(EVAL_OUTPUT_DIR, exist_ok=True)
    write_log("=" * 60)
    write_log("对话多维度评测开始")
    write_log(f"输入文件: {input_file}")
    write_log(f"输出目录: {EVAL_OUTPUT_DIR}")
    write_log(f"并发线程: {MAX_WORKERS}")
    write_log("=" * 60)

    with open(input_file, 'r', encoding='utf-8') as f:
        records = json.load(f)
    write_log(f"✓ 加载记录数: {len(records)}")

    prompts = load_eval_prompts()
    write_log(f"✓ 加载 prompt 数: {len(prompts)}")

    completed = load_progress()
    pending = [(i, rec) for i, rec in enumerate(records) if i not in completed]
    write_log(f"待处理: {len(pending)} / {len(records)}")

    if pending:
        lock_counter = [0]
        save_lock = Lock()

        def worker(args):
            idx, rec = args
            result = evaluate_record(rec, prompts, idx)
            if result is not None:
                with save_lock:
                    completed[idx] = result
                    lock_counter[0] += 1
                    if lock_counter[0] % SAVE_INTERVAL == 0:
                        save_progress(completed)
                        write_log(f"  进度已保存，共完成 {len(completed)} 条")
            return idx, result

        write_log(f"\n开始并发评分（{MAX_WORKERS} 线程）...")
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(worker, item): item[0] for item in pending}
            for future in tqdm(as_completed(futures), total=len(pending), desc="评分进度"):
                try:
                    idx, result = future.result()
                    if result:
                        write_log(f"  ✓ 记录 {idx} 完成，"
                                  f"各维度分: "
                                  + " | ".join(
                                      f"{k}={v['score']}"
                                      for k, v in result["scores"].items()
                                  ))
                except Exception as e:
                    write_log(f"⚠️ 处理异常: {e}")

        save_progress(completed)
        write_log(f"\n✓ 全部评分完成，有效结果: {len(completed)}")

    # 汇总输出
    results_list = [completed[i] for i in sorted(completed.keys())]
    stats = calculate_statistics(results_list)

    final_data = {
        "meta": {
            "input_file": input_file,
            "eval_model": eval_client.model_name,
            "eval_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "total_records": len(records),
            "evaluated_records": len(results_list),
        },
        "statistics": stats,
        "results": results_list,
    }

    with open(EVAL_FINAL_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, ensure_ascii=False, indent=2)

    write_log(f"\n✓ 最终结果已保存: {EVAL_FINAL_FILE}")
    write_log("\n" + "=" * 60)
    write_log("评测结果摘要")
    write_log("=" * 60)
    write_log(f"有效评分记录数: {stats['total']}")
    write_log(f"总体平均分: {stats['overall_mean']:.4f}")
    write_log("各维度得分:")
    for dim, dim_stat in stats["dimensions"].items():
        write_log(f"  {dim}: {dim_stat['mean']:.4f} (±{dim_stat['std']:.4f}), "
                  f"范围 [{dim_stat['min']}, {dim_stat['max']}]")
    write_log("=" * 60)

    return EVAL_FINAL_FILE

