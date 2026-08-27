#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""阶段1：多轮对话生成（用户模拟器 ↔ 角色扮演模型）。"""
import json
import os
import re
import sys
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from threading import Lock

from config import (
    user_client, assistant_client,
    WORD_LIMIT_MODELS, WORD_LIMIT_PROMPT,
    GENERATE_MODE, HISTORY_FILE, PERSONA_FILE,
    BATCH_SIZE, STAGE1_MAX_WORKERS, SAVE_INTERVAL,
    GENERATE_OUTPUT_FILE,
    build_user_simulator_prompt, USER_SIM_STYLE,
)

def user_simulate(messages, persona_info, user_system_prompt, debug=False):
    """用户模拟器（豆包API）"""
    full_system_prompt = build_user_simulator_prompt(
        USER_SIM_STYLE, persona_info, user_system_prompt
    )

    api_messages = [{"role": "system", "content": full_system_prompt}]
    for msg in messages[1:]:
        api_messages.append(msg)
    response = user_client.chat(api_messages, temperature=1.0, max_tokens=200)
    if response != "null":
        response = re.sub(r'【.*?】', '', response).strip()
        response = re.sub(r'（.*?）', '', response).strip()
    if debug:
        print(f"[DEBUG] 用户回复: {response}")
    return response


def role_play_chat(messages, persona_info, debug=False):
    """角色扮演聊天（根据ASSISTANT_MODEL配置选择本地VLLM或外部API模型）"""
    processed = persona_info
    if persona_info.startswith("请你扮演以下人设："):
        processed = persona_info[len("请你扮演以下人设："):]
    
    # 构建基础 system prompt
    system_prompt = f"你是一个角色扮演AI助手。请严格按照以下人设进行角色扮演：\n\n{processed}\n\n"

    # 检查当前模型是否需要字数限制
    if assistant_client.model_name in WORD_LIMIT_MODELS:
        system_prompt = f"{WORD_LIMIT_PROMPT}\n\n{system_prompt}"

    api_messages = [{"role": "system", "content": system_prompt}]
    api_messages.extend(messages)

    response = assistant_client.chat(api_messages, temperature=0.7, max_tokens=1024)

    if debug:
        print(f"[DEBUG] 角色回复: {response}")
    return response


def generate_dialogue(persona_info, user_system_prompt, debug=False):
    """从零开始生成对话"""
    role_messages = []
    user_messages = [{"role": "system", "content": "placeholder"}]
    num_turns = random.randint(8, 10)
    dialogue_history = []

    for turn in range(num_turns):
        if turn == 0:
            user_input = "你好"
            role_messages.append({"role": "user", "content": user_input})
        else:
            user_input = user_messages[-1]["content"] if user_messages else "你好"
            role_messages.append({"role": "user", "content": user_input})

        # 第一轮调用失败时直接抛出异常（API级别错误），不再静默吞掉
        role_response = role_play_chat(role_messages, persona_info, debug=debug)

        # 内容质量过滤（非致命，正常跳过该轮）
        if role_response in ("抱歉，没有收到回复。", "抱歉，我暂时无法回复，请稍后再试。"):
            print(f"  [turn {turn}] 角色扮演返回无效内容，跳过: '{role_response}'")
            break
        if len(role_response) > 500 or len(set(role_response)) < 5:
            print(f"  [turn {turn}] 角色扮演内容不符合质量要求(长度{len(role_response)}, 去重字符{len(set(role_response))}), 跳过")
            break

        role_messages.append({"role": "assistant", "content": role_response})
        user_messages.append({"role": "user", "content": role_response})
        dialogue_history.append({"role": "assistant", "content": role_response})

        # 用户模拟器调用失败直接抛出异常（API级别错误）
        user_response = user_simulate(user_messages, persona_info, user_system_prompt, debug=debug)
        if not user_response or not user_response.strip() or len(user_response) > 500:
            break

        user_messages.append({"role": "assistant", "content": user_response})
        dialogue_history.append({"role": "user", "content": user_response})

    # 最后一轮角色回复
    if len(dialogue_history) > 0 and dialogue_history[-1]["role"] == "user":
        role_messages.append({"role": "user", "content": dialogue_history[-1]["content"]})
        try:
            final = role_play_chat(role_messages, persona_info, debug=debug)
            if len(final) <= 500 and len(set(final)) >= 5:
                dialogue_history.append({"role": "assistant", "content": final})
        except Exception:
            pass

    return {
        "persona_info": persona_info,
        "user_system_prompt": user_system_prompt,
        "dialogue": dialogue_history
    }


def generate_dialogue_continue(original_dialogue_data, debug=False):
    """基于原有对话续写"""
    persona_info = original_dialogue_data.get("persona_info", "")
    user_system_prompt = original_dialogue_data.get("user_system_prompt", "")
    original_dialogue = original_dialogue_data.get("dialogue", [])

    role_messages = []
    user_messages = []
    for msg in original_dialogue:
        if msg["role"] == "user":
            role_messages.append({"role": "user", "content": msg["content"]})
            user_messages.append({"role": "assistant", "content": msg["content"]})
        elif msg["role"] == "assistant":
            role_messages.append({"role": "assistant", "content": msg["content"]})
            user_messages.append({"role": "user", "content": msg["content"]})

    num_turns = 20
    continued_dialogue = []

    for turn in range(num_turns):
        last_role = original_dialogue[-1]["role"] if original_dialogue else "user"
        if (turn == 0 and last_role == "user") or (turn > 0 and continued_dialogue and continued_dialogue[-1]["role"] == "user"):
            # API调用失败会直接抛出异常（RuntimeError），让调用方知道具体错误
            role_response = role_play_chat(role_messages, persona_info, debug=debug)
            # 内容质量过滤（非致命，正常跳过该轮）
            if role_response in ("抱歉，没有收到回复。", "抱歉，我暂时无法回复，请稍后再试。"):
                print(f"  [续写 turn {turn}] 角色扮演返回无效内容，跳过: '{role_response}'")
                break
            if len(role_response) > 500 or len(set(role_response)) < 5:
                print(f"  [续写 turn {turn}] 角色扮演内容不符合质量要求(长度{len(role_response)}, 去重字符{len(set(role_response))}), 跳过")
                break
            role_messages.append({"role": "assistant", "content": role_response})
            user_messages.append({"role": "user", "content": role_response})
            continued_dialogue.append({"role": "assistant", "content": role_response})
        else:
            # 用户模拟器调用失败直接抛出异常（API级别错误）
            user_response = user_simulate(user_messages, persona_info, user_system_prompt, debug=debug)
            if not user_response or not user_response.strip() or len(user_response) > 500:
                break
            role_messages.append({"role": "user", "content": user_response})
            user_messages.append({"role": "assistant", "content": user_response})
            continued_dialogue.append({"role": "user", "content": user_response})

    # 最后一轮角色回复
    if continued_dialogue and continued_dialogue[-1]["role"] == "user":
        try:
            final = role_play_chat(role_messages, persona_info, debug=debug)
            if final not in ("抱歉，没有收到回复。",) and len(final) <= 500 and len(set(final)) >= 5:
                continued_dialogue.append({"role": "assistant", "content": final})
        except Exception:
            pass

    return {
        "persona_info": persona_info,
        "user_system_prompt": user_system_prompt,
        "original_dialogue": original_dialogue,
        "continued_dialogue": continued_dialogue,
        "total_turns": len(original_dialogue) + len(continued_dialogue)
    }


def stage1_generate():
    """阶段1：生成对话（多线程并发版）"""
    print("\n" + "=" * 60)
    print("阶段1：对话生成（多线程）")
    print("=" * 60)

    output_dir = os.path.dirname(GENERATE_OUTPUT_FILE)
    os.makedirs(output_dir, exist_ok=True)

    # ── 用于线程安全的结果收集和进度保存 ──
    results_dict = {}       # {idx: result_data}
    results_lock = Lock()
    save_counter = [0]
    error_counter = [0]     # 记录失败数量
    first_error = [None]    # 记录第一个错误信息

    def _save_progress(results_dict, existing_data):
        """线程安全的进度保存"""
        sorted_results = [results_dict[k] for k in sorted(results_dict.keys())]
        all_output = existing_data + sorted_results
        with open(GENERATE_OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(all_output, f, ensure_ascii=False, indent=2)
        return len(all_output)

    if GENERATE_MODE == "continue":
        print(f"模式: 续写对话")
        print(f"历史文件: {HISTORY_FILE}")
        print(f"并发线程: {STAGE1_MAX_WORKERS}")

        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            history_dialogues = [json.loads(line) for line in f if line.strip()]
        print(f"加载了 {len(history_dialogues)} 个历史对话")

        existing_data = []
        if os.path.exists(GENERATE_OUTPUT_FILE):
            try:
                with open(GENERATE_OUTPUT_FILE, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                print(f"已有 {len(existing_data)} 个已处理对话")
            except Exception:
                existing_data = []

        start_idx = len(existing_data)
        end_idx = min(start_idx + BATCH_SIZE, len(history_dialogues)) if BATCH_SIZE > 0 else len(history_dialogues)
        print(f"处理范围: {start_idx+1} ~ {end_idx}")

        task_items = [(idx, history_dialogues[idx]) for idx in range(start_idx, end_idx)]

        if not task_items:
            print("⚠️ 没有需要处理的对话")
            return GENERATE_OUTPUT_FILE

        # ── 先单线程验证第一条，确保模型服务正常 ──
        print("\n🔍 验证模型服务：正在测试第一条对话...")
        first_idx, first_data = task_items[0]
        try:
            first_result = generate_dialogue_continue(first_data, debug=False)
            if not first_result.get("continued_dialogue"):
                raise RuntimeError("第一条对话续写结果为空，模型可能返回了异常内容")
            print(f"✅ 模型服务验证通过，第一条对话续写成功（续写{len(first_result['continued_dialogue'])}轮）")
            results_dict[first_idx] = first_result
            save_counter[0] += 1
        except Exception as e:
            print(f"\n{'='*60}")
            print(f"❌ 阶段1致命错误：第一条对话生成失败！")
            print(f"{'='*60}")
            print(f"错误详情: {e}")
            print(f"\n可能原因:")
            print(f"  1. 角色扮演模型服务不可用（backend={assistant_client.backend}, "
                  f"model={assistant_client.model_name}）")
            print(f"  2. 用户模拟器服务不可用（model={user_client.model_name}）")
            print(f"  3. API Key 无效/过期，或模型名称错误")
            print(f"\n程序终止。请检查模型服务后重试。")
            sys.exit(1)

        remaining_items = task_items[1:]

        def continue_worker(args):
            idx, dialogue_data = args
            try:
                continued_data = generate_dialogue_continue(dialogue_data, debug=False)
                return idx, continued_data, None
            except Exception as e:
                error_msg = f"第 {idx+1} 个出错: {e}"
                print(error_msg)
                return idx, {
                    "persona_info": dialogue_data.get("persona_info", ""),
                    "user_system_prompt": dialogue_data.get("user_system_prompt", ""),
                    "original_dialogue": dialogue_data.get("dialogue", []),
                    "continued_dialogue": [],
                    "error": str(e)
                }, str(e)

        if remaining_items:
            with ThreadPoolExecutor(max_workers=STAGE1_MAX_WORKERS) as executor:
                futures = {executor.submit(continue_worker, item): item[0] for item in remaining_items}
                for future in tqdm(as_completed(futures), total=len(remaining_items), desc="对话生成(续写)"):
                    try:
                        idx, result, err = future.result()
                        with results_lock:
                            results_dict[idx] = result
                            save_counter[0] += 1
                            if err:
                                error_counter[0] += 1
                                if first_error[0] is None:
                                    first_error[0] = err
                            if save_counter[0] % SAVE_INTERVAL == 0:
                                total = _save_progress(results_dict, existing_data)
                                print(f"  已保存 {total} 条")
                    except Exception as e:
                        print(f"⚠️ 线程异常: {e}")

    else:  # scratch 模式
        print(f"模式: 从零开始生成")
        print(f"并发线程: {STAGE1_MAX_WORKERS}")

        with open(PERSONA_FILE, 'r', encoding='utf-8') as f:
            persona_pairs = json.load(f)
        print(f"加载了 {len(persona_pairs)} 对人设")

        existing_data = []
        if os.path.exists(GENERATE_OUTPUT_FILE):
            try:
                with open(GENERATE_OUTPUT_FILE, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
            except Exception:
                existing_data = []

        start_idx = len(existing_data)
        end_idx = min(start_idx + BATCH_SIZE, len(persona_pairs)) if BATCH_SIZE > 0 else len(persona_pairs)

        task_items = [(idx, persona_pairs[idx]) for idx in range(start_idx, end_idx)]

        if not task_items:
            print("⚠️ 没有需要处理的对话")
            return GENERATE_OUTPUT_FILE

        # ── 先单线程验证第一条，确保模型服务正常 ──
        print("\n🔍 验证模型服务：正在测试第一条对话...")
        first_idx, first_pair = task_items[0]
        try:
            first_result = generate_dialogue(first_pair["model_persona"], first_pair["user_persona"], debug=False)
            if not first_result.get("dialogue"):
                raise RuntimeError("第一条对话生成结果为空，模型可能返回了异常内容")
            print(f"✅ 模型服务验证通过，第一条对话生成成功（{len(first_result['dialogue'])}轮）")
            results_dict[first_idx] = first_result
            save_counter[0] += 1
        except Exception as e:
            print(f"\n{'='*60}")
            print(f"❌ 阶段1致命错误：第一条对话生成失败！")
            print(f"{'='*60}")
            print(f"错误详情: {e}")
            print(f"\n可能原因:")
            print(f"  1. 角色扮演模型服务不可用（backend={assistant_client.backend}, "
                  f"model={assistant_client.model_name}）")
            print(f"  2. 用户模拟器服务不可用（model={user_client.model_name}）")
            print(f"  3. API Key 无效/过期，或模型名称错误")
            print(f"\n程序终止。请检查模型服务后重试。")
            sys.exit(1)

        remaining_items = task_items[1:]

        def scratch_worker(args):
            idx, pair = args
            try:
                dialogue_data = generate_dialogue(pair["model_persona"], pair["user_persona"], debug=False)
                return idx, dialogue_data, None
            except Exception as e:
                error_msg = f"第 {idx+1} 对出错: {e}"
                print(error_msg)
                return idx, {
                    "persona_info": pair.get("model_persona", ""),
                    "user_system_prompt": pair.get("user_persona", ""),
                    "dialogue": [],
                    "error": str(e)
                }, str(e)

        if remaining_items:
            with ThreadPoolExecutor(max_workers=STAGE1_MAX_WORKERS) as executor:
                futures = {executor.submit(scratch_worker, item): item[0] for item in remaining_items}
                for future in tqdm(as_completed(futures), total=len(remaining_items), desc="对话生成(从零)"):
                    try:
                        idx, result, err = future.result()
                        with results_lock:
                            results_dict[idx] = result
                            save_counter[0] += 1
                            if err:
                                error_counter[0] += 1
                                if first_error[0] is None:
                                    first_error[0] = err
                            if save_counter[0] % SAVE_INTERVAL == 0:
                                total = _save_progress(results_dict, existing_data)
                                print(f"  已保存 {total} 条")
                    except Exception as e:
                        print(f"⚠️ 线程异常: {e}")

    # 最终保存
    total = _save_progress(results_dict, existing_data)
    print(f"✓ 阶段1完成，共 {total} 条对话，保存到: {GENERATE_OUTPUT_FILE}")

    # ── 生成质量检查 ──
    total_generated = len(results_dict)
    if total_generated > 0:
        if GENERATE_MODE == "continue":
            empty_count = sum(1 for r in results_dict.values() if not r.get("continued_dialogue"))
        else:
            empty_count = sum(1 for r in results_dict.values() if not r.get("dialogue"))
        
        fail_rate = empty_count / total_generated
        
        # 统计角色扮演回复的平均长度
        all_assistant_lengths = []
        for r in results_dict.values():
            if GENERATE_MODE == "continue":
                dialogues = r.get("continued_dialogue", [])
            else:
                dialogues = r.get("dialogue", [])
            for msg in dialogues:
                if msg.get("role") == "assistant":
                    all_assistant_lengths.append(len(msg["content"]))
        
        print(f"\n📊 Stage1 生成质量统计:")
        print(f"   总数: {total_generated}, 成功: {total_generated - empty_count}, 失败/空结果: {empty_count}")
        print(f"   失败率: {fail_rate:.1%}")
        if all_assistant_lengths:
            avg_len = sum(all_assistant_lengths) / len(all_assistant_lengths)
            max_len = max(all_assistant_lengths)
            min_len = min(all_assistant_lengths)
            print(f"   角色扮演回复统计: 共{len(all_assistant_lengths)}条, 平均长度{avg_len:.1f}字符, 最短{min_len}, 最长{max_len}")
        else:
            print(f"   角色扮演回复统计: 无有效回复")
        if error_counter[0] > 0:
            print(f"   API错误数: {error_counter[0]}")
            print(f"   首个错误: {first_error[0]}")
        
        if fail_rate > 0.5:
            print(f"\n{'='*60}")
            print(f"❌ 阶段1失败率过高 ({fail_rate:.1%} > 50%)，程序终止！")
            print(f"{'='*60}")
            if first_error[0]:
                print(f"首个错误详情: {first_error[0]}")
            print(f"请检查模型服务和API配置后重试。")
            sys.exit(1)

    return GENERATE_OUTPUT_FILE
