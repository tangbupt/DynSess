#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一键启动脚本：对话生成 + 格式转换 + 自动评估
Usage: python run_pipeline.py
"""

import json
import os
import re
import sys
import time
import random
import requests
import numpy as np
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from tqdm import tqdm
from threading import Lock

# ╔══════════════════════════════════════════════════════════════════╗
# ║                    ★★★ 统一配置区 ★★★                          ║
# ╚══════════════════════════════════════════════════════════════════╝

# ─────────────── 模型配置 ───────────────
LOCAL_VLLM_URL = "http://localhost:88/v1/chat/completions"  # 本地VLLM角色扮演模型地址
LOCAL_MODEL_NAME = "persona_general"                           # 本地VLLM模型名称

ARK_API_KEY = os.environ.get("ARK_API_KEY", "")           # 豆包API Key（用户模拟器）
USER_MODEL_NAME = "doubao-1-5-pro-32k-character-250715"        # 豆包用户模拟器模型

# ─────────────── Assistant模型配置（角色扮演模型选择） ───────────────
ASSISTANT_MODEL = "local"  # 选择assistant模型类型: "local" = 本地VLLM模型, "api" = 外部API模型

# 外部API模型配置（仅当 ASSISTANT_MODEL = "api" 时生效）
ASSISTANT_API_KEY = os.environ.get("ASSISTANT_API_KEY", os.environ.get("ARK_API_KEY", ""))                        # 外部API Key
ASSISTANT_API_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"  # 外部API Base URL
ASSISTANT_API_MODEL_NAME = "doubao-1-5-pro-32k-character-250715"     # 外部API模型名称

# ─────────────── 字数限制模型配置 ───────────────
# 在此列表中的模型，会在 system prompt 中加入字数限制（回复不超过50个字）
WORD_LIMIT_MODELS = ["gpt-5.1", "gpt-5.4", "gemini-3-pro-preview"]  # 需要限制字数的模型列表
WORD_LIMIT_PROMPT = "你的本次回复，希望控制字数在50个字左右"  # 字数限制提示语

EVAL_API_URL = os.environ.get("EVAL_API_URL", "https://aigc-api.fuxi.netease.com/v1/chat/completions")  # 评估API地址（可用环境变量覆盖）
EVAL_MODEL = "gemini-3-flash-preview"                                     # 评估模型
EVAL_API_BEARER_TOKEN = os.environ.get("DYNS_EVAL_API_TOKEN", "")          # 评估API Token

# ─────────────── 输入文件（需要用户提供） ───────────────
GENERATE_MODE = "continue"  # 对话生成模式: "continue" = 续写模式, "scratch" = 从零开始
HISTORY_FILE = "./data/test_dialogue_0424.jsonl"           # 续写模式 - 历史对话输入文件
PERSONA_FILE = "./data/personas.json"        # 从零模式 - 人设数据文件（需用户提供，格式: [{"model_persona":..., "user_persona":...}]）
JUDGE_PROMPT_FILE = "./dynsess_rubrics.py"                     # 评判Prompt文件

# ─────────────── 运行参数 ───────────────
MAX_WORKERS = 10      # 评估并发线程数
MAX_RETRIES = 5       # API重试次数
RETRY_DELAY = 5       # 重试间隔（秒）
SAVE_INTERVAL = 5     # 每N条保存一次进度
BATCH_SIZE = 100     # 每批处理数量（0=全部）
STAGE1_MAX_WORKERS = 10   # Stage1 对话生成并发线程数

# ─────────────── 跳过Stage1配置 ───────────────
SKIP_STAGE1 = False  # 是否跳过Stage1（对话生成），直接从Stage2开始
PRE_GENERATE_OUTPUT_FILE = "./evaluate/generate/dialogues_persona_general_20260319_114616.json"  # 跳过Stage1时，使用此文件作为输入（Stage1的预生成结果）
# 例如: PRE_GENERATE_OUTPUT_FILE = "./evaluate/generate/dialogues_persona_general_20260319_100000.json"

# ─────────────── 输出根目录 ───────────────
OUTPUT_ROOT = "./evaluate"

# ─────────────── 自动生成的输出路径（无需手动修改） ───────────────
_RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
_RUN_TAG = f"{LOCAL_MODEL_NAME}_{_RUN_TIMESTAMP}"

# 阶段1：对话生成输出
GENERATE_OUTPUT_FILE = os.path.join(OUTPUT_ROOT, "generate", f"dialogues_{_RUN_TAG}.json")
# 阶段2：格式转换输出
FORMATTED_FILE = os.path.join(OUTPUT_ROOT, "format", f"dialogues_{_RUN_TAG}_format.json")
# 阶段3：评估结果输出目录
EVAL_OUTPUT_DIR = os.path.join(OUTPUT_ROOT, "eval_result", _RUN_TAG)
EVAL_PROGRESS_FILE = os.path.join(EVAL_OUTPUT_DIR, "progress.json")
EVAL_FINAL_FILE = os.path.join(EVAL_OUTPUT_DIR, f"eval_{_RUN_TAG}_final.json")
EVAL_LOG_FILE = os.path.join(EVAL_OUTPUT_DIR, f"eval_{_RUN_TAG}.log")


# ╔══════════════════════════════════════════════════════════════════╗
# ║              阶段1：对话生成 - 工具函数                          ║
# ╚══════════════════════════════════════════════════════════════════╝

ark_client = OpenAI(
    base_url='https://ark.cn-beijing.volces.com/api/v3',
    api_key=ARK_API_KEY,
)

# Assistant API 客户端（仅当 ASSISTANT_MODEL = "api" 时使用）
assistant_api_client = OpenAI(
    base_url=ASSISTANT_API_BASE_URL,
    api_key=ASSISTANT_API_KEY,
)


def llm_call_user(messages, temperature=1.0, max_tokens=200):
    """调用豆包API（用户模拟器）"""
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    try:
        response = ark_client.chat.completions.create(
            model=USER_MODEL_NAME,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body={
                "caching": {"type": "enabled"},
                "thinking": {"type": "disabled"}
            }
        )
        return response.choices[0].message.content
    except Exception as e:
        error_msg = f"豆包API调用出错: {e}"
        print(error_msg)
        raise RuntimeError(error_msg) from e


def llm_call_local(messages, temperature=0.7, max_tokens=1024, max_retries=3):
    """调用本地VLLM模型（角色扮演）"""
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]

    body = {
        "model": LOCAL_MODEL_NAME,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": 0.95,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False}
    }
    headers = {"Content-Type": "application/json"}

    last_error = None
    for attempt in range(max_retries):
        try:
            resp = requests.post(url=LOCAL_VLLM_URL, json=body,
                                 headers=headers, timeout=(30, 60))
            if resp.status_code != 200:
                last_error = f"本地API状态码: {resp.status_code}, 响应: {resp.text[:500]}"
                print(last_error)
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise RuntimeError(last_error)
            data = resp.json()
            if 'choices' in data and len(data['choices']) > 0:
                return data['choices'][0]['message']['content']
            last_error = f"本地API返回数据无choices: {json.dumps(data, ensure_ascii=False)[:500]}"
            print(last_error)
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(last_error)
        except (requests.exceptions.ConnectionError,
                requests.exceptions.Timeout) as e:
            last_error = f"本地API连接/超时 (attempt {attempt+1}): {e}"
            print(last_error)
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise RuntimeError(last_error) from e
        except RuntimeError:
            raise
        except Exception as e:
            last_error = f"本地API未知错误: {e}"
            print(last_error)
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise RuntimeError(last_error) from e
    raise RuntimeError(last_error or "本地API调用失败（未知原因）")


def llm_call_assistant_api(messages, temperature=0.7, max_tokens=1024):
    """调用外部API模型（角色扮演）- 使用豆包API格式"""
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    try:
        response = assistant_api_client.chat.completions.create(
            model=ASSISTANT_API_MODEL_NAME,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body={
                "caching": {"type": "enabled"},
                "thinking": {"type": "disabled"}
            }
        )
        return response.choices[0].message.content
    except Exception as e:
        error_msg = f"外部API调用出错: {e}"
        print(error_msg)
        raise RuntimeError(error_msg) from e


# ─────────────── V2 API 配置（伏羲 API） ───────────────
import hashlib
import string

V2_API_APP_ID = os.environ.get("V2_API_APP_ID", "")
V2_API_APP_KEY = os.environ.get("V2_API_APP_KEY", "")
V2_API_PROJECT_ID = os.environ.get("V2_API_PROJECT_ID", "")
V2_API_URL = "https://aigc-api.fuxi.netease.com/v1/chat/completions"
V2_API_MODEL_NAME = "gemini-3-flash-preview"  # 默认模型名，可以通过参数覆盖


def _v2_signed_headers(app_id=V2_API_APP_ID, app_key=V2_API_APP_KEY, project_id=V2_API_PROJECT_ID):
    """生成 V2 API (伏羲 API) 所需的签名头"""
    nonce = ''.join(random.choices(string.ascii_letters + string.digits, k=10))
    timestamp = str(int(time.time()))
    str2sign = f"appId={app_id}&nonce={nonce}&timestamp={timestamp}&appkey={app_key}"
    sign = hashlib.md5(str2sign.encode('utf-8')).hexdigest().upper()
    
    return {
        "appId": app_id,
        "nonce": nonce,
        "timestamp": timestamp,
        "sign": sign,
        "version": "v2",
        "Content-Type": "application/json",
        "projectid": project_id
    }


def llm_call_assistant_api_v2(messages, temperature=0.7, max_tokens=1024, model_name=None):
    """
    智能调用外部API模型（角色扮演）
    - 如果是 doubao 模型，使用原来的豆包 API 格式
    - 如果不是 doubao 模型，使用伏羲 V2 API 格式
    
    Args:
        messages: 消息列表
        temperature: 温度参数
        max_tokens: 最大 token 数
        model_name: 模型名称，如果为 None 则使用 ASSISTANT_API_MODEL_NAME
    
    Returns:
        模型回复内容
    """
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    
    # 确定使用的模型名称
    actual_model_name = model_name if model_name else ASSISTANT_API_MODEL_NAME
    
    # 判断是否是 doubao 模型
    is_doubao_model = "doubao" in actual_model_name.lower()
    
    if is_doubao_model:
        # 使用原来的豆包 API
        try:
            response = assistant_api_client.chat.completions.create(
                model=actual_model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body={
                    "caching": {"type": "enabled"},
                    "thinking": {"type": "disabled"}
                }
            )
            return response.choices[0].message.content
        except Exception as e:
            error_msg = f"豆包API调用出错: {e}"
            print(error_msg)
            raise RuntimeError(error_msg) from e
    else:
        # 使用伏羲 V2 API
        headers = _v2_signed_headers()
        body = {
            "max_tokens": max_tokens,
            "model": actual_model_name,
            "temperature": temperature,
            "top_p": 0.95,
            "messages": messages,
            "stream": False,
        }

        if "gemini3" in actual_model_name.lower():
             body['thinkingConfig'] = {
                   "includeThoughts": True,
                   "thinkingLevel": "low",
               }

        
        try:
            response = requests.post(V2_API_URL, json=body, headers=headers, timeout=60)
            data = response.json()
            if 'choices' in data and data['choices']:
                return data['choices'][0]['message']['content']
            error_msg = f"V2 API 返回异常: {json.dumps(data, ensure_ascii=False)[:500]}"
            print(error_msg)
            raise RuntimeError(error_msg)
        except RuntimeError:
            raise
        except Exception as e:
            error_msg = f"V2 API调用出错: {e}"
            print(error_msg)
            raise RuntimeError(error_msg) from e


def user_simulate(messages, persona_info, user_system_prompt, debug=False):
    """用户模拟器（豆包API）"""
    processed = persona_info
    if persona_info.startswith("请你扮演以下人设："):
        processed = persona_info[len("请你扮演以下人设："):]

    full_system_prompt = f"""你是一个用户模拟器。你的任务是扮演一个真实的用户，与一个AI角色扮演模型进行对话。
**请时刻牢记：你只是用户，绝对不要扮演对方的角色。**
### 关键行为准则：
1. 像一个懒惰的用户一样，只被动回答对方的问题，或者对对方的话做简短评价。**绝对不要**主动开启新话题或频繁提问，**不要**反问对方（除非非常必要）。
2. **口语化**：说话要像真人发信息一样自然、随意，甚至可以带一点敷衍。
3. **回复简短**：回复长度严格控制在 **10-20个字以内**，通常 **1句话** 即可。

### 你的设定（User Persona）
请完全沉浸在以下人设中：
[
{user_system_prompt}
]

---

你正在与之对话的AI（对方）拥有以下人设（注意：这是对方的设定，**不是你的**，你只需要知道他在扮演这个人即可）：
[
{processed}
]

---

### 当前任务
请根据**你的设定**和**对话上下文**，给出一个简短、自然、被动的回复。"""

    api_messages = [{"role": "system", "content": full_system_prompt}]
    for msg in messages[1:]:
        api_messages.append(msg)
    response = llm_call_user(api_messages, temperature=1.0, max_tokens=200)
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
    if ASSISTANT_MODEL == "api" and ASSISTANT_API_MODEL_NAME in WORD_LIMIT_MODELS:
        system_prompt = f"{WORD_LIMIT_PROMPT}\n\n{system_prompt}"
    
    api_messages = [{"role": "system", "content": system_prompt}]
    api_messages.extend(messages)
    print(api_messages)
    
    # 根据配置选择调用本地模型或外部API模型
    if ASSISTANT_MODEL == "api":
        # 使用 v2 函数：自动判断 doubao 模型用原 API，其他模型用伏羲 V2 API
        response = llm_call_assistant_api_v2(api_messages, temperature=0.7, max_tokens=1024)
    else:  # 默认使用本地模型
        response = llm_call_local(api_messages, temperature=0.7, max_tokens=1024)
    
    print("response:", response)
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


# ╔══════════════════════════════════════════════════════════════════╗
# ║              阶段1：对话生成 - 主流程                            ║
# ╚══════════════════════════════════════════════════════════════════╝



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

        # fake 逻辑
        history_dialogues = history_dialogues[:BATCH_SIZE]

        existing_data = []
        if os.path.exists(GENERATE_OUTPUT_FILE):
            try:
                with open(GENERATE_OUTPUT_FILE, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                print(f"已有 {len(existing_data)} 个已处理对话")
            except:
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
            if ASSISTANT_MODEL == "api":
                print(f"  1. 外部API服务不可用（{ASSISTANT_API_BASE_URL}）")
                print(f"  2. API Key无效或已过期")
                print(f"  3. 模型名称错误（{ASSISTANT_API_MODEL_NAME}）")
            else:
                print(f"  1. 本地VLLM服务未启动或不可达（{LOCAL_VLLM_URL}）")
                print(f"  2. 模型名称错误（{LOCAL_MODEL_NAME}）")
            print(f"  3. 用户模拟器API不可用（豆包API）")
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
            except:
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
            if ASSISTANT_MODEL == "api":
                print(f"  1. 外部API服务不可用（{ASSISTANT_API_BASE_URL}）")
                print(f"  2. API Key无效或已过期")
                print(f"  3. 模型名称错误（{ASSISTANT_API_MODEL_NAME}）")
            else:
                print(f"  1. 本地VLLM服务未启动或不可达（{LOCAL_VLLM_URL}）")
                print(f"  2. 模型名称错误（{LOCAL_MODEL_NAME}）")
            print(f"  3. 用户模拟器API不可用（豆包API）")
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


# ╔══════════════════════════════════════════════════════════════════╗
# ║              阶段2：格式转换                                     ║
# ╚══════════════════════════════════════════════════════════════════╝

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
    return FORMATTED_FILE


# ╔══════════════════════════════════════════════════════════════════╗
# ║              阶段3：自动评估                                     ║
# ╚══════════════════════════════════════════════════════════════════╝

def write_log(msg):
    """写日志"""
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{ts}] {msg}"
    print(line)
    os.makedirs(EVAL_OUTPUT_DIR, exist_ok=True)
    with open(EVAL_LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(line + "\n")


def llm_call_eval(messages, temperature=0.7, max_tokens=4096):
    """调用评估API"""
    body = {
        "max_tokens": max_tokens,
        "model": EVAL_MODEL,
        "temperature": temperature,
        "top_p": 0.95,
        "messages": messages,
    }
    headers = {
        "Authorization": f"Bearer {EVAL_API_BEARER_TOKEN}",
        "Content-Type": "application/json",
    }
    try:
        resp = requests.post(url=EVAL_API_URL, json=body, headers=headers, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        if "choices" in data and data["choices"]:
            content = data["choices"][0]["message"]["content"]
            return str(content) if content is not None else "null"
        return "null"
    except Exception as e:
        write_log(f"⚠️ 评估API请求失败: {e}")
        return "null"


def extract_json_from_response(response):
    """从LLM响应中提取JSON"""
    if not isinstance(response, str):
        return None
    try:
        return json.loads(response)
    except:
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
        except:
            pass
        if '\\n' in s or '\\"' in s:
            try:
                return json.loads(s.encode().decode('unicode_escape'))
            except:
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
            except:
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
        resp = llm_call_eval([{"role": "user", "content": prompt}])
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
            "eval_model": EVAL_MODEL,
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


# ╔══════════════════════════════════════════════════════════════════╗
# ║                    主入口                                        ║
# ╚══════════════════════════════════════════════════════════════════╝

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
        gen_file = stage1_generate()

    # 阶段2：格式转换
    fmt_file = stage2_format(gen_file)

    # 阶段3：自动评估
    result_file = stage3_evaluate(fmt_file)

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

