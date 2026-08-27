#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DynSess 评测统一配置：模型参数、路径、运行参数与 API 客户端。

所有凭证通过环境变量加载（默认空串），不硬编码任何密钥。
所有路径基于仓库根目录（REPO_ROOT），不依赖当前工作目录。
"""
import os
import sys
from datetime import datetime
from llm import LLMClient

# ╔══════════════════════════════════════════════════════════════════╗
# ║                    ★★★ 统一配置区 ★★★                          ║
# ╚══════════════════════════════════════════════════════════════════╝

# ─────────────── 模型配置 ───────────────
LOCAL_VLLM_URL = "http://localhost:88/v1/chat/completions"  # 本地VLLM角色扮演模型地址
LOCAL_MODEL_NAME = "persona_general"                           # 本地VLLM模型名称

ARK_API_KEY = os.environ.get("ARK_API_KEY", "")           # 豆包API Key（用户模拟器）
USER_MODEL_NAME = "doubao-1-5-pro-32k-character-250715"        # 豆包用户模拟器模型
USER_API_BASE_URL = os.environ.get("USER_API_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")  # 用户模拟器API Base URL

# ─────────────── Assistant模型配置（角色扮演模型选择） ───────────────
ASSISTANT_MODEL = "local"  # 选择assistant模型类型: "local" = 本地VLLM模型, "api" = 外部API模型

# 外部API模型配置（仅当 ASSISTANT_MODEL = "api" 时生效）
ASSISTANT_API_KEY = os.environ.get("ASSISTANT_API_KEY", os.environ.get("ARK_API_KEY", ""))                        # 外部API Key
ASSISTANT_API_BASE_URL = os.environ.get("ASSISTANT_API_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")  # 外部API Base URL
ASSISTANT_API_MODEL_NAME = "doubao-1-5-pro-32k-character-250715"     # 外部API模型名称
# 角色扮演API鉴权方式: "bearer" = 标准 OpenAI 兼容（默认）, "signed" = appId/appKey/projectId 签名网关
ASSISTANT_API_AUTH = os.environ.get("ASSISTANT_API_AUTH", "bearer")

# ─────────────── 字数限制模型配置 ───────────────
# 在此列表中的模型，会在 system prompt 中加入字数限制（回复不超过50个字）
WORD_LIMIT_MODELS = ["gpt-5.1", "gpt-5.4", "gemini-3-pro-preview"]  # 需要限制字数的模型列表
WORD_LIMIT_PROMPT = "你的本次回复，希望控制字数在50个字左右"  # 字数限制提示语

EVAL_API_URL = os.environ.get("EVAL_API_URL", "")  # 评估API地址：OpenAI 兼容的评判端点（需通过环境变量提供）
EVAL_MODEL = "gemini-3-flash-preview"                                     # 评估模型
EVAL_API_BEARER_TOKEN = os.environ.get("DYNS_EVAL_API_TOKEN", "")          # 评估API Token

# ─────────────── 路径（基于仓库根目录，不依赖当前工作目录） ───────────────
# 本脚本位于 <repo_root>/eval/，因此仓库根目录 = 脚本所在目录的上一级
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ─────────────── 输入文件（需要用户提供） ───────────────
GENERATE_MODE = "continue"  # 对话生成模式: "continue" = 续写模式, "scratch" = 从零开始
HISTORY_FILE = os.path.join(REPO_ROOT, "data", "test_dialogue_0424.jsonl")           # 续写模式 - 历史对话输入文件
PERSONA_FILE = os.path.join(REPO_ROOT, "data", "personas.json")        # 从零模式 - 人设数据文件（需用户提供，格式: [{"model_persona":..., "user_persona":...}]）
JUDGE_PROMPT_FILE = os.path.join(REPO_ROOT, "prompt", "session_level.py")                     # 评判Prompt文件

# ─────────────── 用户模拟器 prompt 配置 ───────────────
# 用户模拟器风格: "passive"(懒惰被动,默认) | "balanced"(均衡真实) | "proactive"(主动激进)
# prompt 内容隔离在 prompt/user_sim_prompt.py，stage1 只调用 build_user_simulator_prompt
USER_SIM_STYLE = "passive"
sys.path.insert(0, os.path.join(REPO_ROOT, "prompt"))
from user_sim_prompt import build_user_simulator_prompt  # noqa: E402

# ─────────────── 运行参数 ───────────────
MAX_WORKERS = 10      # 评估并发线程数
MAX_RETRIES = 5       # API重试次数
RETRY_DELAY = 5       # 重试间隔（秒）
SAVE_INTERVAL = 5     # 每N条保存一次进度
BATCH_SIZE = 100     # 每批处理数量（0=全部）
STAGE1_MAX_WORKERS = 10   # Stage1 对话生成并发线程数

# ─────────────── 跳过Stage1配置 ───────────────
SKIP_STAGE1 = False  # 是否跳过Stage1（对话生成），直接从Stage2开始
PRE_GENERATE_OUTPUT_FILE = os.path.join(REPO_ROOT, "evaluate", "generate", "dialogues_persona_general_20260319_114616.json")  # 跳过Stage1时，使用此文件作为输入（Stage1的预生成结果）
# 例如: PRE_GENERATE_OUTPUT_FILE = os.path.join(REPO_ROOT, "evaluate", "generate", "dialogues_persona_general_20260319_100000.json")

# ─────────────── 输出根目录 ───────────────
OUTPUT_ROOT = os.path.join(REPO_ROOT, "evaluate")

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

# ─────────────── 签名网关配置（ASSISTANT_API_AUTH=signed 时使用） ───────────────
V2_API_APP_ID = os.environ.get("V2_API_APP_ID", "")
V2_API_APP_KEY = os.environ.get("V2_API_APP_KEY", "")
V2_API_PROJECT_ID = os.environ.get("V2_API_PROJECT_ID", "")
V2_API_URL = os.environ.get("V2_API_URL", "")  # 签名网关地址（ASSISTANT_API_AUTH=signed 时需通过环境变量提供）
V2_API_MODEL_NAME = "gemini-3-flash-preview"  # 默认模型名，可以通过参数覆盖

# ─────────────── LLM 客户端 ───────────────
# 鉴权与模型选择封装在 LLMClient 内（见 llm.py），各 stage 只调用 client.chat(messages)。
# 所有客户端在构造时不发起网络请求，可安全在 import 期构建。

# 用户模拟器（豆包，OpenAI 兼容 Bearer）
user_client = LLMClient(
    backend="openai",
    model_name=USER_MODEL_NAME,
    base_url=USER_API_BASE_URL,
    api_key=ARK_API_KEY,
)

# 角色扮演模型：按 ASSISTANT_MODEL（local/api）+ ASSISTANT_API_AUTH（bearer/signed）选择后端
if ASSISTANT_MODEL == "local":
    assistant_client = LLMClient(
        backend="vllm",
        model_name=LOCAL_MODEL_NAME,
        base_url=LOCAL_VLLM_URL,
        max_retries=3,
    )
else:  # ASSISTANT_MODEL == "api"
    if ASSISTANT_API_AUTH == "signed":
        assistant_client = LLMClient(
            backend="signed",
            model_name=ASSISTANT_API_MODEL_NAME,
            base_url=V2_API_URL,
            app_id=V2_API_APP_ID,
            app_key=V2_API_APP_KEY,
            project_id=V2_API_PROJECT_ID,
        )
    else:  # bearer（默认）
        assistant_client = LLMClient(
            backend="openai",
            model_name=ASSISTANT_API_MODEL_NAME,
            base_url=ASSISTANT_API_BASE_URL,
            api_key=ASSISTANT_API_KEY,
        )

# 评判端点（OpenAI 兼容 Bearer；失败返回 "null"，不抛异常）
eval_client = LLMClient(
    backend="bearer",
    model_name=EVAL_MODEL,
    base_url=EVAL_API_URL,
    api_key=EVAL_API_BEARER_TOKEN,
    timeout=120,
    raise_on_error=False,
)
