#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Model-calling layer copied from different_user_simulation/run_dynsess_eval.py.

Three call paths, matching the working pipeline exactly:
  - llm_call_user   : 豆包用户模拟器，走 ark 的 OpenAI client。
  - llm_call_role   : 角色扮演模型。doubao 模型走 ark；其余模型走伏羲 V2 签名头 API。
  - llm_call_eval   : judge 评估模型，走伏羲 Bearer token API。

凭证与 URL 直接沿用 pipeline 中的内嵌配置，运行时无需再设置环境变量。
"""

from __future__ import annotations

import hashlib
import json
import random
import string
import time
from typing import Optional

import requests
from openai import OpenAI

# ─────────────── 凭证 / URL（沿用 run_dynsess_eval.py） ─────────────── #
ARK_API_KEY = os.environ.get("ARK_API_KEY", "")
ARK_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"

V2_API_APP_ID = os.environ.get("V2_API_APP_ID", "")
V2_API_APP_KEY = os.environ.get("V2_API_APP_KEY", "")
V2_API_PROJECT_ID = os.environ.get("V2_API_PROJECT_ID", "")
V2_API_URL = "https://aigc-api.fuxi.netease.com/v1/chat/completions"

EVAL_API_URL = "https://aigc-api.fuxi.netease.com/v1/chat/completions"
EVAL_API_BEARER_TOKEN = os.environ.get("DYNS_EVAL_API_TOKEN", "")

ark_client = OpenAI(base_url=ARK_BASE_URL, api_key=ARK_API_KEY)


def _v2_signed_headers(app_id=V2_API_APP_ID, app_key=V2_API_APP_KEY, project_id=V2_API_PROJECT_ID) -> dict:
    """生成伏羲 V2 API 所需的签名头。"""
    nonce = "".join(random.choices(string.ascii_letters + string.digits, k=10))
    timestamp = str(int(time.time()))
    str2sign = f"appId={app_id}&nonce={nonce}&timestamp={timestamp}&appkey={app_key}"
    sign = hashlib.md5(str2sign.encode("utf-8")).hexdigest().upper()
    return {
        "appId": app_id,
        "nonce": nonce,
        "timestamp": timestamp,
        "sign": sign,
        "version": "v2",
        "Content-Type": "application/json",
        "projectid": project_id,
    }


def llm_call_user(messages: list[dict], temperature: float = 1.0, max_tokens: int = 200, max_retries: int = 5) -> str:
    """豆包用户模拟器（ark OpenAI client）。"""
    last_error = None
    for attempt in range(max_retries):
        try:
            response = ark_client.chat.completions.create(
                model="doubao-1-5-pro-32k-character-250715",
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=30,
                extra_body={"caching": {"type": "enabled"}, "thinking": {"type": "disabled"}},
            )
            return response.choices[0].message.content
        except Exception as exc:
            last_error = f"豆包API调用出错 (attempt {attempt + 1}/{max_retries}): {exc}"
            print(last_error, flush=True)
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    raise RuntimeError(last_error)


def llm_call_role(
    messages: list[dict],
    model_name: str,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    extra_body: Optional[dict] = None,
    max_retries: int = 5,
) -> str:
    """角色扮演模型：doubao 走 ark，其余走伏羲 V2 签名头 API。"""
    is_doubao = "doubao" in model_name.lower()
    if is_doubao:
        last_error = None
        for attempt in range(max_retries):
            try:
                response = ark_client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=30,
                    extra_body={"caching": {"type": "enabled"}, "thinking": {"type": "disabled"}},
                )
                return response.choices[0].message.content
            except Exception as exc:
                last_error = f"豆包API调用出错 (attempt {attempt + 1}/{max_retries}): {exc}"
                print(last_error, flush=True)
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
        raise RuntimeError(last_error)

    body = {
        "max_tokens": max_tokens,
        "model": model_name,
        "temperature": temperature,
        "top_p": 0.95,
        "messages": messages,
        "stream": False,
    }
    # 部分 bedrock 模型（如 claude）不允许同时指定 temperature 与 top_p。
    if "claude" in model_name.lower() or "anthropic" in model_name.lower():
        body.pop("top_p", None)
    if extra_body:
        body.update(extra_body)
    elif "gemini" in model_name.lower():
        body["thinkingConfig"] = {"includeThoughts": True, "thinkingLevel": "low"}

    last_error = None
    for attempt in range(max_retries):
        try:
            headers = _v2_signed_headers()
            response = requests.post(V2_API_URL, json=body, headers=headers, timeout=120)
            data = response.json()
            if "choices" in data and data["choices"]:
                content = data["choices"][0]["message"]["content"]
                if content:
                    return str(content).strip()
            last_error = f"V2 API 返回异常: {json.dumps(data, ensure_ascii=False)[:500]}"
            print(last_error, flush=True)
        except Exception as exc:
            last_error = f"V2 API调用出错 (attempt {attempt + 1}/{max_retries}): {exc}"
            print(last_error, flush=True)
        if attempt < max_retries - 1:
            time.sleep(2 ** attempt)
    raise RuntimeError(last_error)


def llm_call_eval(
    messages: list[dict],
    model_name: str = "gemini-3-flash-preview",
    temperature: float = 0.7,
    max_tokens: int = 4096,
    max_retries: int = 5,
) -> str:
    """judge 评估模型（伏羲 Bearer token）。"""
    body = {
        "max_tokens": max_tokens,
        "model": model_name,
        "temperature": temperature,
        "top_p": 0.95,
        "messages": messages,
    }
    headers = {"Authorization": f"Bearer {EVAL_API_BEARER_TOKEN}", "Content-Type": "application/json"}
    last_error = None
    for attempt in range(max_retries):
        try:
            resp = requests.post(EVAL_API_URL, json=body, headers=headers, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            if "choices" in data and data["choices"]:
                content = data["choices"][0]["message"]["content"]
                if content:
                    return str(content)
            last_error = f"评估API返回异常: {json.dumps(data, ensure_ascii=False)[:500]}"
            print(last_error, flush=True)
        except Exception as exc:
            last_error = f"评估API请求失败 (attempt {attempt + 1}/{max_retries}): {exc}"
            print(last_error, flush=True)
        if attempt < max_retries - 1:
            time.sleep(2 ** attempt)
    raise RuntimeError(last_error)
