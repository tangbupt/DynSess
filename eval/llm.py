#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""统一 LLM 客户端：封装鉴权与模型选择，对外只暴露 model_name 与 chat(messages)。

所有凭证 / 端点 / 鉴权方式在构造时注入，调用方只需::

    client = LLMClient(backend="openai", model_name=..., base_url=..., api_key=...)
    reply = client.chat(messages, temperature=0.7, max_tokens=1024)

支持四种后端（backend=）：
- ``"openai"``  : 标准 OpenAI 兼容接口（Bearer 鉴权，走 OpenAI SDK）。
                  用于用户模拟器与外部 API 角色扮演模型。
- ``"vllm"``    : 本地 vLLM / SGLang 等推理服务（裸 requests，带指数退避重试）。
                  用于本地角色扮演模型。
- ``"signed"``  : appId/appKey/projectId + MD5 签名网关（裸 requests + 签名头）。
                  用于走签名网关的外部角色扮演模型。
- ``"bearer"``  : OpenAI 兼容的裸 requests 端点（Bearer 鉴权）。
                  用于评判端点；默认 raise_on_error=False，失败返回 "null"。
"""
import json
import time
import random
import hashlib
import string

import requests
from openai import OpenAI


def _is_gemini3(model_name):
    """gemini3 系列模型启用轻度思考（thinkingConfig）。"""
    return "gemini3" in (model_name or "").lower()


class LLMClient:
    """统一 LLM 调用客户端。

    对外只暴露 :attr:`model_name` 与 :meth:`chat`；鉴权、重试、模型选择
    等细节全部封装在内部按 backend 分派。
    """

    def __init__(self, backend, model_name, base_url="", api_key="",
                 app_id="", app_key="", project_id="",
                 max_retries=3, timeout=60, raise_on_error=True,
                 error_log=None):
        """
        Args:
            backend: 后端类型 ("openai" | "vllm" | "signed" | "bearer")。
            model_name: 默认模型名（可通过 chat(model_name=) 单次覆盖）。
            base_url: 服务地址（openai/vllm/signed/bearer 均用此字段）。
            api_key: Bearer 鉴权密钥（openai/bearer 用）。
            app_id/app_key/project_id: signed 网关三件套。
            max_retries: vllm 后端的失败重试次数。
            timeout: 请求超时秒数。
            raise_on_error: True=失败抛 RuntimeError；False=失败返回 "null"。
            error_log: 可选的错误日志回调 ``callable(str)``。
        """
        self.backend = backend
        self.model_name = model_name
        self.base_url = base_url
        self.api_key = api_key
        self.app_id = app_id
        self.app_key = app_key
        self.project_id = project_id
        self.max_retries = max_retries
        self.timeout = timeout
        self.raise_on_error = raise_on_error
        self.error_log = error_log

        # openai 后端复用一个 OpenAI SDK 客户端；其余后端用裸 requests。
        self._client = OpenAI(base_url=base_url, api_key=api_key) \
            if backend == "openai" else None

    # ─────────────────────── 对外接口 ───────────────────────
    def chat(self, messages, temperature=0.7, max_tokens=1024, model_name=None):
        """发起一次对话请求，返回模型回复文本。

        Args:
            messages: 消息列表（或纯字符串，自动包成单条 user 消息）。
            temperature: 采样温度。
            max_tokens: 最大生成 token 数。
            model_name: 单次覆盖模型名，默认用 self.model_name。

        Returns:
            模型回复字符串；raise_on_error=False 且出错时返回 "null"。
        """
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        model = model_name or self.model_name
        try:
            if self.backend == "openai":
                return self._call_openai(messages, temperature, max_tokens, model)
            if self.backend == "vllm":
                return self._call_vllm(messages, temperature, max_tokens, model)
            if self.backend == "signed":
                return self._call_signed(messages, temperature, max_tokens, model)
            if self.backend == "bearer":
                return self._call_bearer(messages, temperature, max_tokens, model)
            raise ValueError(f"未知 backend: {self.backend}")
        except Exception as e:
            if self.error_log:
                self.error_log(f"⚠️ LLM 调用失败 ({self.backend}/{model}): {e}")
            if self.raise_on_error:
                raise RuntimeError(f"LLM 调用出错 ({self.backend}/{model}): {e}") from e
            return "null"

    # ─────────────────────── openai 兼容（SDK） ───────────────────────
    def _call_openai(self, messages, temperature, max_tokens, model):
        extra_body = {
            "caching": {"type": "enabled"},
            "thinking": {"type": "disabled"},
        }
        if _is_gemini3(model):
            extra_body["thinkingConfig"] = {
                "includeThoughts": True,
                "thinkingLevel": "low",
            }
        response = self._client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body=extra_body,
        )
        return response.choices[0].message.content

    # ─────────────────────── 本地 vLLM（裸 requests + 重试） ───────────────────────
    def _call_vllm(self, messages, temperature, max_tokens, model):
        body = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": 0.95,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        headers = {"Content-Type": "application/json"}

        last_error = None
        for attempt in range(self.max_retries):
            try:
                resp = requests.post(self.base_url, json=body, headers=headers,
                                     timeout=(30, self.timeout))
                if resp.status_code != 200:
                    last_error = (f"本地API状态码: {resp.status_code}, "
                                  f"响应: {resp.text[:500]}")
                    print(last_error)
                    if attempt < self.max_retries - 1:
                        time.sleep(2 ** attempt)
                        continue
                    raise RuntimeError(last_error)
                data = resp.json()
                if data.get("choices"):
                    return data["choices"][0]["message"]["content"]
                last_error = f"本地API返回数据无choices: {json.dumps(data, ensure_ascii=False)[:500]}"
                print(last_error)
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise RuntimeError(last_error)
            except (requests.exceptions.ConnectionError,
                    requests.exceptions.Timeout) as e:
                last_error = f"本地API连接/超时 (attempt {attempt + 1}): {e}"
                print(last_error)
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise RuntimeError(last_error) from e
            except RuntimeError:
                raise
            except Exception as e:
                last_error = f"本地API未知错误: {e}"
                print(last_error)
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise RuntimeError(last_error) from e
        raise RuntimeError(last_error or "本地API调用失败（未知原因）")

    # ─────────────────────── 签名网关（裸 requests + MD5 签名头） ───────────────────────
    def _signed_headers(self):
        """生成 appId/appKey/projectId + MD5 签名头。"""
        nonce = ''.join(random.choices(string.ascii_letters + string.digits, k=10))
        timestamp = str(int(time.time()))
        str2sign = (f"appId={self.app_id}&nonce={nonce}"
                    f"&timestamp={timestamp}&appkey={self.app_key}")
        sign = hashlib.md5(str2sign.encode('utf-8')).hexdigest().upper()
        return {
            "appId": self.app_id,
            "nonce": nonce,
            "timestamp": timestamp,
            "sign": sign,
            "version": "v2",
            "Content-Type": "application/json",
            "projectid": self.project_id,
        }

    def _call_signed(self, messages, temperature, max_tokens, model):
        if not self.base_url:
            raise RuntimeError("signed 后端需通过 base_url 提供签名网关地址")
        body = {
            "max_tokens": max_tokens,
            "model": model,
            "temperature": temperature,
            "top_p": 0.95,
            "messages": messages,
            "stream": False,
        }
        if _is_gemini3(model):
            body["thinkingConfig"] = {
                "includeThoughts": True,
                "thinkingLevel": "low",
            }
        response = requests.post(self.base_url, json=body,
                                 headers=self._signed_headers(),
                                 timeout=self.timeout)
        data = response.json()
        if data.get("choices"):
            return data["choices"][0]["message"]["content"]
        error_msg = f"签名网关返回异常: {json.dumps(data, ensure_ascii=False)[:500]}"
        print(error_msg)
        raise RuntimeError(error_msg)

    # ─────────────────────── Bearer 裸 requests（评判端点） ───────────────────────
    def _call_bearer(self, messages, temperature, max_tokens, model):
        if not self.base_url:
            raise RuntimeError("bearer 后端需通过 base_url 提供端点地址")
        body = {
            "max_tokens": max_tokens,
            "model": model,
            "temperature": temperature,
            "top_p": 0.95,
            "messages": messages,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        resp = requests.post(self.base_url, json=body, headers=headers,
                             timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        if data.get("choices"):
            content = data["choices"][0]["message"]["content"]
            return str(content) if content is not None else "null"
        return "null"
