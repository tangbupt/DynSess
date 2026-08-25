#!/usr/bin/env python3
"""Shared helpers for the DynSess long-context rebuttal evaluation."""

from __future__ import annotations

import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Iterable, Optional

import requests


def load_json_or_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        first = handle.read(1)
    if first == "[":
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, list):
            raise ValueError(f"Expected a JSON list in {path}")
        return data

    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
    return records


def dump_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
    temporary.replace(path)


def dump_jsonl(records: Iterable[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


def slugify(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return value.strip("._-") or "model"


def persona_title(persona: str) -> str:
    text = (persona or "").removeprefix("请你扮演以下人设：").strip()
    for delimiter in ("（", "(", "\n", "【"):
        if delimiter in text:
            text = text.split(delimiter, 1)[0]
    return text[:80] or "unknown"


def extract_json_object(text: str) -> Optional[dict]:
    if not isinstance(text, str):
        return None
    candidates = [text.strip()]
    candidates.extend(
        match.group(1).strip()
        for match in re.finditer(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    )
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start:end + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except (TypeError, json.JSONDecodeError):
            continue
    return None


class ChatCompletionsClient:
    """Small OpenAI-compatible HTTP client matching the existing repo calls."""

    def __init__(
        self,
        api_url: str,
        token: str,
        timeout: int = 180,
        max_retries: int = 3,
        retry_delay: float = 3.0,
        mock_api: bool = False,
    ) -> None:
        self.api_url = api_url
        self.token = token
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.mock_api = mock_api

    def complete(
        self,
        model: str,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
        extra_body: Optional[dict] = None,
    ) -> str:
        if self.mock_api:
            return self._mock_complete(model, messages)

        body = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if extra_body:
            body.update(extra_body)
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.post(
                    self.api_url,
                    json=body,
                    headers=headers,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                data = response.json()
                choices = data.get("choices") or []
                if choices:
                    content = choices[0].get("message", {}).get("content")
                    if content:
                        return str(content).strip()
                raise ValueError(f"Malformed chat-completions response: {str(data)[:500]}")
            except Exception as exc:  # API errors are retried and surfaced with context.
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * attempt)
        raise RuntimeError(
            f"API call failed after {self.max_retries} attempts for model={model}: {last_error}"
        ) from last_error

    @staticmethod
    def _mock_complete(model: str, messages: list[dict]) -> str:
        system = messages[0].get("content", "") if messages else ""
        prompt = messages[-1].get("content", "") if messages else ""
        dimensions = (
            "human_likeness",
            "role_consistency",
            "context_consistency",
            "interactive_ability",
        )
        judge_key = next((key for key in dimensions if key in prompt), None)
        if judge_key and ("评分" in prompt or "评测" in prompt or "评估" in prompt):
            key = judge_key
            return json.dumps({key: {"reason": "mock result", "score": 4}})

        last = messages[-1].get("content", "") if messages else ""
        if "用户模拟器" in system:
            return "那你还记得我们之前说过的细节吗？"
        return f"我记得。我们接着聊你刚才提到的事情吧。({model}:{len(last)})"


def resolve_token(explicit: Optional[str], env_name: str, fallback_env: str = "DYNS_API_TOKEN") -> str:
    return explicit or os.environ.get(env_name, "") or os.environ.get(fallback_env, "")


def seeded_greeting(seed: int, persona_id: int) -> str:
    greetings = ["你好", "你好呀", "在吗", "在干嘛", "嗨", "嘿", "你在吗"]
    return random.Random(seed + persona_id * 1009).choice(greetings)
