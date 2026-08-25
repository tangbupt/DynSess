#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prompt variants for rebuttal-time user simulator comparisons."""

from __future__ import annotations

import re
from typing import Dict


STYLE_DESCRIPTIONS: Dict[str, str] = {
    "passive": "Paper-style lazy/passive user simulator, kept as the baseline.",
    "balanced": "A realistic user who mostly reacts but may add information or ask follow-up questions.",
    "proactive": "An active user who drives goals, probes memory/persona, and introduces topic pressure.",
}


def available_styles() -> tuple[str, ...]:
    return tuple(STYLE_DESCRIPTIONS.keys())


def strip_role_persona_prefix(persona_info: str) -> str:
    prefix = "请你扮演以下人设："
    persona_info = persona_info or ""
    if persona_info.startswith(prefix):
        return persona_info[len(prefix):]
    return persona_info


def sanitize_user_response(text: str) -> str:
    """Remove common role-play wrappers from user simulator output."""
    if not text or text == "null":
        return text
    text = re.sub(r"【.*?】", "", text).strip()
    text = re.sub(r"（.*?）", "", text).strip()
    text = re.sub(r"\(.*?\)", "", text).strip()
    text = text.strip().strip('"').strip("'")
    return text


def build_user_simulator_prompt(style: str,
                                persona_info: str,
                                user_system_prompt: str) -> str:
    if style not in STYLE_DESCRIPTIONS:
        raise ValueError(
            f"Unknown simulator style: {style}. "
            f"Choose one of: {', '.join(available_styles())}"
        )

    processed = strip_role_persona_prefix(persona_info)

    shared_header = f"""你是一个用户模拟器。你的唯一任务是生成下一条“用户消息”，用于和一个 AI 角色扮演模型对话。

身份边界：
- 你始终是用户，不是对方正在扮演的角色。
- 不要替对方说话，不要续写对方剧情，不要输出旁白、动作描写或心理描写。
- 只输出用户会发送的一条消息本身，不要加解释、标题、引号或括号说明。

你的用户画像：
[
{user_system_prompt}
]

对方 AI 正在扮演的人设。这里只供你理解对方，不是你的身份：
[
{processed}
]
"""

    if style == "passive":
        style_block = """行为设定：passive baseline
- 像一个懒惰、低主动性的真实用户，只被动回答对方的问题，或对对方的话做简短评价。
- 不主动开启新话题，不主动推进剧情，不频繁提问；除非非常必要，不要反问对方。
- 口语化、自然、可以略显随意或敷衍。
- 回复严格控制在 10-20 个中文字符左右，通常一句话。

当前任务：
根据你的用户画像和对话上下文，给出一句简短、自然、被动的用户回复。"""

    elif style == "balanced":
        style_block = """行为设定：balanced realistic user
- 像一个普通真实用户：多数时候回应上一轮内容，但也会自然补充自己的想法、感受、偏好或限制。
- 可以偶尔追问、澄清、表达不同意见，或提出一个轻量的新方向，但不要每轮都强行推进。
- 你要给角色模型留下可接的话题，避免只说“嗯”“好”“然后呢”这类空回复。
- 回复长度控制在 15-40 个中文字符左右，1-2 句话。
- 保持用户画像，不要变成导演、旁白或另一个角色。

当前任务：
根据你的用户画像和上下文，生成一条自然、有信息量但不过度主导的用户回复。"""

    else:  # proactive
        style_block = """行为设定：proactive / active user
- 你是主动用户，会推动互动，而不是等待 AI 角色独自表演。
- 每轮从以下策略中自然选择 1 个：提出明确目标、做出决定、引入新约束、追问细节、挑战矛盾、回忆前文线索、表达情绪变化、要求对方主动安排下一步。
- 允许更频繁地提问，也可以主动转换话题或制造轻微压力，用来测试角色的长期记忆、人设稳定性和交互引导能力。
- 不要替对方完成剧情，不要告诉对方应该如何表演；你只说用户本人会说的话。
- 回复长度控制在 20-60 个中文字符左右，通常 1-2 句话。

当前任务：
根据你的用户画像和上下文，生成一条主动、有推进力、能检验角色互动能力的用户回复。"""

    return f"{shared_header}\n{style_block}"
