"""Critic execution + response parsing.

Public entry: ``run_critic``. Pure-Python ``_parse_findings`` exposed
for tests so they don't need a live LLM to verify the parser logic.
"""

from __future__ import annotations

import json
import os
import re

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    TextBlock,
    query,
)
from pydantic import ValidationError

from eln_structurer.critic.models import (
    CriticFinding,
    CriticReport,
    _CriticResponseModel,
)
from eln_structurer.critic.prompt import (
    build_critic_system_prompt,
    build_critic_user_prompt,
)


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```\s*$", re.MULTILINE)


def _parse_findings(text: str) -> CriticReport:
    cleaned = _FENCE_RE.sub("", text).strip()
    if not cleaned:
        return CriticReport(raw_text=text, parse_error="critic returned empty output")
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        return CriticReport(raw_text=text, parse_error=f"non-JSON critic output: {exc}")
    try:
        parsed = _CriticResponseModel.model_validate(payload)
    except ValidationError as exc:
        return CriticReport(raw_text=text, parse_error=f"critic schema error: {exc}")
    findings = [
        CriticFinding(path=f.path, severity=f.severity, message=f.message)
        for f in parsed.findings
    ]
    return CriticReport(findings=findings, raw_text=cleaned)


def critic_available() -> bool:
    """The critic needs an Anthropic key. Skip it gracefully without one."""
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


async def run_critic(
    *,
    paragraph: str,
    draft_json: str,
    model: str,
) -> CriticReport:
    """Ask a fresh LLM whether the draft faithfully represents the paragraph."""
    user_prompt = build_critic_user_prompt(paragraph, draft_json)
    options = ClaudeAgentOptions(
        model=model,
        system_prompt=build_critic_system_prompt(),
        allowed_tools=[],
        max_turns=1,
    )
    text_parts: list[str] = []
    async for message in query(prompt=user_prompt, options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    text_parts.append(block.text)
    return _parse_findings("".join(text_parts))
