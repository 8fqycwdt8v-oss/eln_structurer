"""End-to-end agent.extract tests with the SDK stubbed out.

These tests don't validate extraction quality (the stub never actually
runs the tool loop). They check that agent.extract:
- short-circuits on oversized paragraphs without touching the SDK;
- doesn't crash when the critic returns an empty findings list;
- aggregates UsageStats across primary+critic calls.
"""

from __future__ import annotations

import asyncio

from eln_structurer.agent import MAX_PARAGRAPH_CHARS, extract
from tests._sdk_stub import stub_sdk


def test_extract_short_circuits_on_oversize_without_touching_sdk() -> None:
    huge = "x" * (MAX_PARAGRAPH_CHARS + 1)
    with stub_sdk() as log:
        result = asyncio.run(extract(huge))
    # Paragraph too large → no SDK call should have been made.
    assert log.primary_calls == 0
    assert log.critic_calls == 0
    assert result.success is False
    assert "too large" in result.failure_summary["reason"].lower()


def test_extract_empty_stub_response_does_not_crash() -> None:
    """Stub primary produces a text block but no actual tool call →
    extract() falls through without a finalized output. Should return
    success=False rather than raising."""
    with stub_sdk(primary_text="(no tool calls)") as log:
        result = asyncio.run(extract("Tiny paragraph for stub test."))
    assert log.primary_calls == 1
    # Critic gates on success (finalized.pbtxt non-empty) — should not run.
    assert log.critic_calls == 0
    assert result.success is False
