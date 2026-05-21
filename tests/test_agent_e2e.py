"""End-to-end live test against the Anthropic API.

Skipped unless both ANTHROPIC_API_KEY and RUN_LIVE=1 are present in the env.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from eln_structurer.agent import extract


pytestmark = pytest.mark.skipif(
    not (os.environ.get("RUN_LIVE") == "1" and os.environ.get("ANTHROPIC_API_KEY")),
    reason="Live E2E disabled. Set RUN_LIVE=1 and ANTHROPIC_API_KEY to run.",
)


EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


@pytest.mark.asyncio
async def test_aspirin_extraction() -> None:
    paragraph = (EXAMPLES / "aspirin.txt").read_text()
    result = await extract(paragraph, max_iters=5)
    assert result.success, "Agent did not produce a finalized reaction."
    assert "salicylic" in result.pbtxt.lower()
    assert "acetylsalicylic" in result.json_text.lower() or "aspirin" in result.json_text.lower()
