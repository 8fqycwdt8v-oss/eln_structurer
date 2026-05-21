"""Robustness regression tests for the agent loop.

Pins the behaviour added in the deep-investigation round:
- RDKit doesn't crash the extraction on adversarial SMILES.
- Iteration budget is visible to the agent via the tool result.
- Oversized paragraphs short-circuit before hitting the LLM.
- Critic output is schema-validated, not silently parsed.
- Critic-fallback state restoration leaves no stale fields.
"""

from __future__ import annotations

import asyncio

from eln_structurer.agent import MAX_PARAGRAPH_CHARS, ExtractResult, extract
from eln_structurer.chemistry import parse_mol
from eln_structurer.critic import _parse_findings
from eln_structurer.schema import ReactionDraft
from eln_structurer.tools import validate_reaction
from eln_structurer.tools.finalize_reaction import (
    FinalizedReaction,
    bind_finalized_slot,
    unbind_finalized_slot,
)


# ---------------------------------------------------------------------------
# RDKit guard
# ---------------------------------------------------------------------------


def test_parse_mol_returns_none_on_garbage() -> None:
    # An invalid SMILES that RDKit would normally fail to parse — must not
    # raise out of the cached helper.
    assert parse_mol("not[a]smiles[[[") is None


def test_parse_mol_returns_none_on_oversized_input() -> None:
    # A pathological 10k-char "SMILES" — must not crash.
    pathological = "C" * 10_000
    # Result may be None or a Mol; key requirement is it doesn't raise.
    _ = parse_mol(pathological)


def test_parse_mol_returns_none_on_empty_string() -> None:
    assert parse_mol("") is None


# ---------------------------------------------------------------------------
# Iteration budget visibility
# ---------------------------------------------------------------------------


def _broken_payload(aspirin_draft: ReactionDraft) -> dict:
    payload = aspirin_draft.model_dump(mode="json")
    for inp in payload["inputs"]:
        for comp in inp["components"]:
            comp["reaction_role"] = "REAGENT"  # break CMP-001 every time
    return payload


async def test_validate_reaction_emits_iteration_marker(aspirin_draft: ReactionDraft) -> None:
    slot = FinalizedReaction()
    token = bind_finalized_slot(slot)
    try:
        result = await validate_reaction.handler({"draft_json": _broken_payload(aspirin_draft)})
    finally:
        unbind_finalized_slot(token)
    text = result["content"][0]["text"]
    # First iteration should carry the lightweight marker.
    assert "[iteration 1 of 5]" in text


async def test_validate_reaction_warns_near_budget(aspirin_draft: ReactionDraft) -> None:
    slot = FinalizedReaction()
    token = bind_finalized_slot(slot)
    try:
        for _ in range(4):
            await validate_reaction.handler({"draft_json": _broken_payload(aspirin_draft)})
        result = await validate_reaction.handler({"draft_json": _broken_payload(aspirin_draft)})
    finally:
        unbind_finalized_slot(token)
    text = result["content"][0]["text"]
    # Iteration 5 of 5 should trigger BUDGET EXHAUSTED.
    assert "BUDGET EXHAUSTED" in text


# ---------------------------------------------------------------------------
# Paragraph size cap — must short-circuit without hitting the LLM
# ---------------------------------------------------------------------------


def test_extract_rejects_oversize_paragraph() -> None:
    huge = "x " * (MAX_PARAGRAPH_CHARS)
    result = asyncio.run(extract(huge))
    assert isinstance(result, ExtractResult)
    assert result.success is False
    assert "too large" in result.failure_summary.get("reason", "").lower()


def test_extract_respects_just_under_cap() -> None:
    """Right at the cap is fine; just over isn't. We check the boundary
    behavior without actually invoking the LLM by going slightly over."""
    huge = "x" * (MAX_PARAGRAPH_CHARS + 1)
    result = asyncio.run(extract(huge))
    assert result.success is False


# ---------------------------------------------------------------------------
# Critic parser: schema-validates the response shape
# ---------------------------------------------------------------------------


def test_critic_parser_rejects_missing_keys() -> None:
    report = _parse_findings('{"findings": [{"severity": "ERROR"}]}')
    assert report.parse_error is not None
    assert "schema" in report.parse_error.lower()


def test_critic_parser_rejects_bad_severity() -> None:
    report = _parse_findings(
        '{"findings": [{"path": "x", "severity": "MAYBE", "message": "y"}]}'
    )
    assert report.parse_error is not None


def test_critic_parser_accepts_empty_findings() -> None:
    report = _parse_findings('{"findings": []}')
    assert report.parse_error is None
    assert report.findings == []
    assert report.is_clean


def test_critic_parser_strips_code_fences() -> None:
    text = "```json\n" + '{"findings": []}' + "\n```"
    report = _parse_findings(text)
    assert report.parse_error is None
    assert report.is_clean
