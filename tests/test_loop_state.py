"""Tests for repair-loop divergence detection and finalize trust.

Both behaviors hinge on the per-task slot kept by ``finalize_reaction``;
this module exercises them end-to-end by binding a slot and invoking the
tool handlers directly.
"""

from __future__ import annotations

from eln_structurer.schema import ReactionDraft
from eln_structurer.tools import finalize_reaction, validate_reaction
from eln_structurer.tools.finalize_reaction import (
    FinalizedReaction,
    bind_finalized_slot,
    draft_signature,
    unbind_finalized_slot,
)


def _broken_payload(aspirin_draft: ReactionDraft) -> dict:
    """Aspirin fixture with all reactants demoted to REAGENT — triggers
    CMP-001 every time."""
    payload = aspirin_draft.model_dump(mode="json")
    for inp in payload["inputs"]:
        for comp in inp["components"]:
            comp["reaction_role"] = "REAGENT"
    return payload


async def test_validate_reaction_counts_iterations(aspirin_draft: ReactionDraft) -> None:
    slot = FinalizedReaction()
    token = bind_finalized_slot(slot)
    try:
        bad = _broken_payload(aspirin_draft)
        for _ in range(2):
            await validate_reaction.handler({"draft_json": bad})
    finally:
        unbind_finalized_slot(token)
    assert slot.iterations == 2
    assert slot.rule_history["CMP-001"] == 2


async def test_validate_reaction_escalates_after_threshold(
    aspirin_draft: ReactionDraft,
) -> None:
    """Three consecutive failures of the same rule must surface the
    !!! ESCALATION suffix."""
    slot = FinalizedReaction()
    token = bind_finalized_slot(slot)
    try:
        bad = _broken_payload(aspirin_draft)
        results = [
            await validate_reaction.handler({"draft_json": bad}) for _ in range(3)
        ]
    finally:
        unbind_finalized_slot(token)
    last_text = results[-1]["content"][0]["text"]
    assert "ESCALATION" in last_text
    assert "CMP-001" in last_text


async def test_validate_reaction_records_clean_signature(
    aspirin_draft: ReactionDraft,
) -> None:
    """When the rule pack is clean, the slot records the draft's signature
    so finalize_reaction can skip the redundant pass."""
    slot = FinalizedReaction()
    token = bind_finalized_slot(slot)
    try:
        payload = aspirin_draft.model_dump(mode="json")
        result = await validate_reaction.handler({"draft_json": payload})
    finally:
        unbind_finalized_slot(token)
    assert result.get("isError") is not True, result
    assert slot.last_clean_signature == draft_signature(aspirin_draft)


async def test_finalize_trusts_recent_clean_signature(
    aspirin_draft: ReactionDraft, monkeypatch
) -> None:
    """If finalize_reaction sees the same draft just validated clean, it
    must skip run_harness. We verify by patching run_harness to raise."""
    slot = FinalizedReaction()
    token = bind_finalized_slot(slot)
    try:
        payload = aspirin_draft.model_dump(mode="json")
        clean = await validate_reaction.handler({"draft_json": payload})
        assert clean.get("isError") is not True

        # Now patch run_harness in finalize_reaction's module — if the
        # trust path is broken and finalize tries to re-validate, the test
        # will fail loudly. tools/__init__.py re-exports the SdkMcpTool
        # under the same dotted path, so we have to fish the module out
        # of sys.modules directly.
        import sys
        fr_mod = sys.modules["eln_structurer.tools.finalize_reaction"]
        def _boom(_draft):  # pragma: no cover — must not be called
            raise AssertionError("finalize_reaction unexpectedly ran the harness")
        monkeypatch.setattr(fr_mod, "run_harness", _boom)

        result = await finalize_reaction.handler({"draft_json": payload})
    finally:
        unbind_finalized_slot(token)
    assert result.get("isError") is not True, result
    assert slot.pbtxt
    assert slot.json_text
