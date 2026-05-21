"""Harness orchestration tests (no LLM calls)."""

from __future__ import annotations

from eln_structurer.harness import run_harness
from eln_structurer.schema import ReactionDraft


def test_clean_aspirin_draft_has_no_errors(aspirin_draft: ReactionDraft) -> None:
    report = run_harness(aspirin_draft)
    # We allow warnings (e.g. CMP-005 / atmosphere notes), but ZERO rule errors
    # and ZERO bridge errors. ord-schema validation may emit its own warnings
    # we don't try to satisfy here.
    rule_error_ids = {v.rule_id for v in report.errors}
    assert rule_error_ids == set(), report.as_repair_prompt()
    assert report.bridge_error is None, report.bridge_error


def test_broken_draft_produces_repair_prompt(aspirin_draft: ReactionDraft) -> None:
    # Strip all reactants to deliberately break CMP-001.
    for inp in aspirin_draft.inputs:
        for comp in inp.components:
            comp.reaction_role = "REAGENT"
    report = run_harness(aspirin_draft)
    assert not report.is_clean
    prompt = report.as_repair_prompt()
    assert "CMP-001" in prompt
    assert "VALIDATION FAILED" in prompt
    assert "Fix:" in prompt
