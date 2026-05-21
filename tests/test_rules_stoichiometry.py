"""Unit tests for the stoichiometry rule pack."""

from __future__ import annotations

from eln_structurer.rules.stoichiometry import (
    LimitingReagentIdentifiable,
    PlausibleVolumes,
)
from eln_structurer.schema import AmountModel, ReactionDraft


def _ids(violations) -> set[str]:
    return {v.rule_id for v in violations}


def test_limiting_present_passes(aspirin_draft: ReactionDraft) -> None:
    assert LimitingReagentIdentifiable().check(aspirin_draft) == []


def test_limiting_too_many_fires(aspirin_draft: ReactionDraft) -> None:
    # Flag a second compound as limiting.
    aspirin_draft.inputs[1].components[0].is_limiting = True
    violations = LimitingReagentIdentifiable().check(aspirin_draft)
    assert "STO-004" in _ids(violations)


def test_plausible_volumes_warns_on_huge(aspirin_draft: ReactionDraft) -> None:
    for inp in aspirin_draft.inputs:
        for comp in inp.components:
            if comp.amount and comp.amount.units == "mL":
                comp.amount = AmountModel(value=50000.0, units="mL")  # 50 L
                break
    violations = PlausibleVolumes().check(aspirin_draft)
    assert "STO-003" in _ids(violations)


def test_plausible_volumes_errors_on_zero(aspirin_draft: ReactionDraft) -> None:
    for inp in aspirin_draft.inputs:
        for comp in inp.components:
            if comp.amount and comp.amount.units == "mL":
                comp.amount = AmountModel(value=0.0, units="mL")
                break
    violations = PlausibleVolumes().check(aspirin_draft)
    assert "STO-003" in _ids(violations)
