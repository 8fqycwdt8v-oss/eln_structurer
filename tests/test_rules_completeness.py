"""Unit tests for the completeness rule pack."""

from __future__ import annotations

from eln_structurer.rules.completeness import (
    HasDuration,
    HasProduct,
    HasReactant,
    HasTemperature,
    NotesCaptureSource,
)
from eln_structurer.rules.base import Severity
from eln_structurer.schema import ReactionDraft
from tests.conftest import rule_ids as _ids


def test_has_reactant_passes(aspirin_draft: ReactionDraft) -> None:
    assert HasReactant().check(aspirin_draft) == []


def test_has_reactant_fires_when_no_reactant(aspirin_draft: ReactionDraft) -> None:
    for inp in aspirin_draft.inputs:
        for comp in inp.components:
            comp.reaction_role = "REAGENT"
    violations = HasReactant().check(aspirin_draft)
    assert "CMP-001" in _ids(violations)
    assert violations[0].severity is Severity.ERROR


def test_has_product_fires_when_no_outcomes() -> None:
    draft = ReactionDraft.model_validate({
        "identifiers": [],
        "inputs": [{
            "name": "reactant",
            "components": [{
                "identifiers": [{"type": "NAME", "value": "X"}],
                "amount": {"value": 1, "units": "g"},
                "reaction_role": "REACTANT",
                "is_limiting": True,
            }],
        }],
        "conditions": {"temperature": {"control_type": "AMBIENT"}},
        "workups": [],
        "outcomes": [],
        "notes": "n",
        "source_paragraph": "p",
    })
    violations = HasProduct().check(draft)
    assert "CMP-002" in _ids(violations)


def test_has_temperature_fires_when_missing(aspirin_draft: ReactionDraft) -> None:
    aspirin_draft.conditions.temperature = None
    violations = HasTemperature().check(aspirin_draft)
    assert "CMP-003" in _ids(violations)


def test_has_temperature_passes_when_ambient_set(aspirin_draft: ReactionDraft) -> None:
    from eln_structurer.schema import TemperatureModel
    aspirin_draft.conditions.temperature = TemperatureModel(control_type="AMBIENT")
    assert HasTemperature().check(aspirin_draft) == []


def test_has_duration_warns_when_missing(aspirin_draft: ReactionDraft) -> None:
    aspirin_draft.conditions.duration_minutes = None
    for outcome in aspirin_draft.outcomes:
        outcome.reaction_time_minutes = None
    violations = HasDuration().check(aspirin_draft)
    assert "CMP-004" in _ids(violations)
    assert violations[0].severity is Severity.WARNING


def test_notes_capture_source(aspirin_draft: ReactionDraft) -> None:
    aspirin_draft.notes = None
    violations = NotesCaptureSource().check(aspirin_draft)
    assert "CMP-005" in _ids(violations)
