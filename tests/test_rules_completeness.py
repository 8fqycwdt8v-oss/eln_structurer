"""Unit tests for the completeness rule pack."""

from __future__ import annotations

from eln_structurer.rules.completeness import (
    DurationRangeSanity,
    HasDuration,
    HasProduct,
    HasReactant,
    HasTemperature,
    NotesCaptureSource,
    TemperatureRangeSanity,
)
from eln_structurer.rules.base import Severity
from eln_structurer.schema import ReactionDraft, TemperatureModel
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


def test_temperature_range_passes_for_reasonable(aspirin_draft: ReactionDraft) -> None:
    assert TemperatureRangeSanity().check(aspirin_draft) == []


def test_temperature_range_errors_above_300c(aspirin_draft: ReactionDraft) -> None:
    aspirin_draft.conditions.temperature = TemperatureModel(
        setpoint_celsius=500.0, control_type="HEATER"
    )
    violations = TemperatureRangeSanity().check(aspirin_draft)
    assert "CMP-006" in _ids(violations)
    assert violations[0].severity is Severity.ERROR


def test_temperature_range_errors_below_minus_100c(aspirin_draft: ReactionDraft) -> None:
    aspirin_draft.conditions.temperature = TemperatureModel(
        setpoint_celsius=-200.0, control_type="LIQUID_NITROGEN"
    )
    violations = TemperatureRangeSanity().check(aspirin_draft)
    assert "CMP-006" in _ids(violations)


def test_duration_range_errors_above_two_weeks(aspirin_draft: ReactionDraft) -> None:
    aspirin_draft.conditions.duration_minutes = 60 * 24 * 30  # 30 days
    violations = DurationRangeSanity().check(aspirin_draft)
    assert "CMP-007" in _ids(violations)


def test_duration_range_errors_negative(aspirin_draft: ReactionDraft) -> None:
    aspirin_draft.conditions.duration_minutes = -5.0
    violations = DurationRangeSanity().check(aspirin_draft)
    assert "CMP-007" in _ids(violations)
