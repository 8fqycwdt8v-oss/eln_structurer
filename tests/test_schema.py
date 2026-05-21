"""Pydantic schema round-trip tests."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from eln_structurer.schema import (
    AmountModel,
    CompoundIdentifierModel,
    CompoundModel,
    ConditionsModel,
    ReactionDraft,
    ReactionInputModel,
    TemperatureModel,
    reaction_draft_json_schema,
)


def test_amount_unit_enum_rejects_unknown() -> None:
    with pytest.raises(ValidationError):
        AmountModel.model_validate({"value": 1.0, "units": "furlong"})


def test_amount_unit_enum_accepts_known() -> None:
    amt = AmountModel.model_validate({"value": 1.0, "units": "mmol"})
    assert amt.value == 1.0
    assert amt.units == "mmol"


def test_identifier_type_enum() -> None:
    with pytest.raises(ValidationError):
        CompoundIdentifierModel.model_validate({"type": "BARCODE", "value": "x"})


def test_reaction_draft_requires_inputs() -> None:
    with pytest.raises(ValidationError):
        ReactionDraft.model_validate(
            {
                "identifiers": [],
                "inputs": [],
                "conditions": {},
                "workups": [],
                "outcomes": [],
                "notes": None,
                "source_paragraph": "x",
            }
        )


def test_reaction_draft_roundtrip(aspirin_draft: ReactionDraft) -> None:
    payload = aspirin_draft.model_dump(mode="json")
    rebuilt = ReactionDraft.model_validate(payload)
    assert rebuilt == aspirin_draft


def test_reaction_draft_json_serialization(aspirin_draft: ReactionDraft) -> None:
    text = aspirin_draft.model_dump_json()
    rebuilt = ReactionDraft.model_validate(json.loads(text))
    assert rebuilt.inputs[0].components[0].is_limiting is True


def test_json_schema_emits_titles() -> None:
    schema = reaction_draft_json_schema()
    # Confirm Pydantic generated a $defs section with our enums + models.
    assert "$defs" in schema
    assert "ReactionDraft" in json.dumps(schema)
