"""Pydantic -> ORD proto bridge tests."""

from __future__ import annotations

import pytest

from eln_structurer.proto_bridge import draft_to_proto, serialize_reaction
from eln_structurer.schema import ReactionDraft


def test_aspirin_bridges_to_proto(aspirin_draft: ReactionDraft) -> None:
    reaction_pb = draft_to_proto(aspirin_draft)
    assert len(reaction_pb.inputs) == 3
    # The first input was named "limiting_reactant".
    assert "limiting_reactant" in reaction_pb.inputs
    # That input has exactly one component.
    components = reaction_pb.inputs["limiting_reactant"].components
    assert len(components) == 1
    # Salicylic acid was tagged is_limiting=True.
    assert components[0].is_limiting is True


def test_serialize_json(aspirin_draft: ReactionDraft) -> None:
    reaction_pb = draft_to_proto(aspirin_draft)
    text = serialize_reaction(reaction_pb, fmt="json")
    assert text.startswith("{")
    assert "salicylic" in text.lower() or "OC(=O)c1ccccc1O" in text


def test_serialize_pbtxt(aspirin_draft: ReactionDraft) -> None:
    reaction_pb = draft_to_proto(aspirin_draft)
    text = serialize_reaction(reaction_pb, fmt="pbtxt")
    assert "inputs" in text
    assert "limiting_reactant" in text


def test_serialize_rejects_unknown_format(aspirin_draft: ReactionDraft) -> None:
    reaction_pb = draft_to_proto(aspirin_draft)
    with pytest.raises(ValueError):
        serialize_reaction(reaction_pb, fmt="csv")


def test_duplicate_input_names_get_suffixed() -> None:
    from eln_structurer.schema import (
        AmountModel,
        CompoundIdentifierModel,
        CompoundModel,
        ConditionsModel,
        OutcomeModel,
        ProductModel,
        ReactionInputModel,
        TemperatureModel,
    )
    draft = ReactionDraft(
        identifiers=[],
        inputs=[
            ReactionInputModel(
                name="reagent",
                components=[
                    CompoundModel(
                        identifiers=[CompoundIdentifierModel(type="NAME", value="A")],
                        amount=AmountModel(value=1, units="mmol"),
                        reaction_role="REACTANT",
                        is_limiting=True,
                    )
                ],
            ),
            ReactionInputModel(
                name="reagent",  # duplicate on purpose
                components=[
                    CompoundModel(
                        identifiers=[CompoundIdentifierModel(type="NAME", value="B")],
                        amount=AmountModel(value=1, units="mmol"),
                        reaction_role="REAGENT",
                    )
                ],
            ),
        ],
        conditions=ConditionsModel(temperature=TemperatureModel(control_type="AMBIENT")),
        outcomes=[
            OutcomeModel(
                products=[
                    ProductModel(
                        compound=CompoundModel(
                            identifiers=[CompoundIdentifierModel(type="NAME", value="P")],
                            reaction_role="PRODUCT",
                        )
                    )
                ]
            )
        ],
        notes="n",
        source_paragraph="p",
    )
    reaction_pb = draft_to_proto(draft)
    assert "reagent" in reaction_pb.inputs
    assert "reagent_2" in reaction_pb.inputs
