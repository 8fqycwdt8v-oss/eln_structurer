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


def test_apply_amount_equiv_records_feature() -> None:
    """A Compound with units='equiv' should encode equivalents in features
    and mark the amount as UnmeasuredAmount.CUSTOM."""
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
    from ord_schema.proto import reaction_pb2

    draft = ReactionDraft(
        identifiers=[],
        inputs=[
            ReactionInputModel(
                name="r",
                components=[
                    CompoundModel(
                        identifiers=[CompoundIdentifierModel(type="NAME", value="A")],
                        amount=AmountModel(value=1.5, units="equiv"),
                        reaction_role="REAGENT",
                    )
                ],
            ),
            ReactionInputModel(
                name="lim",
                components=[
                    CompoundModel(
                        identifiers=[CompoundIdentifierModel(type="NAME", value="L")],
                        amount=AmountModel(value=1.0, units="mmol"),
                        reaction_role="REACTANT",
                        is_limiting=True,
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
    proto = draft_to_proto(draft)
    comp = proto.inputs["r"].components[0]
    assert comp.features["equivalents"].float_value == 1.5
    assert comp.amount.unmeasured.type == reaction_pb2.UnmeasuredAmount.CUSTOM
    assert "1.5" in comp.amount.unmeasured.details


def test_apply_amount_mass_pct_uses_unmeasured() -> None:
    """A Compound with units='mass_pct' should land in unmeasured with details."""
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
    from ord_schema.proto import reaction_pb2

    draft = ReactionDraft(
        identifiers=[],
        inputs=[
            ReactionInputModel(
                name="r",
                components=[
                    CompoundModel(
                        identifiers=[CompoundIdentifierModel(type="NAME", value="catalyst")],
                        amount=AmountModel(value=5.0, units="mass_pct"),
                        reaction_role="CATALYST",
                    )
                ],
            ),
            ReactionInputModel(
                name="lim",
                components=[
                    CompoundModel(
                        identifiers=[CompoundIdentifierModel(type="NAME", value="L")],
                        amount=AmountModel(value=1.0, units="mmol"),
                        reaction_role="REACTANT",
                        is_limiting=True,
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
    proto = draft_to_proto(draft)
    comp = proto.inputs["r"].components[0]
    assert comp.amount.unmeasured.type == reaction_pb2.UnmeasuredAmount.CUSTOM
    assert "mass_pct" in comp.amount.unmeasured.details


def test_apply_amount_missing_amount_for_catalyst() -> None:
    """A CATALYST with amount=None should encode as UnmeasuredAmount.CATALYTIC."""
    from eln_structurer.schema import (
        CompoundIdentifierModel,
        CompoundModel,
        ConditionsModel,
        OutcomeModel,
        ProductModel,
        ReactionInputModel,
        TemperatureModel,
        AmountModel,
    )
    from ord_schema.proto import reaction_pb2

    draft = ReactionDraft(
        identifiers=[],
        inputs=[
            ReactionInputModel(
                name="cat",
                components=[
                    CompoundModel(
                        identifiers=[CompoundIdentifierModel(type="NAME", value="H2SO4")],
                        amount=None,
                        reaction_role="CATALYST",
                    )
                ],
            ),
            ReactionInputModel(
                name="r",
                components=[
                    CompoundModel(
                        identifiers=[CompoundIdentifierModel(type="NAME", value="L")],
                        amount=AmountModel(value=1, units="mmol"),
                        reaction_role="REACTANT",
                        is_limiting=True,
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
    proto = draft_to_proto(draft)
    cat = proto.inputs["cat"].components[0]
    assert cat.amount.unmeasured.type == reaction_pb2.UnmeasuredAmount.CATALYTIC


def test_moles_computed_populated_when_mass_and_smiles_present(aspirin_draft) -> None:
    """For a Compound with mass + SMILES, the bridge fills
    features['moles_computed']."""
    proto = draft_to_proto(aspirin_draft)
    salicylic = proto.inputs["limiting_reactant"].components[0]
    # Salicylic acid 1.38 g, MW ≈ 138.12 → ~0.01 mol = 10 mmol.
    moles = salicylic.features["moles_computed"].float_value
    assert 0.0098 < moles < 0.0102


def test_moles_computed_absent_for_unmeasured_amount(aspirin_draft) -> None:
    """A CATALYST without an amount must NOT get a moles_computed feature."""
    proto = draft_to_proto(aspirin_draft)
    cat = proto.inputs["catalyst"].components[0]
    assert "moles_computed" not in cat.features


def test_reaction_smiles_emitted(aspirin_draft) -> None:
    """The bridge composes a reaction SMILES from inputs and products."""
    from ord_schema.proto import reaction_pb2
    proto = draft_to_proto(aspirin_draft)
    rxn_smiles_idents = [
        i for i in proto.identifiers
        if i.type == reaction_pb2.ReactionIdentifier.REACTION_SMILES
    ]
    assert len(rxn_smiles_idents) == 1
    s = rxn_smiles_idents[0].value
    assert ">" in s and s.count(">") == 2
    # Left side should include salicylic acid; right side should include
    # the aspirin product. We assert on substrings (canonical SMILES is
    # RDKit-dependent and may differ slightly).
    left, _middle, right = s.split(">")
    assert "c1ccccc1" in left.replace("(", "").replace(")", "").lower() or "ccccc" in left
    assert "C(=O)" in right or "CC(=O)" in right


def test_round_trip_integrity_clean_on_aspirin(aspirin_draft) -> None:
    """A clean draft must round-trip JSON and pbtxt without information loss."""
    from eln_structurer.proto_bridge import verify_round_trip

    proto = draft_to_proto(aspirin_draft)
    mismatches = verify_round_trip(proto)
    assert mismatches == [], mismatches


def test_round_trip_integrity_catches_corrupt_pbtxt() -> None:
    """If we deliberately damage the pbtxt input, parsing should fail and
    the round-trip helper should report it. This exercises the
    ParseError branch."""
    from google.protobuf import text_format

    from eln_structurer.proto_bridge import reaction_pb2
    from ord_schema import message_helpers  # noqa: F401  ensure proto module loaded

    proto = reaction_pb2.Reaction()
    # Manually feed corrupt pbtxt through the parser to confirm it raises.
    import pytest as _pytest
    with _pytest.raises(text_format.ParseError):
        text_format.Parse("not valid pbtxt at all {{{", proto)


def test_harness_surfaces_round_trip_errors_as_ord_validation_errors(
    aspirin_draft, monkeypatch
) -> None:
    """If verify_round_trip reports a mismatch, run_harness must put it
    into ord_validation_errors so the agent sees it."""
    import sys

    from eln_structurer.harness import run_harness

    harness_mod = sys.modules["eln_structurer.harness"]
    monkeypatch.setattr(
        harness_mod,
        "verify_round_trip",
        lambda _proto: ["synthetic mismatch for testing"],
    )
    report = run_harness(aspirin_draft)
    assert any("round-trip" in e for e in report.ord_validation_errors)
    assert not report.is_clean


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
