"""Unit tests for the stoichiometry rule pack."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from eln_structurer.rules.stoichiometry import (
    LimitingReagentIdentifiable,
    LimitingReagentIsActuallyLimiting,
    MassBalanceSanity,
    PlausibleVolumes,
    YieldMassConsistency,
    YieldRangeSanity,
)
from eln_structurer.schema import (
    AmountModel,
    CompoundIdentifierModel,
    CompoundModel,
    ProductMeasurementModel,
    ReactionDraft,
    ReactionInputModel,
)
from tests.conftest import rule_ids as _ids


def test_limiting_present_passes(aspirin_draft: ReactionDraft) -> None:
    assert LimitingReagentIdentifiable().check(aspirin_draft) == []


def test_limiting_too_many_fires(aspirin_draft: ReactionDraft) -> None:
    aspirin_draft.inputs[1].components[0].is_limiting = True
    violations = LimitingReagentIdentifiable().check(aspirin_draft)
    assert "STO-004" in _ids(violations)


def test_limiting_tied_moles_warns(aspirin_draft: ReactionDraft) -> None:
    """Two REACTANTS with identical mole counts should trigger a WARNING."""
    # Replace the catalyst with a second REACTANT carrying the same moles
    # as salicylic acid (10.0 mmol).
    aspirin_draft.inputs[2].components[0] = CompoundModel(
        identifiers=[
            CompoundIdentifierModel(type="NAME", value="acetic anhydride"),
            CompoundIdentifierModel(type="SMILES", value="CC(=O)OC(C)=O"),
        ],
        amount=AmountModel(value=10.0, units="mmol"),
        reaction_role="REACTANT",
    )
    # Clear the explicit is_limiting flag from salicylic acid so the rule
    # has to infer.
    aspirin_draft.inputs[0].components[0].is_limiting = False
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


def test_amount_model_rejects_non_positive() -> None:
    """Pydantic-level guard: AmountModel(value <= 0) is invalid by construction."""
    with pytest.raises(ValidationError):
        AmountModel(value=0.0, units="mL")
    with pytest.raises(ValidationError):
        AmountModel(value=-1.5, units="g")


def test_yield_range_sanity_passes_for_realistic(aspirin_draft: ReactionDraft) -> None:
    # The fixture has yield=90.0 — well within range.
    assert YieldRangeSanity().check(aspirin_draft) == []


def _draft_with_yield(yield_pct: float) -> ReactionDraft:
    return ReactionDraft(
        identifiers=[],
        inputs=[
            ReactionInputModel(
                name="r",
                components=[
                    CompoundModel(
                        identifiers=[CompoundIdentifierModel(type="NAME", value="x")],
                        amount=AmountModel(value=1, units="mmol"),
                        reaction_role="REACTANT",
                        is_limiting=True,
                    )
                ],
            )
        ],
        conditions={"temperature": {"control_type": "AMBIENT"}},
        outcomes=[
            {
                "products": [
                    {
                        "compound": {
                            "identifiers": [{"type": "NAME", "value": "p"}],
                            "reaction_role": "PRODUCT",
                        },
                        "measurements": [
                            ProductMeasurementModel(type="YIELD", value=yield_pct, units="%")
                        ],
                    }
                ]
            }
        ],
        notes="n",
        source_paragraph="p",
    )


def test_yield_range_errors_on_impossible() -> None:
    violations = YieldRangeSanity().check(_draft_with_yield(150.0))
    assert "STO-005" in _ids(violations)


def test_yield_range_warns_on_borderline() -> None:
    violations = YieldRangeSanity().check(_draft_with_yield(103.5))
    assert "STO-005" in _ids(violations)


def test_mass_balance_passes_for_realistic_aspirin(aspirin_draft: ReactionDraft) -> None:
    # n_lim = 10 mmol; MW_aspirin = 180.16 g/mol -> max ≈ 1.98 g.
    # Fixture reports 1.62 g — comfortably below the ceiling.
    assert MassBalanceSanity().check(aspirin_draft) == []


def test_mass_balance_errors_on_impossible_yield(aspirin_draft: ReactionDraft) -> None:
    # Bump reported product AMOUNT to 5 g; theoretical max is ~1.98 g.
    aspirin_draft.outcomes[0].products[0].measurements[1] = ProductMeasurementModel(
        type="AMOUNT", value=5.0, units="g"
    )
    violations = MassBalanceSanity().check(aspirin_draft)
    assert "STO-006" in _ids(violations)


def test_yield_mass_consistency_passes_for_aspirin(aspirin_draft: ReactionDraft) -> None:
    """Aspirin fixture has yield=90%, mass=1.62 g, n_lim=10 mmol, MW≈180.16.
    Expected: 0.9 × 0.01 × 180.16 = 1.62 g. Exact match."""
    assert YieldMassConsistency().check(aspirin_draft) == []


def test_yield_mass_consistency_fires_on_mismatch(aspirin_draft: ReactionDraft) -> None:
    """Keep yield at 90% but change the reported mass to 0.5 g — incompatible."""
    aspirin_draft.outcomes[0].products[0].measurements[1] = ProductMeasurementModel(
        type="AMOUNT", value=0.5, units="g"
    )
    violations = YieldMassConsistency().check(aspirin_draft)
    assert "STO-007" in _ids(violations)


def test_yield_mass_consistency_silent_without_both(aspirin_draft: ReactionDraft) -> None:
    """Rule needs both a YIELD and an AMOUNT to fire."""
    # Drop the AMOUNT measurement; keep only YIELD.
    aspirin_draft.outcomes[0].products[0].measurements = [
        ProductMeasurementModel(type="YIELD", value=90.0, units="%"),
    ]
    assert YieldMassConsistency().check(aspirin_draft) == []


def test_limiting_actually_limiting_passes(aspirin_draft: ReactionDraft) -> None:
    """Aspirin fixture flags salicylic acid (the only quantified REACTANT)
    as limiting — trivially correct."""
    assert LimitingReagentIsActuallyLimiting().check(aspirin_draft) == []


def test_limiting_actually_limiting_fires_when_wrong(aspirin_draft: ReactionDraft) -> None:
    """Add a second REACTANT with smaller moles than the currently-flagged
    limiting reagent. The rule must catch the misidentification."""
    from eln_structurer.schema import (
        AmountModel,
        CompoundIdentifierModel,
        CompoundModel,
    )
    # Salicylic acid currently is_limiting with 10 mmol. Add a phantom
    # REACTANT with 1 mmol that would actually be limiting.
    aspirin_draft.inputs[2].components[0] = CompoundModel(
        identifiers=[
            CompoundIdentifierModel(type="NAME", value="trace reagent"),
            CompoundIdentifierModel(type="SMILES", value="CCO"),
        ],
        amount=AmountModel(value=1.0, units="mmol"),
        reaction_role="REACTANT",
        is_limiting=False,
    )
    violations = LimitingReagentIsActuallyLimiting().check(aspirin_draft)
    assert "STO-008" in _ids(violations)
