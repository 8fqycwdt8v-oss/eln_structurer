"""Coverage-gap closing tests: STO-001, STO-002, ORD-001."""

from __future__ import annotations

from eln_structurer.rules.ordering import WorkupKeywordsDeclared
from eln_structurer.rules.stoichiometry import (
    AmountHasUnits,
    EquivalentsConsistentWithLimiting,
)
from eln_structurer.schema import (
    AmountModel,
    CompoundIdentifierModel,
    CompoundModel,
    ConditionsModel,
    OutcomeModel,
    ProductModel,
    ReactionDraft,
    ReactionInputModel,
    TemperatureModel,
    WorkupModel,
)
from tests.conftest import rule_ids as _ids


def _minimal_draft() -> ReactionDraft:
    return ReactionDraft(
        identifiers=[],
        inputs=[
            ReactionInputModel(
                name="r",
                components=[
                    CompoundModel(
                        identifiers=[CompoundIdentifierModel(type="NAME", value="x")],
                        amount=AmountModel(value=1.0, units="mmol"),
                        reaction_role="REACTANT",
                        is_limiting=True,
                    )
                ],
            )
        ],
        conditions=ConditionsModel(temperature=TemperatureModel(control_type="AMBIENT")),
        outcomes=[
            OutcomeModel(
                products=[
                    ProductModel(
                        compound=CompoundModel(
                            identifiers=[CompoundIdentifierModel(type="NAME", value="p")],
                            reaction_role="PRODUCT",
                        )
                    )
                ]
            )
        ],
        notes="n",
        source_paragraph="p",
    )


# STO-001 -------------------------------------------------------------------


def test_sto001_passes_with_well_formed_amounts(aspirin_draft) -> None:
    """STO-001 is defensive — Pydantic already validates the units enum.
    A clean draft must produce zero violations."""
    assert AmountHasUnits().check(aspirin_draft) == []


def test_sto001_silent_when_amount_omitted() -> None:
    """A compound without an amount field shouldn't trigger STO-001."""
    draft = _minimal_draft()
    draft.inputs[0].components[0].amount = None
    draft.inputs[0].components[0].is_limiting = False
    draft.inputs.append(
        ReactionInputModel(
            name="r2",
            components=[
                CompoundModel(
                    identifiers=[CompoundIdentifierModel(type="NAME", value="y")],
                    amount=AmountModel(value=1.0, units="mmol"),
                    reaction_role="REACTANT",
                    is_limiting=True,
                )
            ],
        )
    )
    assert AmountHasUnits().check(draft) == []


# STO-002 (currently positivity-only) ---------------------------------------


def test_sto002_passes_for_positive_equivalents(aspirin_draft) -> None:
    assert EquivalentsConsistentWithLimiting().check(aspirin_draft) == []


def test_sto002_silent_without_limiting_reagent() -> None:
    """No limiting reagent → STO-002 returns silently (can't make claim)."""
    draft = _minimal_draft()
    draft.inputs[0].components[0].is_limiting = False
    assert EquivalentsConsistentWithLimiting().check(draft) == []


# ORD-001 -------------------------------------------------------------------


def test_ord001_warns_when_workup_keyword_undeclared() -> None:
    draft = _minimal_draft()
    draft.workups.append(
        WorkupModel(
            type="WASH",
            description="washed with brine (3 x 10 mL)",  # brine not declared
            order=1,
        )
    )
    violations = WorkupKeywordsDeclared().check(draft)
    assert "ORD-001" in _ids(violations)


def test_ord001_silent_when_brine_declared(aspirin_draft) -> None:
    """If a workup mentions brine AND brine appears as a component
    elsewhere in the draft, ORD-001 stays silent."""
    aspirin_draft.workups.append(
        WorkupModel(
            type="WASH",
            description="washed with brine",
            components=[
                CompoundModel(
                    identifiers=[CompoundIdentifierModel(type="NAME", value="brine")],
                    reaction_role="WORKUP",
                )
            ],
            order=10,
        )
    )
    violations = WorkupKeywordsDeclared().check(aspirin_draft)
    # The new brine workup is fine; the OTHER existing workups should
    # not produce ORD-001 either.
    assert "ORD-001" not in _ids(violations)
