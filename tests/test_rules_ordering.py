"""Unit tests for the ordering rule pack."""

from __future__ import annotations

from eln_structurer.rules.ordering import (
    QuenchAfterReaction,
    SolventPresentBeforeHeating,
    StirringBeforeHeating,
    WorkupKeywordsDeclared,
    WorkupOrderMonotonic,
)
from eln_structurer.schema import (
    CompoundIdentifierModel,
    CompoundModel,
    StirringModel,
    TemperatureModel,
    WorkupModel,
)
from tests.conftest import rule_ids as _ids


def test_solvent_required_when_heating(aspirin_draft) -> None:
    # Set an active heating control and ensure no SOLVENT input is present.
    aspirin_draft.conditions.temperature = TemperatureModel(
        setpoint_celsius=85.0, control_type="OIL_BATH"
    )
    for inp in aspirin_draft.inputs:
        for comp in inp.components:
            if comp.reaction_role == "SOLVENT":
                comp.reaction_role = "REAGENT"
    violations = SolventPresentBeforeHeating().check(aspirin_draft)
    assert "ORD-002" in _ids(violations)


def test_solvent_not_required_when_ambient(aspirin_draft) -> None:
    aspirin_draft.conditions.temperature = TemperatureModel(
        setpoint_celsius=22.0, control_type="AMBIENT"
    )
    for inp in aspirin_draft.inputs:
        for comp in inp.components:
            if comp.reaction_role == "SOLVENT":
                comp.reaction_role = "REAGENT"
    assert SolventPresentBeforeHeating().check(aspirin_draft) == []


def test_solvent_not_required_for_unspecified_control(aspirin_draft) -> None:
    """Neat / solid-state reactions opt out of ORD-002 via UNSPECIFIED."""
    aspirin_draft.conditions.temperature = TemperatureModel(
        setpoint_celsius=150.0, control_type="UNSPECIFIED"
    )
    for inp in aspirin_draft.inputs:
        for comp in inp.components:
            if comp.reaction_role == "SOLVENT":
                comp.reaction_role = "REAGENT"
    assert SolventPresentBeforeHeating().check(aspirin_draft) == []


def test_stirring_warning_when_heated_unstirred(aspirin_draft) -> None:
    # Heating in a conventional vessel + no stirring → warning.
    aspirin_draft.conditions.temperature = TemperatureModel(
        setpoint_celsius=85.0, control_type="OIL_BATH"
    )
    aspirin_draft.conditions.stirring = StirringModel(type="NONE")
    violations = StirringBeforeHeating().check(aspirin_draft)
    assert "ORD-003" in _ids(violations)


def test_stirring_silent_for_reflux(aspirin_draft) -> None:
    """REFLUX control_type opts out of ORD-003."""
    aspirin_draft.conditions.temperature = TemperatureModel(
        setpoint_celsius=80.0, control_type="REFLUX"
    )
    aspirin_draft.conditions.stirring = StirringModel(type="NONE")
    assert StirringBeforeHeating().check(aspirin_draft) == []


def test_workup_keywords_warn_when_undeclared(aspirin_draft) -> None:
    """ORD-001: description mentions 'brine' but no compound declares it."""
    aspirin_draft.workups.append(
        WorkupModel(
            type="WASH",
            description="The organic layer was washed with brine (20 mL).",
            order=4,
        )
    )
    violations = WorkupKeywordsDeclared().check(aspirin_draft)
    assert "ORD-001" in _ids(violations)


def test_workup_keywords_silent_when_declared(aspirin_draft) -> None:
    """ORD-001 passes when the keyword is also present as a component."""
    aspirin_draft.workups.append(
        WorkupModel(
            type="WASH",
            description="The organic layer was washed with brine (20 mL).",
            components=[
                CompoundModel(
                    identifiers=[CompoundIdentifierModel(type="NAME", value="brine")],
                    reaction_role="WORKUP",
                )
            ],
            order=4,
        )
    )
    assert WorkupKeywordsDeclared().check(aspirin_draft) == []


def test_workup_monotonic_warns_on_decrease(aspirin_draft) -> None:
    aspirin_draft.workups.append(
        WorkupModel(type="WASH", description="extra wash", order=1)
    )
    violations = WorkupOrderMonotonic().check(aspirin_draft)
    assert "ORD-005" in _ids(violations)


def test_quench_order_must_be_positive() -> None:
    from eln_structurer.schema import (
        ConditionsModel,
        ReactionDraft,
        ReactionInputModel,
        CompoundModel,
        CompoundIdentifierModel,
        AmountModel,
        TemperatureModel,
    )
    draft = ReactionDraft(
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
        conditions=ConditionsModel(temperature=TemperatureModel(control_type="AMBIENT")),
        workups=[
            WorkupModel(
                type="ADDITION",
                description="The reaction was quenched with saturated NH4Cl.",
                order=0,
            )
        ],
        notes="n",
        source_paragraph="p",
    )
    violations = QuenchAfterReaction().check(draft)
    assert "ORD-004" in _ids(violations)
