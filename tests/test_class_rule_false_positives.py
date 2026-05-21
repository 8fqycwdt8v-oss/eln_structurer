"""False-positive regression tests for class-specific rules.

Class-specific rules can hurt model performance two ways:
1. Firing on the WRONG class (classifier says X but rule for class Y still
   triggers) — the agent then chases a phantom error and may diverge.
2. Firing on a CORRECTLY-classified reaction that legitimately omits the
   thing the rule looks for (e.g. an intramolecular Wittig has no
   separate carbonyl REACTANT).

These tests pin both behaviours. For every class-specific rule we assert:
- it produces NO violations on the aspirin fixture (UNKNOWN classification).
- it produces NO violations on a minimal correctly-formed example of its
  own class.
- it DOES fire on a deliberately-broken example of its own class.
"""

from __future__ import annotations

from eln_structurer.rules import ALL_RULES
from eln_structurer.rules.class_specific import (
    AmideCouplingHasAmineAndAcid,
    BocDeprotectionNeedsAcid,
    BuchwaldHartwigComponents,
    EsterificationNeedsBothPartners,
    GrignardRequiresInertAtmosphere,
    HalogenatingAgentIsNotLimiting,
    MitsunobuComponents,
    NAlkylationNeedsElectrophile,
    OxidantIsNotLimiting,
    ReductionNeedsReducingAgent,
    ReductiveAminationHasCarbonylAndAmine,
    SuzukiRequiredComponents,
    WittigNeedsCarbonyl,
)
from eln_structurer.reaction_class import ReactionClass, classify_reaction
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
)
from tests.conftest import rule_ids as _ids


# ---------------------------------------------------------------------------
# Helper builders kept tiny so the test reads as data.
# ---------------------------------------------------------------------------

def _input(name: str, role: str, *, limiting: bool = False):
    return ReactionInputModel(
        name=name,
        components=[
            CompoundModel(
                identifiers=[CompoundIdentifierModel(type="NAME", value=name)],
                amount=AmountModel(value=1.0, units="mmol"),
                reaction_role=role,
                is_limiting=limiting,
            )
        ],
    )


def _draft(inputs: list[ReactionInputModel], *, atmosphere: str | None = None) -> ReactionDraft:
    return ReactionDraft(
        identifiers=[],
        inputs=inputs,
        conditions=ConditionsModel(
            temperature=TemperatureModel(control_type="AMBIENT"),
            atmosphere=atmosphere,
        ),
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


# ---------------------------------------------------------------------------
# Aspirin fixture is UNKNOWN-classified — none of the class rules may fire.
# ---------------------------------------------------------------------------

ALL_CLASS_RULES = [
    SuzukiRequiredComponents(),
    GrignardRequiresInertAtmosphere(),
    ReductionNeedsReducingAgent(),
    EsterificationNeedsBothPartners(),
    AmideCouplingHasAmineAndAcid(),
    BocDeprotectionNeedsAcid(),
    ReductiveAminationHasCarbonylAndAmine(),
    BuchwaldHartwigComponents(),
    MitsunobuComponents(),
    WittigNeedsCarbonyl(),
    HalogenatingAgentIsNotLimiting(),
    OxidantIsNotLimiting(),
    NAlkylationNeedsElectrophile(),
]


def test_aspirin_fixture_classifies_as_unknown(aspirin_draft: ReactionDraft) -> None:
    """Aspirin fixture is acetic anhydride + salicylic acid + H2SO4 — none
    of our class patterns should claim it."""
    assert classify_reaction(aspirin_draft).cls == ReactionClass.UNKNOWN


def test_no_class_rule_fires_on_aspirin(aspirin_draft: ReactionDraft) -> None:
    """Every single class-specific rule must report zero violations on the
    aspirin fixture. Crucial regression guard against the 'rule fires on
    every reaction' failure mode that would force the model into endless
    repair loops."""
    for rule in ALL_CLASS_RULES:
        violations = rule.check(aspirin_draft)
        assert violations == [], (
            f"{rule.__class__.__name__} ({rule.id}) wrongly fired on aspirin: "
            f"{[v.message for v in violations]}"
        )


# ---------------------------------------------------------------------------
# Each new rule: positive case (clean) + negative case (rule fires).
# Existing rules are already covered by test_reaction_class.py.
# ---------------------------------------------------------------------------

# WITTIG (CLS-010) -----------------------------------------------------------

def test_wittig_passes_when_carbonyl_present() -> None:
    draft = _draft([
        _input("benzaldehyde", "REACTANT", limiting=True),
        _input("Ph3P=CHCO2Et ylide", "REAGENT"),
    ])
    assert classify_reaction(draft).cls == ReactionClass.WITTIG
    assert WittigNeedsCarbonyl().check(draft) == []


def test_wittig_warns_without_carbonyl() -> None:
    """Wittig classifier fired but no aldehyde/ketone in REACTANTs."""
    draft = _draft([
        _input("substrate X", "REACTANT", limiting=True),
        _input("Ph3P=CHCO2Et ylide", "REAGENT"),
    ])
    violations = WittigNeedsCarbonyl().check(draft)
    assert "CLS-010" in _ids(violations)


# HALOGENATION (CLS-011) -----------------------------------------------------

def test_halogenation_passes_when_substrate_is_limiting() -> None:
    draft = _draft([
        _input("anisole", "REACTANT", limiting=True),
        _input("NBS", "REAGENT"),
    ])
    assert classify_reaction(draft).cls == ReactionClass.HALOGENATION
    assert HalogenatingAgentIsNotLimiting().check(draft) == []


def test_halogenation_fires_when_nbs_is_limiting() -> None:
    """NBS wrongly flagged as limiting reagent."""
    draft = _draft([
        _input("anisole", "REACTANT"),
        _input("NBS", "REAGENT", limiting=True),
    ])
    violations = HalogenatingAgentIsNotLimiting().check(draft)
    assert "CLS-011" in _ids(violations)


# OXIDATION (CLS-012) --------------------------------------------------------

def test_oxidation_passes_when_substrate_is_limiting() -> None:
    draft = _draft([
        _input("benzyl alcohol", "REACTANT", limiting=True),
        _input("Dess-Martin periodinane", "REAGENT"),
    ])
    assert classify_reaction(draft).cls == ReactionClass.OXIDATION
    assert OxidantIsNotLimiting().check(draft) == []


def test_oxidation_fires_when_oxidant_is_limiting() -> None:
    draft = _draft([
        _input("benzyl alcohol", "REACTANT"),
        _input("Dess-Martin periodinane", "REAGENT", limiting=True),
    ])
    violations = OxidantIsNotLimiting().check(draft)
    assert "CLS-012" in _ids(violations)


# N_ALKYLATION (CLS-013) -----------------------------------------------------

def test_n_alkylation_passes_with_amine_and_electrophile() -> None:
    draft = _draft([
        _input("morpholine", "REACTANT", limiting=True),
        _input("benzyl bromide", "REACTANT"),
        _input("N-alkylation", "REAGENT"),  # hint to trigger the classifier
    ])
    assert classify_reaction(draft).cls == ReactionClass.N_ALKYLATION
    assert NAlkylationNeedsElectrophile().check(draft) == []


def test_n_alkylation_warns_without_electrophile() -> None:
    draft = _draft([
        _input("morpholine", "REACTANT", limiting=True),
        _input("substrate X", "REACTANT"),  # no halide / mesylate / etc.
        _input("N-alkylation reagent", "REAGENT"),
    ])
    violations = NAlkylationNeedsElectrophile().check(draft)
    assert "CLS-013" in _ids(violations)


# ---------------------------------------------------------------------------
# Performance regression: aspirin must remain clean across the FULL rule
# pack, not just the class-specific rules. This guards against any new
# rule that incidentally fires on the canonical fixture.
# ---------------------------------------------------------------------------

def test_full_rule_pack_does_not_error_on_aspirin(aspirin_draft: ReactionDraft) -> None:
    """No ERROR-severity violation on the aspirin fixture across ALL rules.
    Warnings are acceptable; errors would block finalize and starve the
    repair loop."""
    from eln_structurer.rules.base import Severity
    error_ids: list[str] = []
    for rule in ALL_RULES:
        for v in rule.check(aspirin_draft):
            if v.severity is Severity.ERROR:
                error_ids.append(f"{v.rule_id}: {v.message}")
    assert error_ids == [], "Aspirin fixture must not produce ERROR-severity violations:\n" + "\n".join(error_ids)
