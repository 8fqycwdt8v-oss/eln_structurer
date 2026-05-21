"""Tests for the heuristic reaction classifier and class-specific rules."""

from __future__ import annotations

from eln_structurer.reaction_class import ReactionClass, classify_reaction
from eln_structurer.rules.class_specific import (
    AmideCouplingHasAmineAndAcid,
    BocDeprotectionNeedsAcid,
    BuchwaldHartwigComponents,
    EsterificationNeedsBothPartners,
    GrignardRequiresInertAtmosphere,
    MitsunobuComponents,
    ReductionNeedsReducingAgent,
    ReductiveAminationHasCarbonylAndAmine,
    SuzukiRequiredComponents,
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
)
from tests.conftest import rule_ids as _ids


def _make_draft(
    inputs: list[ReactionInputModel],
    *,
    atmosphere: str | None = None,
    products: list[ProductModel] | None = None,
) -> ReactionDraft:
    return ReactionDraft(
        identifiers=[],
        inputs=inputs,
        conditions=ConditionsModel(
            temperature=TemperatureModel(control_type="AMBIENT"),
            atmosphere=atmosphere,
        ),
        outcomes=[
            OutcomeModel(
                products=products
                or [
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


def _input(name: str, role: str, *, smiles: str | None = None, limiting: bool = False):
    idents = [CompoundIdentifierModel(type="NAME", value=name)]
    if smiles:
        idents.append(CompoundIdentifierModel(type="SMILES", value=smiles))
    return ReactionInputModel(
        name=name,
        components=[
            CompoundModel(
                identifiers=idents,
                amount=AmountModel(value=1.0, units="mmol"),
                reaction_role=role,
                is_limiting=limiting,
            )
        ],
    )


def test_classifier_detects_suzuki() -> None:
    draft = _make_draft([
        _input("4-bromoanisole", "REACTANT", limiting=True),
        _input("phenylboronic acid", "REACTANT"),
        _input("Pd(PPh3)4", "CATALYST"),
        _input("K2CO3", "REAGENT"),
    ])
    result = classify_reaction(draft)
    assert result.cls == ReactionClass.SUZUKI_COUPLING


def test_classifier_detects_grignard() -> None:
    draft = _make_draft([
        _input("magnesium turnings", "REACTANT"),
        _input("bromobenzene", "REACTANT", limiting=True),
    ])
    result = classify_reaction(draft)
    assert result.cls == ReactionClass.GRIGNARD


def test_classifier_detects_reduction() -> None:
    draft = _make_draft([
        _input("benzaldehyde", "REACTANT", limiting=True),
        _input("NaBH4", "REAGENT"),
    ])
    result = classify_reaction(draft)
    assert result.cls == ReactionClass.REDUCTION


def test_classifier_returns_unknown_for_generic_reaction() -> None:
    draft = _make_draft([
        _input("compound A", "REACTANT", limiting=True),
        _input("compound B", "REACTANT"),
    ])
    assert classify_reaction(draft).cls == ReactionClass.UNKNOWN


def test_suzuki_missing_base_fires() -> None:
    """Suzuki coupling without a base should fire CLS-001."""
    draft = _make_draft([
        _input("4-bromoanisole", "REACTANT", limiting=True),
        _input("phenylboronic acid", "REACTANT"),
        _input("Pd(PPh3)4", "CATALYST"),
        # K2CO3 deliberately omitted
    ])
    violations = SuzukiRequiredComponents().check(draft)
    assert "CLS-001" in _ids(violations)


def test_suzuki_with_base_passes() -> None:
    draft = _make_draft([
        _input("4-bromoanisole", "REACTANT", limiting=True),
        _input("phenylboronic acid", "REACTANT"),
        _input("Pd(PPh3)4", "CATALYST"),
        _input("K2CO3", "REAGENT"),
    ])
    assert SuzukiRequiredComponents().check(draft) == []


def test_grignard_without_atmosphere_fires_cls002() -> None:
    draft = _make_draft(
        [
            _input("magnesium turnings", "REACTANT"),
            _input("bromobenzene", "REACTANT", limiting=True),
        ],
        atmosphere=None,
    )
    violations = GrignardRequiresInertAtmosphere().check(draft)
    assert "CLS-002" in _ids(violations)


def test_grignard_with_argon_passes() -> None:
    draft = _make_draft(
        [
            _input("magnesium turnings", "REACTANT"),
            _input("bromobenzene", "REACTANT", limiting=True),
        ],
        atmosphere="argon",
    )
    assert GrignardRequiresInertAtmosphere().check(draft) == []


def test_reduction_with_reductant_in_workup_only_fires() -> None:
    """If the reducing agent is only in a workup, CLS-003 fires."""
    # Construct a draft where the classifier still detects REDUCTION (the
    # NaBH4 name is somewhere in the inputs) but its role is wrong.
    draft = _make_draft([
        _input("benzaldehyde", "REACTANT", limiting=True),
        _input("NaBH4", "PRODUCT"),  # deliberately wrong role
    ])
    violations = ReductionNeedsReducingAgent().check(draft)
    assert "CLS-003" in _ids(violations)


def test_reduction_with_proper_reagent_role_passes() -> None:
    draft = _make_draft([
        _input("benzaldehyde", "REACTANT", limiting=True),
        _input("NaBH4", "REAGENT"),
    ])
    assert ReductionNeedsReducingAgent().check(draft) == []


def test_esterification_classifier_and_rule() -> None:
    draft = _make_draft([
        _input("acetic acid", "REACTANT", limiting=True),
        _input("ethanol", "REACTANT"),
        _input("sulfuric acid", "CATALYST"),
    ])
    assert classify_reaction(draft).cls == ReactionClass.ESTERIFICATION
    assert EsterificationNeedsBothPartners().check(draft) == []


def test_classifier_detects_amide_coupling() -> None:
    draft = _make_draft([
        _input("benzoic acid", "REACTANT", limiting=True),
        _input("aniline", "REACTANT"),
        _input("EDC", "REAGENT"),
        _input("HOBt", "REAGENT"),
    ])
    assert classify_reaction(draft).cls == ReactionClass.AMIDE_FORMATION


def test_classifier_detects_buchwald_hartwig() -> None:
    draft = _make_draft([
        _input("4-bromoanisole", "REACTANT", limiting=True),
        _input("morpholine", "REACTANT"),
        _input("Pd(OAc)2", "CATALYST"),
        _input("XPhos", "REAGENT"),
        _input("Cs2CO3", "REAGENT"),
    ])
    assert classify_reaction(draft).cls == ReactionClass.BUCHWALD_HARTWIG


def test_classifier_detects_reductive_amination() -> None:
    draft = _make_draft([
        _input("4-aminobenzaldehyde", "REACTANT", limiting=True),
        _input("piperidine", "REACTANT"),
        _input("NaBH(OAc)3", "REAGENT"),
    ])
    assert classify_reaction(draft).cls == ReactionClass.REDUCTIVE_AMINATION


def test_classifier_detects_mitsunobu() -> None:
    draft = _make_draft([
        _input("phenol", "REACTANT", limiting=True),
        _input("methanol", "REACTANT"),
        _input("DIAD", "REAGENT"),
        _input("PPh3", "REAGENT"),
    ])
    assert classify_reaction(draft).cls == ReactionClass.MITSUNOBU


def test_classifier_detects_boc_deprotection() -> None:
    draft = _make_draft([
        _input("N-Boc-piperidine", "REACTANT", limiting=True),
        _input("TFA", "REAGENT"),
    ])
    assert classify_reaction(draft).cls == ReactionClass.BOC_DEPROTECTION


def test_classifier_detects_halogenation() -> None:
    draft = _make_draft([
        _input("anisole", "REACTANT", limiting=True),
        _input("NBS", "REAGENT"),
    ])
    assert classify_reaction(draft).cls == ReactionClass.HALOGENATION


def test_classifier_detects_oxidation() -> None:
    draft = _make_draft([
        _input("benzyl alcohol", "REACTANT", limiting=True),
        _input("Dess-Martin periodinane", "REAGENT"),
    ])
    assert classify_reaction(draft).cls == ReactionClass.OXIDATION


def test_amide_coupling_warns_without_amine_reactant() -> None:
    """Coupling reagent present but no amine REACTANT → CLS-005 fires."""
    draft = _make_draft([
        _input("benzoic acid", "REACTANT", limiting=True),
        # aniline is mis-flagged as SOLVENT here
        _input("aniline", "SOLVENT"),
        _input("EDC", "REAGENT"),
    ])
    violations = AmideCouplingHasAmineAndAcid().check(draft)
    assert "CLS-005" in _ids(violations)


def test_amide_coupling_passes_when_both_partners_reactant() -> None:
    draft = _make_draft([
        _input("benzoic acid", "REACTANT", limiting=True),
        _input("aniline", "REACTANT"),
        _input("HATU", "REAGENT"),
    ])
    assert AmideCouplingHasAmineAndAcid().check(draft) == []


def test_boc_deprotection_without_acid_fires() -> None:
    draft = _make_draft([
        _input("N-Boc-piperidine", "REACTANT", limiting=True),
        _input("deprotection helper", "REAGENT"),  # not an acid
    ])
    violations = BocDeprotectionNeedsAcid().check(draft)
    assert "CLS-006" in _ids(violations)


def test_boc_deprotection_with_tfa_passes() -> None:
    draft = _make_draft([
        _input("N-Boc-piperidine", "REACTANT", limiting=True),
        _input("TFA", "REAGENT"),
    ])
    assert BocDeprotectionNeedsAcid().check(draft) == []


def test_reductive_amination_with_full_partners_passes() -> None:
    draft = _make_draft([
        _input("4-aminobenzaldehyde", "REACTANT", limiting=True),  # has both
        _input("piperidine", "REACTANT"),
        _input("NaBH(OAc)3", "REAGENT"),
    ])
    assert ReductiveAminationHasCarbonylAndAmine().check(draft) == []


def test_reductive_amination_warns_without_amine() -> None:
    draft = _make_draft([
        _input("benzaldehyde", "REACTANT", limiting=True),
        _input("NaBH(OAc)3", "REAGENT"),
        # no amine
    ])
    violations = ReductiveAminationHasCarbonylAndAmine().check(draft)
    assert "CLS-007" in _ids(violations)


def test_buchwald_passes_with_all_components() -> None:
    draft = _make_draft([
        _input("4-bromoanisole", "REACTANT", limiting=True),
        _input("morpholine", "REACTANT"),
        _input("Pd(OAc)2", "CATALYST"),
        _input("XPhos", "REAGENT"),
        _input("Cs2CO3", "REAGENT"),
    ])
    assert BuchwaldHartwigComponents().check(draft) == []


def test_buchwald_errors_without_base() -> None:
    draft = _make_draft([
        _input("4-bromoanisole", "REACTANT", limiting=True),
        _input("morpholine", "REACTANT"),
        _input("Pd(OAc)2", "CATALYST"),
        _input("XPhos", "REAGENT"),
        # base intentionally absent
    ])
    violations = BuchwaldHartwigComponents().check(draft)
    assert "CLS-008" in _ids(violations)


def test_mitsunobu_errors_without_pph3() -> None:
    draft = _make_draft([
        _input("phenol", "REACTANT", limiting=True),
        _input("methanol", "REACTANT"),
        _input("DIAD", "REAGENT"),
        # PPh3 intentionally absent
    ])
    # The classifier requires BOTH DIAD and PPh3 to fire; without PPh3,
    # the classifier returns UNKNOWN and CLS-009 is silent. To exercise
    # the rule body, classify it as Mitsunobu via the literal name.
    draft.inputs[0].components[0].identifiers[0].value = "phenol (Mitsunobu)"
    violations = MitsunobuComponents().check(draft)
    assert "CLS-009" in _ids(violations)


def test_esterification_warns_when_alcohol_misclassified() -> None:
    draft = _make_draft([
        _input("acetic acid", "REACTANT", limiting=True),
        _input("ethanol", "SOLVENT"),  # ethanol should be REACTANT
        _input("sulfuric acid", "CATALYST"),
    ])
    violations = EsterificationNeedsBothPartners().check(draft)
    assert "CLS-004" in _ids(violations)
