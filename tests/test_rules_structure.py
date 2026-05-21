"""Unit tests for the structure rule pack."""

from __future__ import annotations

from eln_structurer.rules.structure import (
    AtomBalanceSanity,
    NameOrSmilesPresent,
    SmilesParses,
)
from eln_structurer.schema import (
    CompoundIdentifierModel,
    CompoundModel,
    ProductModel,
    ReactionDraft,
)
from tests.conftest import rule_ids as _ids


def test_smiles_parses_passes_for_valid(aspirin_draft: ReactionDraft) -> None:
    assert SmilesParses().check(aspirin_draft) == []


def test_smiles_parses_fires_for_garbage(aspirin_draft: ReactionDraft) -> None:
    # Inject a non-parseable SMILES.
    aspirin_draft.inputs[0].components[0].identifiers.append(
        CompoundIdentifierModel(type="SMILES", value="this-is-not-smiles[[[")
    )
    violations = SmilesParses().check(aspirin_draft)
    assert "STR-001" in _ids(violations)


def test_name_or_smiles_present_passes(aspirin_draft: ReactionDraft) -> None:
    assert NameOrSmilesPresent().check(aspirin_draft) == []


def test_name_or_smiles_present_fires_when_only_cas(aspirin_draft: ReactionDraft) -> None:
    aspirin_draft.inputs[0].components[0].identifiers = [
        CompoundIdentifierModel(type="CAS_NUMBER", value="69-72-7"),
    ]
    violations = NameOrSmilesPresent().check(aspirin_draft)
    assert "STR-002" in _ids(violations)


def test_smiles_consistency_passes_for_equivalent_smiles(aspirin_draft: ReactionDraft) -> None:
    """Two SMILES that canonicalize to the same molecule are fine."""
    from eln_structurer.rules.structure import SmilesIdentifiersAreConsistent
    # Add an alternative SMILES that's equivalent to salicylic acid.
    aspirin_draft.inputs[0].components[0].identifiers.append(
        CompoundIdentifierModel(type="SMILES", value="O=C(O)c1ccccc1O")
    )
    assert SmilesIdentifiersAreConsistent().check(aspirin_draft) == []


def test_smiles_consistency_fires_on_distinct_molecules(aspirin_draft: ReactionDraft) -> None:
    """Two SMILES on the same compound that name different molecules."""
    from eln_structurer.rules.structure import SmilesIdentifiersAreConsistent
    # Add a SMILES for a different molecule (phenol).
    aspirin_draft.inputs[0].components[0].identifiers.append(
        CompoundIdentifierModel(type="SMILES", value="c1ccc(O)cc1")
    )
    violations = SmilesIdentifiersAreConsistent().check(aspirin_draft)
    assert "STR-004" in _ids(violations)


def test_atom_balance_warns_when_product_too_big(aspirin_draft: ReactionDraft) -> None:
    # Replace the product with a much larger molecule.
    big_product = ProductModel(
        compound=CompoundModel(
            identifiers=[
                CompoundIdentifierModel(type="NAME", value="impossible"),
                CompoundIdentifierModel(
                    type="SMILES",
                    value="C" * 100,  # 100-carbon chain
                ),
            ],
            reaction_role="PRODUCT",
        )
    )
    aspirin_draft.outcomes[0].products = [big_product]
    violations = AtomBalanceSanity().check(aspirin_draft)
    assert "STR-003" in _ids(violations)
