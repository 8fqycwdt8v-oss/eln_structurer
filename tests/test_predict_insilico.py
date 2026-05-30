"""Tier-4 tests: local descriptors + backend probe + DFT lookup."""

from __future__ import annotations

from eln_structurer.predict import (
    DescriptorProfile,
    backend_available,
    known_substrate_tags,
    local_descriptors,
    lookup_activation_energy,
    register_backend,
)
from eln_structurer.reaction_class import ReactionClass


# ---------- local descriptors ---------------------------------------------


def test_local_descriptors_benzene() -> None:
    profile = local_descriptors("c1ccccc1")
    assert isinstance(profile, DescriptorProfile)
    assert profile.heavy_atom_count == 6
    assert profile.aromatic_rings == 1
    assert profile.rings == 1
    # MolWt for C6H6 ≈ 78.11
    assert 77 < profile.mol_weight < 80


def test_local_descriptors_carboxylic_acid_has_donor_and_acceptor() -> None:
    profile = local_descriptors("CC(=O)O")
    assert profile.h_bond_donors == 1
    assert profile.h_bond_acceptors >= 1


def test_local_descriptors_invalid_returns_none() -> None:
    assert local_descriptors("not[a]smiles[[[") is None


def test_local_descriptors_empty_string_returns_none() -> None:
    assert local_descriptors("") is None


# ---------- backend probe -------------------------------------------------


def test_backend_available_defaults_false() -> None:
    # No backend should be available out of the box (network-free config).
    for name in ("xtb", "crest", "chemprop", "dft"):
        assert backend_available(name) is False


def test_backend_available_unknown_name_false() -> None:
    assert backend_available("definitely-not-a-backend") is False


def test_register_backend_round_trip() -> None:
    register_backend("xtb", available=True)
    try:
        assert backend_available("xtb") is True
    finally:
        register_backend("xtb", available=False)


# ---------- DFT lookup ----------------------------------------------------


def test_lookup_activation_energy_known_suzuki() -> None:
    ea = lookup_activation_energy(ReactionClass.SUZUKI_COUPLING, "aryl_bromide")
    assert ea is not None
    assert 50.0 < ea < 80.0


def test_lookup_activation_energy_chloride_higher_than_bromide() -> None:
    # The "harder substrate has higher Ea" relationship is the whole
    # point of the table — pin it explicitly.
    br = lookup_activation_energy(ReactionClass.SUZUKI_COUPLING, "aryl_bromide")
    cl = lookup_activation_energy(ReactionClass.SUZUKI_COUPLING, "aryl_chloride")
    assert br is not None and cl is not None
    assert cl > br


def test_lookup_activation_energy_unknown_returns_none() -> None:
    assert lookup_activation_energy(ReactionClass.SUZUKI_COUPLING, "moon-rock") is None
    # Unknown class entirely.
    assert lookup_activation_energy(ReactionClass.UNKNOWN, "anything") is None


def test_known_substrate_tags_returns_lists() -> None:
    tags = known_substrate_tags(ReactionClass.SUZUKI_COUPLING)
    assert "aryl_bromide" in tags
    assert "aryl_chloride" in tags


def test_known_substrate_tags_empty_for_unknown_class() -> None:
    assert known_substrate_tags(ReactionClass.UNKNOWN) == []
