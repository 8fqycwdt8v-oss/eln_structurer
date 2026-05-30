"""In-silico priors for the predictor.

This Tier-4 surface is deliberately thin. The eventual production
implementation will call out to xtb / CREST → DFT pipelines for HOMO/
LUMO energies and steric maps; ChemProp models for pKa / logP /
solubility; and curated transition-state libraries for class × substrate
pattern matching. Those backends are heavy and version-sensitive, and
the project is locked to "no public APIs except the LLM" anyway.

For now we ship:

- :func:`local_descriptors` — RDKit-only molecular descriptors (HBA,
  HBD, TPSA, rotatable bonds, formula weight) that are cheap, exact,
  and useful for downstream "is this substrate hindered / polar /
  small?" heuristics.
- :class:`DescriptorProfile` — frozen result type so callers can pass
  it around without coupling to RDKit.
- :func:`backend_available` — explicit capability probe so the agent
  knows when it can ask for DFT-level data and when to stop trying.

The architecture is in place; the real models plug in by replacing the
bodies of these functions. Every call has a graceful "not available"
return path so the predictor never crashes when a heavyweight backend
is offline.
"""

from __future__ import annotations

from dataclasses import dataclass

from eln_structurer.chemistry import parse_mol


@dataclass(frozen=True)
class DescriptorProfile:
    """RDKit-derived descriptors for a single substrate.

    Every field is optional — when parsing fails or RDKit lacks the
    descriptor we return ``None`` for that field rather than 0, so the
    composer can distinguish "we computed zero rotatable bonds" from
    "we couldn't compute".
    """
    smiles: str
    canonical_smiles: str | None = None
    mol_weight: float | None = None
    heavy_atom_count: int | None = None
    h_bond_donors: int | None = None
    h_bond_acceptors: int | None = None
    rotatable_bonds: int | None = None
    tpsa: float | None = None
    log_p_crippen: float | None = None
    rings: int | None = None
    aromatic_rings: int | None = None


def local_descriptors(smiles: str) -> DescriptorProfile | None:
    """Compute RDKit descriptors for ``smiles``. Returns None on parse failure.

    All descriptors come from RDKit's built-ins — no network, no
    external models. Used as Tier-4 "always-available" priors.
    """
    mol = parse_mol(smiles)
    if mol is None:
        return None
    try:
        from rdkit import Chem
        from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors
    except Exception:           # pragma: no cover - defensive
        return DescriptorProfile(smiles=smiles)
    try:
        canon = Chem.MolToSmiles(mol)
    except Exception:           # pragma: no cover - defensive
        canon = None
    return DescriptorProfile(
        smiles=smiles,
        canonical_smiles=canon,
        mol_weight=Descriptors.MolWt(mol),
        heavy_atom_count=mol.GetNumHeavyAtoms(),
        h_bond_donors=Lipinski.NumHDonors(mol),
        h_bond_acceptors=Lipinski.NumHAcceptors(mol),
        rotatable_bonds=Lipinski.NumRotatableBonds(mol),
        tpsa=Descriptors.TPSA(mol),
        log_p_crippen=Crippen.MolLogP(mol),
        rings=rdMolDescriptors.CalcNumRings(mol),
        aromatic_rings=rdMolDescriptors.CalcNumAromaticRings(mol),
    )


# ---------------------------------------------------------------------------
# Heavyweight-backend availability probe
# ---------------------------------------------------------------------------


_BACKENDS: dict[str, bool] = {
    # Defaults reflect the no-network constraint: every external compute
    # backend is "not available" until somebody flips this in a
    # deployment-specific bootstrap.
    "xtb": False,
    "crest": False,
    "chemprop": False,
    "dft": False,
}


def backend_available(name: str) -> bool:
    """Report whether the named heavyweight backend is installed.

    Returns ``False`` for unknown names — callers should fall back to
    the always-available local descriptors when this returns False.
    """
    return _BACKENDS.get(name, False)


def register_backend(name: str, *, available: bool) -> None:
    """Test / deployment hook to enable a backend after first install."""
    _BACKENDS[name] = available


__all__ = [
    "DescriptorProfile",
    "local_descriptors",
    "backend_available",
    "register_backend",
]
