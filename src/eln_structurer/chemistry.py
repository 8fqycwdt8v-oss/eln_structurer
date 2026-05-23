"""Local chemistry primitives shared across rules, tools, and benchmarks.

Provides:
- Identifier lookup (NAME / SMILES) on a ``CompoundModel``.
- RDKit parsing wrapped in an LRU cache; the same SMILES is read by several
  rules per ``run_harness`` pass and by the validate_smiles tool, so caching
  turns repeats into dict lookups.

This module is the only place in the codebase that touches RDKit directly
for compound work. Network access is forbidden: it stays purely local.
"""

from __future__ import annotations

from functools import lru_cache

from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import Descriptors
from rdkit.Chem.rdchem import Mol

from eln_structurer.schema import CompoundModel

RDLogger.DisableLog("rdApp.*")


def smiles_of(comp: CompoundModel) -> str | None:
    """Return the first SMILES identifier on ``comp``, or None."""
    for ident in comp.identifiers:
        if ident.type == "SMILES":
            return ident.value
    return None


def name_of(comp: CompoundModel) -> str | None:
    """Return the first NAME or IUPAC_NAME identifier on ``comp``, or None."""
    for ident in comp.identifiers:
        if ident.type in {"NAME", "IUPAC_NAME"}:
            return ident.value
    return None


def has_name_or_smiles(comp: CompoundModel) -> bool:
    return any(i.type in {"NAME", "SMILES", "IUPAC_NAME"} for i in comp.identifiers)


@lru_cache(maxsize=2048)
def parse_mol(smiles: str) -> Mol | None:
    """RDKit MolFromSmiles, cached. Returns None for unparseable input.

    Catches every exception RDKit can raise — segfaults aside, malformed
    SMILES can also raise ValueError, RuntimeError, or OverflowError on
    very large strings. Returning None here keeps the agent loop alive
    instead of letting an adversarial input crash the entire extraction.
    """
    if not smiles:
        return None
    try:
        return Chem.MolFromSmiles(smiles)
    except Exception:  # noqa: BLE001 — defensive, see docstring
        return None


def heavy_atoms(smiles: str) -> int | None:
    mol = parse_mol(smiles)
    return None if mol is None else mol.GetNumHeavyAtoms()


def mol_weight(smiles: str) -> float | None:
    mol = parse_mol(smiles)
    return None if mol is None else Descriptors.MolWt(mol)


def canonical_smiles(smiles: str) -> str | None:
    mol = parse_mol(smiles)
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol)
    except Exception:  # noqa: BLE001 — defensive, same rationale as parse_mol
        return None
