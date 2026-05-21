"""Shared helpers for reading and validating compound data.

Centralizes:
- Identifier lookup (NAME / SMILES) on a CompoundModel
- Cached RDKit parsing — the same SMILES often gets read by 3–4 rules in a
  single `run_harness` pass; caching turns the repeat parses into dict lookups.

Keep this module free of network access; it only wraps local libraries.
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
    """RDKit MolFromSmiles, cached. Returns None for unparseable input."""
    if not smiles:
        return None
    return Chem.MolFromSmiles(smiles)


def heavy_atoms(smiles: str) -> int | None:
    mol = parse_mol(smiles)
    return None if mol is None else mol.GetNumHeavyAtoms()


def mol_weight(smiles: str) -> float | None:
    mol = parse_mol(smiles)
    return None if mol is None else Descriptors.MolWt(mol)


def canonical_smiles(smiles: str) -> str | None:
    mol = parse_mol(smiles)
    return None if mol is None else Chem.MolToSmiles(mol)
