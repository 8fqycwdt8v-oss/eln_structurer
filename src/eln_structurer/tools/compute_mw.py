"""compute_mw — RDKit-based molecular-weight lookup tool.

Pure core function ``compute_mw_from_smiles`` is independent of the SDK.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from claude_agent_sdk import tool

from eln_structurer.chemistry import heavy_atoms, mol_weight, parse_mol


@dataclass(frozen=True)
class MwResult:
    ok: bool
    smiles: str
    canonical_smiles: str | None
    mw_g_per_mol: float | None
    heavy_atom_count: int | None
    error: str | None

    @classmethod
    def valid(cls, smiles: str, canonical: str, mw: float, ha: int) -> "MwResult":
        return cls(
            ok=True,
            smiles=smiles,
            canonical_smiles=canonical,
            mw_g_per_mol=mw,
            heavy_atom_count=ha,
            error=None,
        )

    @classmethod
    def invalid(cls, smiles: str, error: str) -> "MwResult":
        return cls(
            ok=False,
            smiles=smiles,
            canonical_smiles=None,
            mw_g_per_mol=None,
            heavy_atom_count=None,
            error=error,
        )


def compute_mw_from_smiles(smiles: str) -> MwResult:
    """Compute MW and heavy-atom count from a SMILES, locally via RDKit."""
    if not isinstance(smiles, str) or not smiles.strip():
        return MwResult.invalid(smiles, "smiles must be a non-empty string")
    mol = parse_mol(smiles)
    if mol is None:
        return MwResult.invalid(smiles, f"RDKit cannot parse {smiles!r}")
    from rdkit import Chem
    canon = Chem.MolToSmiles(mol)
    mw = mol_weight(smiles)
    ha = heavy_atoms(smiles)
    assert mw is not None and ha is not None  # parse_mol succeeded
    return MwResult.valid(smiles, canon, mw, ha)


@tool(
    "compute_mw",
    (
        "Given a SMILES, return the molecular weight (g/mol), the canonical "
        "SMILES, and the heavy-atom count. Use this to cross-check equivalents "
        "claims (claimed_equiv = (m_g / MW) / n_limiting) and to sanity-check "
        "product atom budgets before finalize_reaction."
    ),
    {"smiles": str},
)
async def compute_mw(args: dict[str, Any]) -> dict[str, Any]:
    result = compute_mw_from_smiles(args.get("smiles", ""))
    if not result.ok:
        return {
            "content": [{"type": "text", "text": f"ERROR: {result.error}"}],
            "isError": True,
        }
    return {
        "content": [
            {
                "type": "text",
                "text": (
                    f"SMILES (canonical): {result.canonical_smiles}\n"
                    f"MW: {result.mw_g_per_mol:.2f} g/mol\n"
                    f"Heavy atoms: {result.heavy_atom_count}"
                ),
            }
        ]
    }
