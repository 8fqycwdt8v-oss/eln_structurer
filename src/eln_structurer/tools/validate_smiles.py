"""SMILES validation tool.

The pure ``check_smiles`` function returns a ``SmilesCheck`` result and is
trivially callable from any context. The ``@tool``-decorated ``validate_smiles``
is just a thin SDK marshaler around it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rdkit import Chem

from claude_agent_sdk import tool

from eln_structurer.chemistry import parse_mol


@dataclass(frozen=True)
class SmilesCheck:
    ok: bool
    canonical: str | None
    error: str | None

    @classmethod
    def valid(cls, canonical: str) -> "SmilesCheck":
        return cls(ok=True, canonical=canonical, error=None)

    @classmethod
    def invalid(cls, error: str) -> "SmilesCheck":
        return cls(ok=False, canonical=None, error=error)


def check_smiles(smiles: str) -> SmilesCheck:
    """Validate a SMILES string locally via RDKit. No network access."""
    if not isinstance(smiles, str) or not smiles.strip():
        return SmilesCheck.invalid("smiles must be a non-empty string")
    mol = parse_mol(smiles)
    if mol is None:
        return SmilesCheck.invalid(f"RDKit cannot parse SMILES {smiles!r}")
    return SmilesCheck.valid(Chem.MolToSmiles(mol))


@tool(
    "validate_smiles",
    (
        "Check that a SMILES string parses via RDKit. Returns the canonical SMILES "
        "if valid, or an INVALID message otherwise. Use this to sanity-check your "
        "SMILES before committing them to the draft. No external network is used."
    ),
    {"smiles": str},
)
async def validate_smiles(args: dict[str, Any]) -> dict[str, Any]:
    result = check_smiles(args.get("smiles", ""))
    if not result.ok:
        return {
            "content": [{"type": "text", "text": f"INVALID: {result.error}."}],
            "isError": True,
        }
    return {
        "content": [
            {"type": "text", "text": f"VALID. Canonical SMILES: {result.canonical}"}
        ]
    }
