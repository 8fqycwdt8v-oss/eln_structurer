"""Validate a single SMILES string using RDKit (local; no network)."""

from __future__ import annotations

from typing import Any

from rdkit import Chem
from rdkit import RDLogger

from claude_agent_sdk import tool

RDLogger.DisableLog("rdApp.*")


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
    smiles = args.get("smiles", "")
    if not isinstance(smiles, str) or not smiles.strip():
        return {
            "content": [
                {"type": "text", "text": "ERROR: smiles must be a non-empty string."}
            ],
            "isError": True,
        }
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {
            "content": [
                {
                    "type": "text",
                    "text": f"INVALID: RDKit cannot parse SMILES {smiles!r}.",
                }
            ],
            "isError": True,
        }
    canonical = Chem.MolToSmiles(mol)
    return {
        "content": [
            {"type": "text", "text": f"VALID. Canonical SMILES: {canonical}"}
        ]
    }
