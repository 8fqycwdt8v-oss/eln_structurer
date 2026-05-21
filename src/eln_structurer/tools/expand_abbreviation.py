"""expand_abbreviation — local chemistry-abbreviation lookup tool.

The agent uses this for ambiguous tokens (e.g. "dr", "ee", "o.n.") instead
of guessing. Pure dictionary lookup — no network, no fuzzy matching.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from claude_agent_sdk import tool

from eln_structurer.solvents import lookup_solvent_bp


_ABBREVIATIONS: dict[str, str] = {
    "rt": "room temperature (≈ 20–25 °C, AMBIENT)",
    "r.t.": "room temperature (≈ 20–25 °C, AMBIENT)",
    "o.n.": "overnight (≈ 12–16 hours)",
    "o/n": "overnight (≈ 12–16 hours)",
    "aq.": "aqueous",
    "sat.": "saturated",
    "satd.": "saturated",
    "conc.": "concentrated",
    "anhyd.": "anhydrous",
    "abs.": "absolute (e.g. abs. ethanol = anhydrous ethanol)",
    "dr": "diastereomeric ratio (use ProductMeasurement.SELECTIVITY)",
    "ee": "enantiomeric excess (use ProductMeasurement.SELECTIVITY)",
    "de": "diastereomeric excess (use ProductMeasurement.SELECTIVITY)",
    "tlc": "thin-layer chromatography",
    "nmr": "nuclear magnetic resonance",
    "hplc": "high-performance liquid chromatography",
    "mw": "microwave irradiation",
    "uv": "ultraviolet light",
    "etoac": "ethyl acetate (SMILES CCOC(C)=O)",
    "thf": "tetrahydrofuran (SMILES C1CCOC1, bp 66 °C)",
    "dmf": "N,N-dimethylformamide (SMILES CN(C)C=O, bp 153 °C)",
    "dmso": "dimethyl sulfoxide (SMILES CS(=O)C, bp 189 °C)",
    "dcm": "dichloromethane (SMILES ClCCl, bp 40 °C)",
    "meoh": "methanol (SMILES CO, bp 65 °C)",
    "etoh": "ethanol (SMILES CCO, bp 78 °C)",
    "ipa": "isopropanol (SMILES CC(C)O, bp 82 °C)",
    "mecn": "acetonitrile (SMILES CC#N, bp 82 °C)",
    "et2o": "diethyl ether (SMILES CCOCC, bp 35 °C)",
    "mtbe": "tert-butyl methyl ether (SMILES CC(C)(C)OC, bp 55 °C)",
    "ndp": "non-degassed (atmosphere not specified)",
    "tba": "tert-butyl alcohol",
    "h2o": "water (SMILES O, bp 100 °C)",
    "n2": "nitrogen atmosphere",
    "ar": "argon atmosphere",
}


@dataclass(frozen=True)
class AbbreviationLookup:
    token: str
    expansion: str | None
    bp_celsius: float | None  # populated if the token also names a solvent


def lookup_abbreviation(token: str) -> AbbreviationLookup:
    """Return the canonical expansion of ``token`` (case-insensitive).

    Also returns a boiling point when the token is recognized as a solvent
    name (via ``solvents.lookup_solvent_bp``) — helpful when the agent is
    reconciling reflux temperature with the chosen solvent.
    """
    key = (token or "").strip().lower()
    expansion = _ABBREVIATIONS.get(key)
    bp = lookup_solvent_bp(key)
    return AbbreviationLookup(token=token, expansion=expansion, bp_celsius=bp)


@tool(
    "expand_abbreviation",
    (
        "Look up a chemistry abbreviation or common solvent name in the local "
        "reference table. Returns the canonical expansion and (when applicable) "
        "the atmospheric boiling point. Use this whenever the paragraph contains "
        "a short form whose meaning could be ambiguous."
    ),
    {"token": str},
)
async def expand_abbreviation(args: dict[str, Any]) -> dict[str, Any]:
    token = args.get("token", "")
    if not isinstance(token, str) or not token.strip():
        return {
            "content": [
                {"type": "text", "text": "ERROR: token must be a non-empty string."}
            ],
            "isError": True,
        }
    result = lookup_abbreviation(token)
    if result.expansion is None and result.bp_celsius is None:
        return {
            "content": [
                {
                    "type": "text",
                    "text": (
                        f"UNKNOWN: no entry for {token!r}. If it's a real "
                        "abbreviation, use your prior knowledge; do not invent."
                    ),
                }
            ]
        }
    lines = []
    if result.expansion:
        lines.append(f"{token} = {result.expansion}")
    if result.bp_celsius is not None:
        lines.append(f"boiling point (atmospheric): {result.bp_celsius} °C")
    return {"content": [{"type": "text", "text": "\n".join(lines)}]}
