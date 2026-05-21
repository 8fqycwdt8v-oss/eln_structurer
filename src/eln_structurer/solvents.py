"""Local solvent reference table.

Maps common solvent NAME strings (lowercased) to their atmospheric boiling
points in °C. Used by:
- ORD-007 reflux-temperature sanity rule
- the expand_abbreviation tool (informational lookups)

Stays local — no network access. The table is intentionally small and biased
toward synthesis bench staples; extending it is purely additive.
"""

from __future__ import annotations


SOLVENT_BP_CELSIUS: dict[str, float] = {
    # Hydrocarbons
    "pentane": 36.0,
    "hexane": 69.0,
    "hexanes": 69.0,
    "heptane": 98.0,
    "cyclohexane": 81.0,
    "benzene": 80.0,
    "toluene": 111.0,
    "xylene": 138.0,
    # Halogenated
    "dichloromethane": 40.0,
    "dcm": 40.0,
    "ch2cl2": 40.0,
    "chloroform": 61.0,
    "chcl3": 61.0,
    "carbon tetrachloride": 77.0,
    "1,2-dichloroethane": 84.0,
    # Ethers
    "diethyl ether": 35.0,
    "ether": 35.0,
    "et2o": 35.0,
    "thf": 66.0,
    "tetrahydrofuran": 66.0,
    "2-methyltetrahydrofuran": 80.0,
    "1,4-dioxane": 101.0,
    "dioxane": 101.0,
    "dme": 85.0,
    "1,2-dimethoxyethane": 85.0,
    "mtbe": 55.0,
    "tert-butyl methyl ether": 55.0,
    # Alcohols
    "methanol": 65.0,
    "meoh": 65.0,
    "ethanol": 78.0,
    "etoh": 78.0,
    "isopropanol": 82.0,
    "ipa": 82.0,
    "2-propanol": 82.0,
    "n-butanol": 117.0,
    "t-butanol": 82.0,
    "tert-butanol": 82.0,
    # Nitrogen/sulfur-containing dipolar aprotic
    "dmf": 153.0,
    "dimethylformamide": 153.0,
    "dmac": 165.0,
    "dimethylacetamide": 165.0,
    "dmso": 189.0,
    "dimethyl sulfoxide": 189.0,
    "nmp": 202.0,
    "n-methylpyrrolidone": 202.0,
    "n-methyl-2-pyrrolidone": 202.0,
    "hmpa": 233.0,
    "acetonitrile": 82.0,
    "mecn": 82.0,
    "acn": 82.0,
    # Esters / ketones
    "ethyl acetate": 77.0,
    "etoac": 77.0,
    "acetone": 56.0,
    # Acids
    "acetic acid": 118.0,
    "formic acid": 100.0,
    # Aqueous
    "water": 100.0,
    "h2o": 100.0,
}


def lookup_solvent_bp(name: str) -> float | None:
    """Look up a solvent's atmospheric boiling point. Case-insensitive."""
    key = (name or "").strip().lower()
    return SOLVENT_BP_CELSIUS.get(key)
