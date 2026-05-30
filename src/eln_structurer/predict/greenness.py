"""Solvent greenness scoring (CHEM21 / GSK-style).

A condensed table derived from the CHEM21 Solvent Selection Guide
(RSC Green Chem. 2016) covering the solvents that appear most often
in pharmaceutical synthesis. Values are normalised to a 0..1 score
where 1.0 = greenest, 0.0 = strongly discouraged. Where CHEM21 doesn't
publish an entry we use a conservative default from the GSK Solvent
Selection Guide.

The scorer is deliberately small and hand-maintained — adding a new
solvent is a one-line change; the table never grows by ML.
"""

from __future__ import annotations

from eln_structurer.text_utils import normalize_compound_name


# CHEM21 categories collapsed onto a numeric score:
#   recommended (5)  → 1.0
#   problematic (3)  → 0.5
#   hazardous (2)    → 0.25
#   highly hazardous (1) → 0.0
_SOLVENT_GREENNESS: dict[str, float] = {
    # Recommended
    "water": 1.0,
    "ethanol": 1.0,
    "1-butanol": 1.0,
    "ethyl acetate": 1.0,
    "isopropyl acetate": 1.0,
    "methanol": 0.75,           # CHEM21 puts MeOH at "recommended but..."
    "n-butanol": 1.0,
    "isopropanol": 1.0,
    "ipa": 1.0,
    "acetic acid": 0.75,
    # Problematic
    "methyl tert-butyl ether": 0.5,
    "mtbe": 0.5,
    "tetrahydrofuran": 0.5,
    "thf": 0.5,
    "2-methyltetrahydrofuran": 0.75,    # GSK upgrades 2-MeTHF
    "acetonitrile": 0.5,
    "mecn": 0.5,
    "acetone": 0.75,
    "toluene": 0.5,
    "dmso": 0.5,
    "dimethyl sulfoxide": 0.5,
    "n,n-dimethylformamide": 0.25,
    "dmf": 0.25,
    "dimethylformamide": 0.25,
    "1,4-dioxane": 0.25,
    "dioxane": 0.25,
    "n-methylpyrrolidone": 0.25,
    "nmp": 0.25,
    # Hazardous
    "diethyl ether": 0.25,
    "et2o": 0.25,
    "ether": 0.25,
    "diisopropyl ether": 0.25,
    "hexane": 0.25,
    "hexanes": 0.25,
    "heptane": 0.5,
    "pentane": 0.25,
    "cyclohexane": 0.5,
    "xylene": 0.5,
    "dichloromethane": 0.25,
    "dcm": 0.25,
    "ch2cl2": 0.25,
    # Highly hazardous (strongly discouraged)
    "chloroform": 0.0,
    "chcl3": 0.0,
    "carbon tetrachloride": 0.0,
    "ccl4": 0.0,
    "1,2-dichloroethane": 0.0,
    "dce": 0.0,
    "benzene": 0.0,
    "1,2-dimethoxyethane": 0.25,
    "dme": 0.25,
    "pyridine": 0.25,
    "hmpa": 0.0,
    "hexamethylphosphoramide": 0.0,
}


def solvent_score(name: str) -> float:
    """0.0–1.0 greenness for a single named solvent.

    Unknown solvents return 0.5 (neutral) — we don't penalise
    obscurity, only known-bad chemistry.
    """
    key = normalize_compound_name(name)
    if not key:
        return 0.5
    return _SOLVENT_GREENNESS.get(key, 0.5)


def protocol_solvent_score(solvent_names: list[str]) -> float:
    """Aggregate score across all solvents in a protocol.

    Returns the minimum of each individual score — one bad solvent in a
    mixture drags the whole protocol down. Matches a common chemist's
    review heuristic ("if any of your solvents is chloroform you can't
    call your route green").
    """
    if not solvent_names:
        return 0.5
    return min(solvent_score(n) for n in solvent_names)


__all__ = ["solvent_score", "protocol_solvent_score"]
