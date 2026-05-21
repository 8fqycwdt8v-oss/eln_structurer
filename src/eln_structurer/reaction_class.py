"""Heuristic reaction-class detection.

Looks at the compound NAME identifiers and assigns one of a small set of
canonical reaction classes. The classifier is intentionally conservative —
when in doubt it returns ``UNKNOWN`` so the class-specific rules stay quiet.

Used by ``rules/class_specific.py`` to dispatch to focused checks (e.g. a
Suzuki coupling must have a Pd catalyst plus a boronic acid; a Grignard
formation must have Mg plus an alkyl/aryl halide).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from eln_structurer.chemistry import name_of
from eln_structurer.schema import ReactionDraft


class ReactionClass(str, Enum):
    SUZUKI_COUPLING = "SUZUKI_COUPLING"
    GRIGNARD = "GRIGNARD"
    REDUCTION = "REDUCTION"
    ESTERIFICATION = "ESTERIFICATION"
    UNKNOWN = "UNKNOWN"


# Compiled patterns matched against the lowercased compound NAME strings.
_PD_CATALYSTS = re.compile(
    r"\b(pd\(pph3\)4|pd2\(dba\)3|pdcl2|palladium|pd/c|pd\(oac\)2)\b"
)
# "boronic acid" intentionally lacks a leading \b — most reagent names are
# fused (phenylboronic, 4-tolylboronic, …) so a word boundary would miss
# them. The substring "boronic acid" is unambiguous enough to be safe.
_BORONIC_ACID = re.compile(r"boronic\s+acid|boronate\b|\bb\(oh\)2\b")
_ARYL_HALIDE = re.compile(r"\b\d?-?(bromo|iodo|chloro)")
_MG = re.compile(r"\bmagnesium\b|\bmg\s*turnings?\b")
_REDUCING_AGENTS = re.compile(
    r"\b(nabh4|sodium borohydride|lialh4|lithium aluminum hydride|"
    r"dibal|diisobutylaluminum hydride|nabh\(oac\)3|na\(bh\)4|"
    r"raney nickel|h2/pd|hydrogen.*palladium)\b"
)
_CARBOXYLIC_ACID_NAME = re.compile(r"\b(\w+ic acid|carboxylic acid)\b")
_ALCOHOL_NAME = re.compile(r"\b(\w*ol|\w*alcohol)\b")
_DEHYDRATION_AGENT = re.compile(r"\b(dcc|edc|hatu|h2so4|sulfuric acid|p-tsoh)\b")


@dataclass(frozen=True)
class ClassificationResult:
    cls: ReactionClass
    confidence: float
    rationale: str

    @classmethod
    def unknown(cls) -> "ClassificationResult":
        return cls(ReactionClass.UNKNOWN, 0.0, "no patterns matched")


def _all_input_names(draft: ReactionDraft) -> list[str]:
    out: list[str] = []
    for inp in draft.inputs:
        for comp in inp.components:
            n = name_of(comp)
            if n:
                out.append(n.lower())
    return out


def classify_reaction(draft: ReactionDraft) -> ClassificationResult:
    """Heuristic classifier. Returns the most likely class or UNKNOWN."""
    names = _all_input_names(draft)
    joined = " | ".join(names)

    has_pd = bool(_PD_CATALYSTS.search(joined))
    has_boronic = bool(_BORONIC_ACID.search(joined))
    has_halide = bool(_ARYL_HALIDE.search(joined))
    has_mg = bool(_MG.search(joined))
    has_reductant = bool(_REDUCING_AGENTS.search(joined))
    has_acid = bool(_CARBOXYLIC_ACID_NAME.search(joined))
    has_alcohol = bool(_ALCOHOL_NAME.search(joined))
    has_dehydration = bool(_DEHYDRATION_AGENT.search(joined))

    # Order matters — more-specific patterns first.
    if has_pd and has_boronic and has_halide:
        return ClassificationResult(
            ReactionClass.SUZUKI_COUPLING,
            0.9,
            "Pd catalyst + boronic acid + aryl halide all present.",
        )
    if has_mg and has_halide:
        return ClassificationResult(
            ReactionClass.GRIGNARD,
            0.85,
            "Magnesium + alkyl/aryl halide indicates Grignard formation.",
        )
    if has_reductant:
        return ClassificationResult(
            ReactionClass.REDUCTION,
            0.7,
            "Known reducing agent present in inputs.",
        )
    if has_acid and has_alcohol and has_dehydration:
        return ClassificationResult(
            ReactionClass.ESTERIFICATION,
            0.65,
            "Carboxylic acid + alcohol + dehydration/coupling agent.",
        )
    return ClassificationResult.unknown()
