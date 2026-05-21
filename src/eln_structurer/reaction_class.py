"""Heuristic reaction-class detection.

Calibrated against pharma medicinal-chemistry reaction distributions:
- Carey et al. 2006 (J. Med. Chem.) and Roughley & Jordan 2011 — amide
  bond formation alone is ~16% of all reactions; the top 12–15 classes
  cumulatively cover ~95%.
- Schneider et al. 2016 (J. Med. Chem.) confirms the same skew across
  decades of patent data.

The classifier returns ONE ``ReactionClass`` (the highest-priority match);
single-class is a deliberate simplification — a paragraph that contains
both, say, a Suzuki coupling and an in-situ Boc deprotection will be
classified as the higher-priority one and only that class's rules fire.
Multi-class support is a future refactor.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from eln_structurer.chemistry import name_of
from eln_structurer.schema import ReactionDraft


class ReactionClass(str, Enum):
    AMIDE_FORMATION = "AMIDE_FORMATION"
    BUCHWALD_HARTWIG = "BUCHWALD_HARTWIG"
    SUZUKI_COUPLING = "SUZUKI_COUPLING"
    REDUCTIVE_AMINATION = "REDUCTIVE_AMINATION"
    MITSUNOBU = "MITSUNOBU"
    WITTIG = "WITTIG"
    BOC_DEPROTECTION = "BOC_DEPROTECTION"
    GRIGNARD = "GRIGNARD"
    HALOGENATION = "HALOGENATION"
    OXIDATION = "OXIDATION"
    REDUCTION = "REDUCTION"
    N_ALKYLATION = "N_ALKYLATION"
    ESTERIFICATION = "ESTERIFICATION"
    UNKNOWN = "UNKNOWN"


# ----------------------------------------------------------------------
# Compiled detection patterns. Order of evaluation in classify_reaction()
# matters: more-specific patterns must run first so a Buchwald-Hartwig
# isn't misclassified as a generic SNAr, an STAB-mediated red-am isn't
# called a plain reduction, etc.
# ----------------------------------------------------------------------

# Amide coupling — peptide coupling reagents are unmistakable.
_AMIDE_COUPLING_REAGENT = re.compile(
    r"\b(edc|edci|edc[\.\-·]hcl|dcc|dic|hatu|hbtu|tbtu|"
    r"pybop|pyaop|pybrop|t3p|cdi|carbonyldiimidazole|"
    r"hobt|hoat|oxyma|propanephosphonic anhydride|"
    r"isobutyl chloroformate|ethyl chloroformate)\b"
)
_AMIDE_NAME_HINT = re.compile(
    r"\b(amide coupling|amidation|peptide coupling|amide formation)\b"
)

# Buchwald–Hartwig — specific ligands or named reaction.
_BUCHWALD_LIGAND = re.compile(
    r"\b(xantphos|brettphos|davephos|johnphos|t[-\s]?xphos|xphos|"
    r"sphos|ruphos|binap|cataCXium|ad[-\s]?gen|"
    r"pd[-\s]?(g1|g2|g3|g4|peppsi))\b"
)
_BUCHWALD_NAME = re.compile(r"\b(buchwald|hartwig|c[-\s]?n coupling)\b")

# Suzuki — Pd + boronic acid (kept from prior round).
_PD_CATALYST = re.compile(
    r"\b(pd\(pph3\)4|pd2\(dba\)3|pdcl2|palladium|pd/c|pd\(oac\)2|"
    r"pd\(ppf\)|pd\(dba\)|pd\s*xphos|pd\s*sphos)\b"
)
_BORONIC_ACID = re.compile(
    r"boronic\s+acid|boronate\b|\bb\(oh\)2\b|"
    r"\b(pinacol\s+ester|bpin)\b"
)
_ARYL_HALIDE = re.compile(r"\b\d?-?(bromo|iodo|chloro)|\b(arx|aryl[-\s]?halide)")

# Reductive amination — STAB / NaBH3CN are red-am-specific; NaBH4 alone
# is ambiguous (plain reduction). Plus the explicit name.
_RED_AM_REDUCTANT = re.compile(
    r"\b(nabh\(oac\)3|sodium triacetoxyborohydride|stab|"
    r"nabh3cn|sodium cyanoborohydride|"
    r"picoline borane|borane[- ]pyridine|2[-\s]picolineborane)\b"
)
_RED_AM_NAME = re.compile(r"\breductive amination\b")

# Mitsunobu — DIAD / DEAD + PPh3 (in either order).
_MITSUNOBU_NAME = re.compile(r"\bmitsunobu\b")
_DIAD = re.compile(r"\b(diad|dead|diethyl azodicarboxylate|diisopropyl azodicarboxylate)\b")
_PPH3 = re.compile(r"\b(pph3|triphenylphosphine|ph3p)\b")

# Wittig / HWE.
_WITTIG_NAME = re.compile(
    r"\b(wittig|horner[-\s]?wadsworth[-\s]?emmons|hwe|julia|julia[-\s]?kocienski)\b"
)
_PHOSPHONIUM_YLIDE = re.compile(r"\b(ph3p=|ylide|phosphonium|phosphonate)\b")

# Boc / Cbz / Fmoc deprotection. "deprotection" is also a strong hint.
_PROTECTED_GROUP = re.compile(
    r"\b(boc|tert[-\s]?butoxycarbonyl|cbz|benzyloxycarbonyl|fmoc|"
    r"fluorenylmethyloxycarbonyl)\b"
)
_DEPROTECTION_REAGENT = re.compile(
    r"\b(tfa|trifluoroacetic acid|hcl[/\s]+(dioxane|etoac|ether|meoh|water)|"
    r"piperidine|h2[/\s]+(pd|pd/c)|"
    r"4m\s+hcl)\b"
)
_DEPROTECTION_NAME = re.compile(r"\bdeprotection\b")

# Grignard — already present, kept.
_MG = re.compile(r"\bmagnesium\b|\bmg\s*turnings?\b")

# Halogenation — named succinimide-class or fluorinating agents.
_HALOGENATION_REAGENT = re.compile(
    r"\b(nbs|n[-\s]?bromosuccinimide|ncs|n[-\s]?chlorosuccinimide|"
    r"nis|n[-\s]?iodosuccinimide|"
    r"selectfluor|deoxo[-\s]?fluor|dast|nfsi|n[-\s]?fluoro[-\s]?benzenesulfonimide|"
    r"sulfuryl chloride|so2cl2|"
    r"phosphorus tribromide|pbr3)\b"
)
_HALOGENATION_NAME = re.compile(r"\b(halogenation|bromination|chlorination|iodination|fluorination)\b")

# Oxidation — named oxidants.
_OXIDANT = re.compile(
    r"\b(dess[-\s]?martin|dmp|swern|moffatt|pcc|pyridinium chlorochromate|"
    r"pdc|jones reagent|kmno4|potassium permanganate|"
    r"mno2|manganese dioxide|tempo|baib|"
    r"oxone|m[-\s]?cpba|mcpba|3[-\s]?chloroperbenzoic|"
    r"hydrogen peroxide|h2o2|nai04|naio4|periodate)\b"
)
_OXIDATION_NAME = re.compile(r"\boxidation\b|\boxidative\b")

# Reduction — broad reductants. Plain reduction only if not red-am.
_REDUCTANT = re.compile(
    r"\b(nabh4|sodium borohydride|lialh4|lithium aluminum hydride|"
    r"dibal|diisobutylaluminum hydride|"
    r"raney nickel|h2[/\s]+(pd|pd/c|pt|raney)|hydrogen.*palladium|"
    r"l[-\s]?selectride|red[-\s]?al)\b"
)
_REDUCTION_NAME = re.compile(r"\b(reduction|hydrogenation)\b")

# N-alkylation — generic; relies on amine + alkyl halide + base text patterns.
_N_ALKYLATION_NAME = re.compile(
    r"\bn[-\s]?alkylation\b|\bn[-\s]?methylation\b|\balkylation of\b.*amine"
)

# Esterification — acid + alcohol + dehydrating agent (kept from before).
_CARBOXYLIC_ACID_NAME = re.compile(r"\b\w+ic\s+acid\b")
_ALCOHOL_NAME = re.compile(r"\b\w+ol\b|\b\w+alcohol\b")
_ESTER_DEHYDRATING_AGENT = re.compile(
    r"\b(dcc|edc|hatu|h2so4|sulfuric acid|p[-\s]?tsoh|tosic acid|"
    r"trifluoroacetic anhydride|tfaa|acetic anhydride)\b"
)


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
    """Heuristic classifier. Returns the highest-priority class match or UNKNOWN.

    Priority is encoded by the order of checks below. More-specific patterns
    run first so e.g. a Buchwald–Hartwig (specific ligand) wins over a
    generic Pd reaction.
    """
    names = _all_input_names(draft)
    joined = " | ".join(names)

    # 1. Amide coupling — very specific reagent fingerprint.
    if _AMIDE_COUPLING_REAGENT.search(joined) or _AMIDE_NAME_HINT.search(joined):
        return ClassificationResult(
            ReactionClass.AMIDE_FORMATION,
            0.9,
            "amide coupling reagent (EDC/HATU/DCC/PyBOP/T3P/CDI) detected.",
        )

    # 2. Buchwald–Hartwig — distinguished from Suzuki by ligand or amine
    #    nucleophile pattern (Pd + named ligand).
    if _BUCHWALD_NAME.search(joined) or (
        _PD_CATALYST.search(joined) and _BUCHWALD_LIGAND.search(joined)
    ):
        return ClassificationResult(
            ReactionClass.BUCHWALD_HARTWIG,
            0.85,
            "Pd catalyst + Buchwald-type ligand (XPhos/SPhos/RuPhos/Xantphos/BINAP).",
        )

    # 3. Suzuki — Pd + boronic acid + halide.
    if _PD_CATALYST.search(joined) and _BORONIC_ACID.search(joined) and _ARYL_HALIDE.search(joined):
        return ClassificationResult(
            ReactionClass.SUZUKI_COUPLING,
            0.9,
            "Pd catalyst + boronic acid + aryl halide all present.",
        )

    # 4. Reductive amination — STAB/NaBH3CN/picoline borane specifically,
    #    or the literal name.
    if _RED_AM_REDUCTANT.search(joined) or _RED_AM_NAME.search(joined):
        return ClassificationResult(
            ReactionClass.REDUCTIVE_AMINATION,
            0.85,
            "reductive-amination–specific reductant (STAB/NaBH3CN/picoline borane).",
        )

    # 5. Mitsunobu — needs both DIAD/DEAD and PPh3 (or the literal name).
    if _MITSUNOBU_NAME.search(joined) or (
        _DIAD.search(joined) and _PPH3.search(joined)
    ):
        return ClassificationResult(
            ReactionClass.MITSUNOBU,
            0.95,
            "DIAD/DEAD + PPh3 = Mitsunobu signature.",
        )

    # 6. Wittig / HWE.
    if _WITTIG_NAME.search(joined) or _PHOSPHONIUM_YLIDE.search(joined):
        return ClassificationResult(
            ReactionClass.WITTIG,
            0.8,
            "Wittig/HWE phosphorus ylide or phosphonate detected.",
        )

    # 7. Boc / Cbz / Fmoc deprotection.
    if (_PROTECTED_GROUP.search(joined) and _DEPROTECTION_REAGENT.search(joined)) or \
       _DEPROTECTION_NAME.search(joined):
        return ClassificationResult(
            ReactionClass.BOC_DEPROTECTION,
            0.85,
            "Boc/Cbz/Fmoc protected substrate + acid/H2 deprotection reagent.",
        )

    # 8. Grignard — already present.
    if _MG.search(joined) and _ARYL_HALIDE.search(joined):
        return ClassificationResult(
            ReactionClass.GRIGNARD,
            0.85,
            "magnesium + alkyl/aryl halide indicates Grignard formation.",
        )

    # 9. Halogenation — named reagents.
    if _HALOGENATION_REAGENT.search(joined) or _HALOGENATION_NAME.search(joined):
        return ClassificationResult(
            ReactionClass.HALOGENATION,
            0.85,
            "halogenating reagent (NBS/NCS/Selectfluor/DAST) detected.",
        )

    # 10. Oxidation — named oxidants.
    if _OXIDANT.search(joined) or _OXIDATION_NAME.search(joined):
        return ClassificationResult(
            ReactionClass.OXIDATION,
            0.8,
            "oxidant (DMP/Swern/PCC/TEMPO/mCPBA/KMnO4) detected.",
        )

    # 11. Reduction — generic.
    if _REDUCTANT.search(joined) or _REDUCTION_NAME.search(joined):
        return ClassificationResult(
            ReactionClass.REDUCTION,
            0.7,
            "general reductant (NaBH4/LAH/DIBAL/H2-cat) detected.",
        )

    # 12. N-alkylation — named hint.
    if _N_ALKYLATION_NAME.search(joined):
        return ClassificationResult(
            ReactionClass.N_ALKYLATION,
            0.65,
            "N-alkylation named in inputs.",
        )

    # 13. Esterification — acid + alcohol + dehydrating agent.
    if (
        _CARBOXYLIC_ACID_NAME.search(joined)
        and _ALCOHOL_NAME.search(joined)
        and _ESTER_DEHYDRATING_AGENT.search(joined)
    ):
        return ClassificationResult(
            ReactionClass.ESTERIFICATION,
            0.65,
            "carboxylic acid + alcohol + dehydration/coupling agent.",
        )

    return ClassificationResult.unknown()
