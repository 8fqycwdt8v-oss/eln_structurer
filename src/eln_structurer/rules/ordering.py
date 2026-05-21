"""Ordering rules (ORD-*) — temporal sanity of additions and workups."""

from __future__ import annotations

import re

from eln_structurer.rules.base import Rule, RuleViolation, Severity
from eln_structurer.schema import ReactionDraft


_QUENCH_PATTERNS = [
    r"\bquench(ed|ing)?\b",
    r"saturated\s+NH4Cl",
    r"sat\.?\s*NH4Cl",
    r"sat\.?\s*NaHCO3",
    r"saturated\s+NaHCO3",
]


# Common workup reagents that often appear in a workup `description` but get
# forgotten as Compound entries in `workup.components`. Matching is
# substring/case-insensitive on the description text, after a light cleanup.
_WORKUP_KEYWORDS = {
    "brine": "brine",
    "celite": "Celite",
    "na2so4": "Na2SO4",
    "magnesium sulfate": "MgSO4",
    "mgso4": "MgSO4",
    "sodium sulfate": "Na2SO4",
    "k2co3": "K2CO3",
    "nahco3": "NaHCO3",
    "nh4cl": "NH4Cl",
    "water": "water",
    "ethyl acetate": "ethyl acetate",
    "etoac": "ethyl acetate",
    "hexanes": "hexanes",
    "hexane": "hexanes",
    "dichloromethane": "DCM",
    "dcm": "DCM",
    "ether": "diethyl ether",
    "diethyl ether": "diethyl ether",
    "et2o": "diethyl ether",
    "methanol": "methanol",
    "meoh": "methanol",
}


def _declared_compound_names(draft: ReactionDraft) -> set[str]:
    """All lowercased NAME/IUPAC_NAME identifiers across inputs and workups."""
    names: set[str] = set()
    for inp in draft.inputs:
        for comp in inp.components:
            for ident in comp.identifiers:
                if ident.type in {"NAME", "IUPAC_NAME"}:
                    names.add(ident.value.strip().lower())
    for wu in draft.workups:
        for comp in wu.components:
            for ident in comp.identifiers:
                if ident.type in {"NAME", "IUPAC_NAME"}:
                    names.add(ident.value.strip().lower())
    return names


class WorkupKeywordsDeclared(Rule):
    """ORD-001: workup descriptions mentioning a known workup reagent must
    have that reagent declared as a Compound somewhere in the draft.

    This catches the common LLM mistake of describing a wash in prose
    ("washed with brine") without ever attaching brine as a structured
    component.
    """

    id = "ORD-001"
    description = (
        "Workup description text must not reference a known workup reagent "
        "(brine, Celite, Na2SO4, etc.) that is not declared as a Compound "
        "anywhere in the draft."
    )

    def check(self, draft: ReactionDraft) -> list[RuleViolation]:
        declared = _declared_compound_names(draft)
        violations: list[RuleViolation] = []
        for idx, wu in enumerate(draft.workups):
            text = (wu.description or "").lower()
            for keyword, canonical in _WORKUP_KEYWORDS.items():
                if keyword not in text:
                    continue
                # Cheap match — already declared under any spelling we know?
                if canonical.lower() in declared or keyword in declared:
                    continue
                violations.append(
                    RuleViolation(
                        rule_id=self.id,
                        severity=Severity.WARNING,
                        message=(
                            f"Workup #{idx} description mentions {canonical!r} but "
                            "no input or workup.components entry declares it."
                        ),
                        fix_hint=(
                            f"Add {canonical!r} to workups[{idx}].components with "
                            "reaction_role='WORKUP', so the structured record "
                            "matches the prose."
                        ),
                        path=f"workups[{idx}].description",
                    )
                )
        return violations


# Active heating control types — anything where energy is intentionally
# supplied to the vessel. AMBIENT and ICE_BATH / LIQUID_NITROGEN are cooling
# or passive; UNSPECIFIED means "we don't know".
_HEATING_CONTROL_TYPES = {"OIL_BATH", "WATER_BATH", "HEATER", "REFLUX"}


class SolventPresentBeforeHeating(Rule):
    id = "ORD-002"
    description = (
        "Heated reactions (control_type in {OIL_BATH, WATER_BATH, HEATER, "
        "REFLUX}) must declare a SOLVENT input."
    )

    def check(self, draft: ReactionDraft) -> list[RuleViolation]:
        temp = draft.conditions.temperature
        if temp is None:
            return []
        # Heating intent is determined by the *control_type*, not a numeric
        # cutoff — that way neat/melt reactions intentionally above ambient
        # don't trigger the rule.
        if temp.control_type not in _HEATING_CONTROL_TYPES:
            return []
        for inp in draft.inputs:
            for comp in inp.components:
                if comp.reaction_role == "SOLVENT":
                    return []
        return [
            RuleViolation(
                rule_id=self.id,
                severity=Severity.ERROR,
                message=(
                    f"Reaction uses heating control_type={temp.control_type} "
                    "but no input is marked as SOLVENT."
                ),
                fix_hint=(
                    "Identify the solvent from the paragraph (DMF, THF, toluene, "
                    "water, etc.) and add it as an input with "
                    "reaction_role='SOLVENT'. For neat/melt/solid-state reactions "
                    "without a solvent, leave control_type='UNSPECIFIED' so this "
                    "rule does not fire."
                ),
                path="inputs[*].components[*].reaction_role",
            )
        ]


# Control types where stirring is the norm. REFLUX without stirring is
# unusual but valid (simple bp distillation); microwave reactions don't
# need stirring; photochemistry usually requires only gentle agitation.
_REQUIRES_STIRRING_CONTROL_TYPES = {"OIL_BATH", "WATER_BATH", "HEATER"}


class StirringBeforeHeating(Rule):
    id = "ORD-003"
    description = (
        "Heated reactions in conventional vessels (oil bath, water bath, heater) "
        "should declare stirring; reflux and microwave reactions are allowed "
        "to opt out."
    )

    def check(self, draft: ReactionDraft) -> list[RuleViolation]:
        temp = draft.conditions.temperature
        if temp is None or temp.control_type not in _REQUIRES_STIRRING_CONTROL_TYPES:
            return []
        stir = draft.conditions.stirring
        if stir is None or stir.type in {"NONE", "UNSPECIFIED"}:
            return [
                RuleViolation(
                    rule_id=self.id,
                    severity=Severity.WARNING,
                    message=(
                        f"Heated reaction (control_type={temp.control_type}) "
                        f"with stirring={stir.type if stir else 'missing'}."
                    ),
                    fix_hint=(
                        "Almost all heated reactions in conventional vessels are "
                        "stirred. Set stirring.type='MAGNETIC' (or 'OVERHEAD' for "
                        "large scale) unless the paragraph explicitly says "
                        "otherwise."
                    ),
                    path="conditions.stirring",
                )
            ]
        return []


class QuenchAfterReaction(Rule):
    id = "ORD-004"
    description = "Quench-like workups must come after the main reaction step."

    def check(self, draft: ReactionDraft) -> list[RuleViolation]:
        violations: list[RuleViolation] = []
        for idx, wu in enumerate(draft.workups):
            text = (wu.description or "").lower()
            looks_like_quench = any(re.search(p, text, re.IGNORECASE) for p in _QUENCH_PATTERNS)
            if looks_like_quench and wu.order < 1:
                violations.append(
                    RuleViolation(
                        rule_id=self.id,
                        severity=Severity.ERROR,
                        message=(
                            f"Workup #{idx} looks like a quench but its order "
                            f"({wu.order}) is < 1."
                        ),
                        fix_hint=(
                            "Quench steps are post-reaction; set the workup "
                            "order to a positive integer reflecting its place "
                            "in the workup sequence."
                        ),
                        path=f"workups[{idx}].order",
                    )
                )
        return violations


# Reagents that decompose or ignite on contact with O2 or moisture. Match
# is substring/case-insensitive on the NAME identifier. The list is
# intentionally narrow — we want zero false positives.
_AIR_SENSITIVE_NAME_PATTERNS = [
    "grignard",
    "n-buli", "nbuli", "n-butyllithium", "butyllithium",
    "s-buli", "sec-buli", "sec-butyllithium",
    "t-buli", "tert-buli", "tert-butyllithium",
    "phenyllithium", "phli",
    "methyllithium", "meli",
    "lda", "lithium diisopropylamide",
    "lihmds", "lithium hexamethyldisilazide",
    "nahmds", "sodium hexamethyldisilazide",
    "khmds", "potassium hexamethyldisilazide",
    "sodium hydride", "nah",
    "potassium hydride", "kh",
    "diethylzinc",
    "trimethylaluminum", "triethylaluminum",
    "dibal", "diisobutylaluminum hydride",
]

_INERT_ATMOSPHERES = {"nitrogen", "argon", "n2", "ar", "inert"}


def _looks_air_sensitive(name: str) -> str | None:
    """Return the matched pattern if ``name`` suggests air sensitivity."""
    n = name.strip().lower()
    for pat in _AIR_SENSITIVE_NAME_PATTERNS:
        if pat in n:
            return pat
    return None


def _looks_like_grignard_setup(draft: ReactionDraft) -> bool:
    """Heuristic: an input named 'magnesium' (turnings, dust, etc.) plus an
    alkyl/aryl halide suggests an in-situ Grignard formation."""
    has_mg = False
    has_halide = False
    for inp in draft.inputs:
        for comp in inp.components:
            for ident in comp.identifiers:
                if ident.type not in {"NAME", "IUPAC_NAME"}:
                    continue
                lower = ident.value.lower()
                if "magnesium" in lower:
                    has_mg = True
                if "bromo" in lower or "iodo" in lower or "chloro" in lower:
                    has_halide = True
    return has_mg and has_halide


class InertAtmosphereForSensitiveReagents(Rule):
    """ORD-006: air- and moisture-sensitive reagents require inert atmosphere.

    Fires when an input compound NAME matches a known sensitive reagent
    (Grignards, organolithiums, hydride bases, dialkylzincs, ...) OR the
    input set looks like an in-situ Grignard formation (Mg + alkyl halide),
    AND ``conditions.atmosphere`` is missing or not inert.
    """

    id = "ORD-006"
    description = (
        "Air/moisture-sensitive reagents (organolithiums, Grignards, NaH, "
        "LDA, …) require conditions.atmosphere set to nitrogen or argon."
    )

    def check(self, draft: ReactionDraft) -> list[RuleViolation]:
        sensitive_hits: list[tuple[int, int, str]] = []
        for i, inp in enumerate(draft.inputs):
            for j, comp in enumerate(inp.components):
                for ident in comp.identifiers:
                    if ident.type not in {"NAME", "IUPAC_NAME"}:
                        continue
                    pat = _looks_air_sensitive(ident.value)
                    if pat is not None:
                        sensitive_hits.append((i, j, pat))
                        break

        if not sensitive_hits and not _looks_like_grignard_setup(draft):
            return []

        atmosphere = (draft.conditions.atmosphere or "").strip().lower()
        if atmosphere in _INERT_ATMOSPHERES:
            return []

        if sensitive_hits:
            example = sensitive_hits[0]
            msg = (
                f"Input[{example[0]}].components[{example[1]}] is "
                f"{example[2]!r}, which is air/moisture sensitive, but "
                "conditions.atmosphere is "
                f"{draft.conditions.atmosphere!r}."
            )
        else:
            msg = (
                "Input set looks like an in-situ Grignard formation "
                "(magnesium + alkyl/aryl halide) but conditions.atmosphere "
                f"is {draft.conditions.atmosphere!r}."
            )

        return [
            RuleViolation(
                rule_id=self.id,
                severity=Severity.ERROR,
                message=msg,
                fix_hint=(
                    "Set conditions.atmosphere to 'nitrogen' or 'argon' "
                    "(the paragraph almost certainly says so — look for "
                    "'under N2', 'under argon', 'flame-dried', or 'inert')."
                ),
                path="conditions.atmosphere",
            )
        ]


class WorkupOrderMonotonic(Rule):
    id = "ORD-005"
    description = "Workup order field should be monotonically increasing."

    def check(self, draft: ReactionDraft) -> list[RuleViolation]:
        prev = 0
        violations: list[RuleViolation] = []
        for idx, wu in enumerate(draft.workups):
            if wu.order < prev:
                violations.append(
                    RuleViolation(
                        rule_id=self.id,
                        severity=Severity.WARNING,
                        message=(
                            f"Workup #{idx} has order={wu.order}, which is less "
                            f"than the previous order ({prev})."
                        ),
                        fix_hint=(
                            "Renumber workup.order so that the steps follow the "
                            "narrative sequence of the paragraph (1, 2, 3, ...)."
                        ),
                        path=f"workups[{idx}].order",
                    )
                )
            prev = max(prev, wu.order)
        return violations


ORD_RULES: list[Rule] = [
    WorkupKeywordsDeclared(),
    SolventPresentBeforeHeating(),
    StirringBeforeHeating(),
    QuenchAfterReaction(),
    WorkupOrderMonotonic(),
    InertAtmosphereForSensitiveReagents(),
]
