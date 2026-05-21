"""Class-specific rules (CLS-*).

These only fire when the heuristic reaction classifier identifies a
specific reaction class. They cover the "obvious-when-named" requirements
that the general rules can't enforce (a Suzuki needs a base; a Grignard
needs Mg; a reduction needs a reducing agent).

The classifier is best-effort: when it returns UNKNOWN, none of these rules
fire and we fall back on the general rule pack alone.
"""

from __future__ import annotations

import re

from eln_structurer.chemistry import name_of
from eln_structurer.reaction_class import ReactionClass, classify_reaction
from eln_structurer.rules.base import Rule, RuleViolation, Severity
from eln_structurer.schema import ReactionDraft


def _has_role(draft: ReactionDraft, role: str) -> bool:
    return any(
        comp.reaction_role == role
        for inp in draft.inputs
        for comp in inp.components
    )


def _name_matches(draft: ReactionDraft, pattern: re.Pattern) -> bool:
    for inp in draft.inputs:
        for comp in inp.components:
            n = name_of(comp)
            if n and pattern.search(n.lower()):
                return True
    return False


class SuzukiRequiredComponents(Rule):
    """CLS-001: Suzuki coupling needs Pd catalyst, base, boronic acid,
    and aryl halide. The classifier already detected Pd + boronic + halide
    by names; this rule additionally asks for an explicit base (REAGENT)."""

    id = "CLS-001"
    description = (
        "Suzuki coupling requires a Pd catalyst, a boronic acid, an aryl "
        "halide, and a base (typically K2CO3, Cs2CO3, K3PO4, or NaOtBu)."
    )

    _BASE_PATTERN = re.compile(
        r"\b(k2co3|cs2co3|k3po4|na2co3|naoh|koh|naotbu|kotbu|k2hpo4|"
        r"triethylamine|tea|diisopropylethylamine|dipea|hunig)\b"
    )

    def check(self, draft: ReactionDraft) -> list[RuleViolation]:
        if classify_reaction(draft).cls != ReactionClass.SUZUKI_COUPLING:
            return []
        if _name_matches(draft, self._BASE_PATTERN):
            return []
        return [
            RuleViolation(
                rule_id=self.id,
                severity=Severity.ERROR,
                message=(
                    "Detected a Suzuki coupling (Pd + boronic acid + aryl "
                    "halide) but no base is declared in the inputs."
                ),
                fix_hint=(
                    "Suzuki couplings require a base (e.g. K2CO3, Cs2CO3, "
                    "K3PO4). Check the paragraph and add it as an input "
                    "with reaction_role='REAGENT'."
                ),
                path="inputs[*].components[*]",
            )
        ]


class GrignardRequiresInertAtmosphere(Rule):
    """CLS-002: Grignard formation must run under nitrogen or argon. ORD-006
    already enforces this for any input named ``magnesium`` + halide, so
    this rule is mostly a documented duplicate kept here for class-based
    reporting. It strengthens the error message with the reaction-class
    context."""

    id = "CLS-002"
    description = (
        "Grignard reagent formation requires conditions.atmosphere set "
        "to 'nitrogen' or 'argon'."
    )

    _INERT = {"nitrogen", "argon", "n2", "ar", "inert"}

    def check(self, draft: ReactionDraft) -> list[RuleViolation]:
        if classify_reaction(draft).cls != ReactionClass.GRIGNARD:
            return []
        atmo = (draft.conditions.atmosphere or "").strip().lower()
        if atmo in self._INERT:
            return []
        return [
            RuleViolation(
                rule_id=self.id,
                severity=Severity.ERROR,
                message=(
                    "Detected a Grignard formation (Mg + alkyl/aryl halide) "
                    f"but conditions.atmosphere is {draft.conditions.atmosphere!r}. "
                    "Grignards ignite in air."
                ),
                fix_hint=(
                    "Set conditions.atmosphere to 'nitrogen' or 'argon'. The "
                    "paragraph almost always says so."
                ),
                path="conditions.atmosphere",
            )
        ]


class ReductionNeedsReducingAgent(Rule):
    """CLS-003: a reaction classified as REDUCTION must declare the reducing
    agent as a non-SOLVENT, non-PRODUCT input. The classifier itself uses
    name matching; this rule ensures the matched compound is actually
    structurally present in the inputs with an appropriate role."""

    id = "CLS-003"
    description = "A REDUCTION must include the reducing agent as a REACTANT or REAGENT."

    _REDUCTANT_PATTERN = re.compile(
        r"\b(nabh4|sodium borohydride|lialh4|lithium aluminum hydride|"
        r"dibal|diisobutylaluminum hydride|nabh\(oac\)3|raney nickel|h2/pd)\b"
    )

    def check(self, draft: ReactionDraft) -> list[RuleViolation]:
        if classify_reaction(draft).cls != ReactionClass.REDUCTION:
            return []
        for inp in draft.inputs:
            for comp in inp.components:
                if comp.reaction_role in {"SOLVENT", "PRODUCT"}:
                    continue
                n = name_of(comp)
                if n and self._REDUCTANT_PATTERN.search(n.lower()):
                    return []
        return [
            RuleViolation(
                rule_id=self.id,
                severity=Severity.ERROR,
                message=(
                    "Reduction detected but the reducing agent is not "
                    "declared with reaction_role REACTANT or REAGENT."
                ),
                fix_hint=(
                    "Identify the reducing agent (NaBH4, LiAlH4, DIBAL, "
                    "H2/Pd, etc.) and mark it REAGENT or REACTANT. Don't "
                    "leave it inside a workup component."
                ),
                path="inputs[*].components[*].reaction_role",
            )
        ]


class EsterificationNeedsBothPartners(Rule):
    """CLS-004: a classical Fischer-type esterification needs the acid and
    the alcohol both declared as REACTANTS."""

    id = "CLS-004"
    description = (
        "Esterification must declare both the carboxylic acid and the "
        "alcohol as REACTANT inputs."
    )

    _ACID_PATTERN = re.compile(r"\b\w+ic\s+acid\b")
    _ALCOHOL_PATTERN = re.compile(r"\b(\w*ol|\w*alcohol)\b")

    def check(self, draft: ReactionDraft) -> list[RuleViolation]:
        if classify_reaction(draft).cls != ReactionClass.ESTERIFICATION:
            return []
        has_acid_reactant = False
        has_alcohol_reactant = False
        for inp in draft.inputs:
            for comp in inp.components:
                if comp.reaction_role != "REACTANT":
                    continue
                n = name_of(comp)
                if not n:
                    continue
                low = n.lower()
                if self._ACID_PATTERN.search(low):
                    has_acid_reactant = True
                if self._ALCOHOL_PATTERN.search(low):
                    has_alcohol_reactant = True
        if has_acid_reactant and has_alcohol_reactant:
            return []
        return [
            RuleViolation(
                rule_id=self.id,
                severity=Severity.WARNING,
                message=(
                    "Esterification heuristically detected, but the acid "
                    "and/or alcohol partner is not flagged as REACTANT."
                ),
                fix_hint=(
                    "Mark both the carboxylic acid and the alcohol with "
                    "reaction_role='REACTANT'. The dehydration agent "
                    "(DCC, EDC, H2SO4, etc.) is a REAGENT or CATALYST."
                ),
                path="inputs[*].components[*].reaction_role",
            )
        ]


CLS_RULES: list[Rule] = [
    SuzukiRequiredComponents(),
    GrignardRequiresInertAtmosphere(),
    ReductionNeedsReducingAgent(),
    EsterificationNeedsBothPartners(),
]
