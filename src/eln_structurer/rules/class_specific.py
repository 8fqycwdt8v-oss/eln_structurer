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
from eln_structurer.rules.base import Rule, RuleViolation, Severity, register_rule
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


@register_rule
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


@register_rule
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


@register_rule
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


@register_rule
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


@register_rule
class AmideCouplingHasAmineAndAcid(Rule):
    """CLS-005: amide coupling needs an amine REACTANT and a carboxylic
    acid (or activated acid) REACTANT.

    The classifier already detected a coupling reagent (EDC / HATU / etc.).
    What it cannot verify by name is that the two coupling partners are
    actually flagged as REACTANTs and not stuck in the wrong role.
    """

    id = "CLS-005"
    description = "Amide coupling requires an amine and a carboxylic acid (or activated acid) as REACTANTs."

    _AMINE_PATTERN = re.compile(
        r"\bamine\b|\baniline\b|piperazine|piperidine|morpholine|"
        r"\bnh2\b|\bnh\b|amino[- ]"
    )
    _ACID_OR_ACTIVATED = re.compile(
        r"\b\w+ic\s+acid\b|\b\w+yl\s+chloride\b|carboxylic|"
        r"\bcarbonyl chloride\b|acyl chloride|acid chloride"
    )

    def check(self, draft: ReactionDraft) -> list[RuleViolation]:
        if classify_reaction(draft).cls != ReactionClass.AMIDE_FORMATION:
            return []
        has_amine = False
        has_acid = False
        for inp in draft.inputs:
            for comp in inp.components:
                if comp.reaction_role != "REACTANT":
                    continue
                n = name_of(comp)
                if not n:
                    continue
                low = n.lower()
                if self._AMINE_PATTERN.search(low):
                    has_amine = True
                if self._ACID_OR_ACTIVATED.search(low):
                    has_acid = True
        if has_amine and has_acid:
            return []
        return [
            RuleViolation(
                rule_id=self.id,
                severity=Severity.WARNING,
                message=(
                    "Amide coupling detected (peptide coupling reagent present) "
                    "but the amine and/or carboxylic acid partner is not flagged "
                    "as REACTANT."
                ),
                fix_hint=(
                    "Mark both the amine and the carboxylic acid (or acyl "
                    "chloride) with reaction_role='REACTANT'. The coupling "
                    "reagent (EDC, HATU, DCC, …) is a REAGENT, not a REACTANT."
                ),
                path="inputs[*].components[*].reaction_role",
            )
        ]


@register_rule
class BocDeprotectionNeedsAcid(Rule):
    """CLS-006: Boc deprotection must list an acid (TFA, HCl/dioxane) as
    a REAGENT.
    """

    id = "CLS-006"
    description = "Boc deprotection requires TFA or HCl as a REAGENT."

    _DEPROT_ACID = re.compile(
        r"\b(tfa|trifluoroacetic acid|hcl|hydrochloric acid|"
        r"\d+\s*m\s+hcl)\b"
    )

    def check(self, draft: ReactionDraft) -> list[RuleViolation]:
        if classify_reaction(draft).cls != ReactionClass.BOC_DEPROTECTION:
            return []
        for inp in draft.inputs:
            for comp in inp.components:
                if comp.reaction_role not in {"REAGENT", "SOLVENT"}:
                    continue
                n = name_of(comp)
                if n and self._DEPROT_ACID.search(n.lower()):
                    return []
        return [
            RuleViolation(
                rule_id=self.id,
                severity=Severity.ERROR,
                message=(
                    "Boc/Cbz/Fmoc deprotection detected but no acid (TFA, "
                    "HCl/dioxane, HCl/EtOAc) is declared as a REAGENT or SOLVENT."
                ),
                fix_hint=(
                    "Add the deprotection acid as a REAGENT (TFA, HCl in "
                    "dioxane, or HCl/EtOAc). Hydrogenolysis of Cbz needs "
                    "H2/Pd-C instead."
                ),
                path="inputs[*].components[*]",
            )
        ]


@register_rule
class ReductiveAminationHasCarbonylAndAmine(Rule):
    """CLS-007: reductive amination needs an amine + a carbonyl (aldehyde
    or ketone) + a hydride source."""

    id = "CLS-007"
    description = "Reductive amination requires an amine REACTANT, a carbonyl REACTANT, and a hydride reductant."

    _CARBONYL_PATTERN = re.compile(
        r"\b\w+aldehyde\b|\b\w+anal\b|\b\w+anone\b|"
        r"acetone|ketone|aldehyde"
    )
    _AMINE_PATTERN = re.compile(
        r"\bamine\b|\baniline\b|piperazine|piperidine|morpholine|amino[- ]"
    )

    def check(self, draft: ReactionDraft) -> list[RuleViolation]:
        if classify_reaction(draft).cls != ReactionClass.REDUCTIVE_AMINATION:
            return []
        has_carbonyl = False
        has_amine = False
        for inp in draft.inputs:
            for comp in inp.components:
                if comp.reaction_role not in {"REACTANT", "REAGENT"}:
                    continue
                n = name_of(comp)
                if not n:
                    continue
                low = n.lower()
                if self._CARBONYL_PATTERN.search(low):
                    has_carbonyl = True
                if self._AMINE_PATTERN.search(low):
                    has_amine = True
        if has_carbonyl and has_amine:
            return []
        missing = []
        if not has_carbonyl:
            missing.append("carbonyl (aldehyde/ketone)")
        if not has_amine:
            missing.append("amine")
        return [
            RuleViolation(
                rule_id=self.id,
                severity=Severity.WARNING,
                message=(
                    "Reductive-amination–specific reductant present but "
                    f"missing: {', '.join(missing)}."
                ),
                fix_hint=(
                    "Reductive amination joins an amine with a carbonyl via "
                    "hydride reduction of the in-situ iminium. Make sure both "
                    "partners are in the inputs with REACTANT or REAGENT role."
                ),
                path="inputs[*].components[*]",
            )
        ]


@register_rule
class BuchwaldHartwigComponents(Rule):
    """CLS-008: Buchwald–Hartwig needs Pd + ligand + base + amine + aryl halide."""

    id = "CLS-008"
    description = "Buchwald–Hartwig amination requires Pd, a phosphine ligand, a base, an amine, and an aryl halide."

    _BASE_PATTERN = re.compile(
        r"\b(k2co3|cs2co3|k3po4|naotbu|kotbu|naotms|naoh|"
        r"sodium tert-butoxide|potassium tert-butoxide|lihmds|nahmds)\b"
    )
    _AMINE_PATTERN = re.compile(
        r"\bamine\b|piperazine|piperidine|morpholine|amino[- ]|aniline"
    )

    def check(self, draft: ReactionDraft) -> list[RuleViolation]:
        if classify_reaction(draft).cls != ReactionClass.BUCHWALD_HARTWIG:
            return []
        has_base = False
        has_amine = False
        for inp in draft.inputs:
            for comp in inp.components:
                n = name_of(comp)
                if not n:
                    continue
                low = n.lower()
                if self._BASE_PATTERN.search(low):
                    has_base = True
                if comp.reaction_role == "REACTANT" and self._AMINE_PATTERN.search(low):
                    has_amine = True
        missing = []
        if not has_base:
            missing.append("base (e.g. K2CO3, Cs2CO3, NaOtBu)")
        if not has_amine:
            missing.append("amine REACTANT")
        if not missing:
            return []
        return [
            RuleViolation(
                rule_id=self.id,
                severity=Severity.ERROR,
                message=(
                    "Buchwald–Hartwig detected (Pd + phosphine ligand) but "
                    f"missing: {', '.join(missing)}."
                ),
                fix_hint=(
                    "Add the missing component as an input with the right "
                    "reaction_role. The base activates the amine; the amine "
                    "is the REACTANT."
                ),
                path="inputs[*].components[*]",
            )
        ]


@register_rule
class MitsunobuComponents(Rule):
    """CLS-009: Mitsunobu requires DIAD + PPh3 + a nucleophile and an alcohol."""

    id = "CLS-009"
    description = "Mitsunobu requires both DIAD/DEAD and PPh3."

    _DIAD = re.compile(
        r"\b(diad|dead|diethyl azodicarboxylate|diisopropyl azodicarboxylate)\b"
    )
    _PPH3 = re.compile(r"\b(pph3|triphenylphosphine|ph3p)\b")

    def check(self, draft: ReactionDraft) -> list[RuleViolation]:
        if classify_reaction(draft).cls != ReactionClass.MITSUNOBU:
            return []
        has_diad = False
        has_pph3 = False
        for inp in draft.inputs:
            for comp in inp.components:
                n = name_of(comp)
                if not n:
                    continue
                low = n.lower()
                if self._DIAD.search(low):
                    has_diad = True
                if self._PPH3.search(low):
                    has_pph3 = True
        if has_diad and has_pph3:
            return []
        missing = []
        if not has_diad:
            missing.append("DIAD or DEAD")
        if not has_pph3:
            missing.append("PPh3 (triphenylphosphine)")
        return [
            RuleViolation(
                rule_id=self.id,
                severity=Severity.ERROR,
                message=(
                    f"Mitsunobu detected but missing: {', '.join(missing)}. "
                    "Both DIAD/DEAD and PPh3 are required."
                ),
                fix_hint=(
                    "Add the missing reagent as an input with reaction_role"
                    "='REAGENT'."
                ),
                path="inputs[*].components[*]",
            )
        ]


@register_rule
class WittigNeedsCarbonyl(Rule):
    """CLS-010: Wittig (or HWE / Julia) reactions need a carbonyl REACTANT.

    Severity is WARNING: intramolecular variants and stabilized-ylide edge
    cases exist, and we'd rather miss a real Wittig than wrongly block a
    finalize.
    """

    id = "CLS-010"
    description = "Wittig / HWE / Julia reactions need an aldehyde or ketone REACTANT."

    _CARBONYL_PATTERN = re.compile(
        r"\b\w*aldehyde\b|\b\w+anone\b|\b\w+anal\b|"
        r"\bacetone\b|\bketone\b|\baldehyde\b"
    )

    def check(self, draft: ReactionDraft) -> list[RuleViolation]:
        if classify_reaction(draft).cls != ReactionClass.WITTIG:
            return []
        for inp in draft.inputs:
            for comp in inp.components:
                if comp.reaction_role != "REACTANT":
                    continue
                n = name_of(comp)
                if n and self._CARBONYL_PATTERN.search(n.lower()):
                    return []
        return [
            RuleViolation(
                rule_id=self.id,
                severity=Severity.WARNING,
                message=(
                    "Wittig / HWE detected (phosphonium ylide or "
                    "phosphonate present) but no carbonyl REACTANT found."
                ),
                fix_hint=(
                    "Identify the aldehyde or ketone partner from the "
                    "paragraph and flag it as REACTANT. Intramolecular "
                    "Wittigs are possible — if the substrate carries the "
                    "carbonyl internally, the warning is safe to ignore."
                ),
                path="inputs[*].components[*].reaction_role",
            )
        ]


@register_rule
class HalogenatingAgentIsNotLimiting(Rule):
    """CLS-011: in a halogenation, the substrate (not NBS/Selectfluor/etc.)
    should be the limiting reagent.

    Mismatched is_limiting on the halogenating agent is a common LLM bug:
    the model picks the smallest-mole compound regardless of role. Severity
    ERROR because it inverts the chemistry — yields, equivalents, and
    mass-balance checks downstream all go wrong if the wrong compound
    is flagged limiting.
    """

    id = "CLS-011"
    description = "Halogenating agents (NBS, NCS, Selectfluor, DAST, …) should not be flagged is_limiting=True."

    _HALOGENATING_NAMES = re.compile(
        r"\b(nbs|n[-\s]?bromosuccinimide|ncs|n[-\s]?chlorosuccinimide|"
        r"nis|n[-\s]?iodosuccinimide|selectfluor|deoxo[-\s]?fluor|"
        r"dast|nfsi)\b"
    )

    def check(self, draft: ReactionDraft) -> list[RuleViolation]:
        if classify_reaction(draft).cls != ReactionClass.HALOGENATION:
            return []
        violations: list[RuleViolation] = []
        for i, inp in enumerate(draft.inputs):
            for j, comp in enumerate(inp.components):
                if not comp.is_limiting:
                    continue
                n = name_of(comp)
                if n and self._HALOGENATING_NAMES.search(n.lower()):
                    violations.append(
                        RuleViolation(
                            rule_id=self.id,
                            severity=Severity.ERROR,
                            message=(
                                f"is_limiting=True on a halogenating agent "
                                f"({n!r}). The substrate is the limiting "
                                "reagent in a halogenation, not the reagent."
                            ),
                            fix_hint=(
                                "Move is_limiting=True to the organic substrate "
                                "(typically a REACTANT) and lower this compound's "
                                "role to REAGENT."
                            ),
                            path=f"inputs[{i}].components[{j}].is_limiting",
                        )
                    )
        return violations


@register_rule
class OxidantIsNotLimiting(Rule):
    """CLS-012: in an oxidation, the substrate (not the oxidant) is the
    limiting reagent.

    Same anti-pattern as CLS-011, applied to DMP / Swern / mCPBA / PCC /
    TEMPO / KMnO4 / periodate / H2O2 etc. ERROR-severity because flipping
    the limiting reagent breaks every downstream stoichiometry check.
    """

    id = "CLS-012"
    description = "Oxidants (DMP, Swern, PCC, mCPBA, TEMPO, KMnO4, H2O2, …) should not be flagged is_limiting=True."

    _OXIDANT_NAMES = re.compile(
        r"\b(dess[-\s]?martin|dmp|swern|moffatt|pcc|"
        r"pyridinium chlorochromate|pdc|jones|kmno4|"
        r"potassium permanganate|mno2|manganese dioxide|"
        r"tempo|baib|oxone|m[-\s]?cpba|mcpba|"
        r"3[-\s]?chloroperbenzoic|hydrogen peroxide|h2o2|"
        r"nai04|naio4|periodate)\b"
    )

    def check(self, draft: ReactionDraft) -> list[RuleViolation]:
        if classify_reaction(draft).cls != ReactionClass.OXIDATION:
            return []
        violations: list[RuleViolation] = []
        for i, inp in enumerate(draft.inputs):
            for j, comp in enumerate(inp.components):
                if not comp.is_limiting:
                    continue
                n = name_of(comp)
                if n and self._OXIDANT_NAMES.search(n.lower()):
                    violations.append(
                        RuleViolation(
                            rule_id=self.id,
                            severity=Severity.ERROR,
                            message=(
                                f"is_limiting=True on an oxidant ({n!r}). "
                                "The substrate is the limiting reagent in "
                                "an oxidation, not the oxidant."
                            ),
                            fix_hint=(
                                "Move is_limiting=True to the alcohol / "
                                "alkene / sulfide substrate and lower this "
                                "compound's role to REAGENT."
                            ),
                            path=f"inputs[{i}].components[{j}].is_limiting",
                        )
                    )
        return violations


@register_rule
class NAlkylationNeedsElectrophile(Rule):
    """CLS-013: N-alkylation needs both an amine REACTANT and an
    alkylating electrophile (alkyl halide, mesylate, tosylate, etc.).

    The classifier only fires on the literal "N-alkylation" text hint, so
    we already know the agent labelled the reaction this way; the rule
    enforces that both partners are structurally present in the inputs.
    Severity WARNING because some alkylations use less-named electrophiles
    (e.g. epoxides, Michael acceptors) we don't recognize by name.
    """

    id = "CLS-013"
    description = "N-alkylation requires an amine REACTANT and a named electrophile."

    _AMINE = re.compile(
        r"\bamine\b|piperazine|piperidine|morpholine|amino[- ]|aniline"
    )
    # Matches three shapes of alkylating electrophile:
    # - "4-bromoanisole" / "iodobenzene" / "chloroacetonitrile"
    # - "benzyl bromide" / "methyl iodide" / "ethyl chloride"
    # - "mesylate" / "tosylate" / "triflate" with any leading word
    # - bare "epoxide" / "oxirane"
    _ELECTROPHILE = re.compile(
        r"\b\d?-?(bromo|iodo|chloro)\w+"
        r"|\b\w+\s+(bromide|iodide|chloride)\b"
        r"|\b\w+\s+(mesylate|tosylate|triflate)\b"
        r"|\bepoxide\b|\boxirane\b"
    )

    def check(self, draft: ReactionDraft) -> list[RuleViolation]:
        if classify_reaction(draft).cls != ReactionClass.N_ALKYLATION:
            return []
        has_amine = False
        has_electrophile = False
        for inp in draft.inputs:
            for comp in inp.components:
                if comp.reaction_role != "REACTANT":
                    continue
                n = name_of(comp)
                if not n:
                    continue
                low = n.lower()
                if self._AMINE.search(low):
                    has_amine = True
                if self._ELECTROPHILE.search(low):
                    has_electrophile = True
        if has_amine and has_electrophile:
            return []
        missing = []
        if not has_amine:
            missing.append("amine REACTANT")
        if not has_electrophile:
            missing.append("alkylating electrophile (alkyl halide / mesylate / tosylate / triflate)")
        return [
            RuleViolation(
                rule_id=self.id,
                severity=Severity.WARNING,
                message=(
                    f"N-alkylation declared but missing: {', '.join(missing)}."
                ),
                fix_hint=(
                    "Flag both the amine and the electrophile as REACTANTs. "
                    "If the electrophile is unusual (epoxide / Michael "
                    "acceptor) the warning may be acceptable."
                ),
                path="inputs[*].components[*]",
            )
        ]


