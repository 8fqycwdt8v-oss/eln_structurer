"""Numeric grounding rules (NUM-*).

These rules enforce two invariants designed to combat numeric
hallucination in LLM-driven extraction:

1. **Grounding**: every numeric value carrying a ``source_quote`` must
   have that quote present verbatim in the source paragraph (after light
   whitespace normalisation). NUM-001 / NUM-002.

2. **Honest gaps**: the ``unspecified_fields`` list must be (a) well-shaped
   JSONPath-like strings and (b) point at fields that are actually empty
   in the draft. Prevents the agent claiming a field is "unspecified"
   while quietly filling it in. NUM-003.

Grounding is OPTIONAL — drafts that don't populate source_quote skip
NUM-001/002 entirely. Adoption is encouraged via the prompt; the rules
fire only when the agent has claimed a quote that doesn't hold up.
"""

from __future__ import annotations

import re

from eln_structurer.rules.base import Rule, RuleViolation, Severity, register_rule
from eln_structurer.schema import ReactionDraft
from eln_structurer.text_utils import normalize_for_substring_search as _normalise_for_search


# Recognised unit aliases for the unit-in-quote check. Keys are the
# AmountUnit enum values; values are sets of strings any of which may
# appear in the source quote to satisfy NUM-002.
_UNIT_ALIASES: dict[str, set[str]] = {
    "g": {"g", "gram", "grams"},
    "mg": {"mg", "milligram", "milligrams"},
    "kg": {"kg", "kilogram", "kilograms"},
    "mol": {"mol", "mole", "moles"},
    "mmol": {"mmol", "millimole", "millimoles"},
    "umol": {"umol", "µmol", "μmol", "micromole", "micromoles"},
    "L": {"l", "liter", "liters", "litre", "litres"},
    "mL": {"ml", "milliliter", "milliliters", "millilitre", "millilitres"},
    "uL": {"ul", "µl", "μl", "microliter", "microlitre"},
    "equiv": {"equiv", "equivalent", "equivalents", "eq"},
    "mass_pct": {"%", "wt%", "w/w"},
    "mol_pct": {"%", "mol%"},
    "vol_pct": {"%", "v/v"},
}


@register_rule
class NumericValueGrounded(Rule):
    """NUM-001: source_quote (when set) must appear in source_paragraph."""

    id = "NUM-001"
    description = "source_quote on numeric fields must appear verbatim in source_paragraph."

    def check(self, draft: ReactionDraft) -> list[RuleViolation]:
        haystack = _normalise_for_search(draft.source_paragraph or "")
        if not haystack:
            return []
        violations: list[RuleViolation] = []
        for i, inp in enumerate(draft.inputs):
            for j, comp in enumerate(inp.components):
                if comp.amount is None or comp.amount.source_quote is None:
                    continue
                if comp.amount.inferred:
                    continue
                quote = _normalise_for_search(comp.amount.source_quote)
                if quote and quote not in haystack:
                    violations.append(
                        RuleViolation(
                            rule_id=self.id,
                            severity=Severity.ERROR,
                            message=(
                                "Amount.source_quote does not appear in "
                                "source_paragraph. Either the quote is wrong "
                                "or the value was hallucinated."
                            ),
                            fix_hint=(
                                "Re-read the paragraph and set source_quote "
                                "to the exact substring that contains both "
                                "the number and its unit, or set inferred=True "
                                "if the value was derived."
                            ),
                            path=f"inputs[{i}].components[{j}].amount.source_quote",
                            actual_value=repr(comp.amount.source_quote),
                        )
                    )

        for oi, outcome in enumerate(draft.outcomes):
            for pi, prod in enumerate(outcome.products):
                for mi, m in enumerate(prod.measurements):
                    if m.source_quote is None or m.inferred:
                        continue
                    quote = _normalise_for_search(m.source_quote)
                    if quote and quote not in haystack:
                        violations.append(
                            RuleViolation(
                                rule_id=self.id,
                                severity=Severity.ERROR,
                                message=(
                                    "ProductMeasurement.source_quote does not "
                                    "appear in source_paragraph."
                                ),
                                fix_hint=(
                                    "Set source_quote to a verbatim substring "
                                    "or mark inferred=True."
                                ),
                                path=(
                                    f"outcomes[{oi}].products[{pi}]"
                                    f".measurements[{mi}].source_quote"
                                ),
                                actual_value=repr(m.source_quote),
                            )
                        )
        return violations


@register_rule
class UnitMatchesQuote(Rule):
    """NUM-002: the source_quote should contain the unit (or an alias).

    Severity is WARNING — units can sometimes appear elsewhere in the
    paragraph, but the rule reduces hallucinations where the agent
    captures a number and units that don't co-occur in the text.
    """

    id = "NUM-002"
    description = "source_quote should contain the declared unit (or a known alias)."

    def check(self, draft: ReactionDraft) -> list[RuleViolation]:
        violations: list[RuleViolation] = []
        for i, inp in enumerate(draft.inputs):
            for j, comp in enumerate(inp.components):
                if comp.amount is None or comp.amount.source_quote is None:
                    continue
                if comp.amount.inferred:
                    continue
                aliases = _UNIT_ALIASES.get(comp.amount.units, {comp.amount.units})
                low = comp.amount.source_quote.lower()
                if not any(alias in low for alias in aliases):
                    violations.append(
                        RuleViolation(
                            rule_id=self.id,
                            severity=Severity.WARNING,
                            message=(
                                f"source_quote {comp.amount.source_quote!r} "
                                f"does not contain the declared unit "
                                f"{comp.amount.units!r} (or a known alias)."
                            ),
                            fix_hint=(
                                "Extend the quote to include the unit, or "
                                "correct the units field to match what the "
                                "paragraph actually says."
                            ),
                            path=f"inputs[{i}].components[{j}].amount",
                            actual_value=comp.amount.source_quote,
                        )
                    )
        return violations


# Loose JSONPath-like pattern: starts with a known top-level key, may
# contain bracketed indices and dotted attributes. Strict enough to catch
# typos, loose enough to allow the obvious shapes the prompt suggests.
_JSONPATH_RE = re.compile(
    r"^(identifiers|inputs|conditions|workups|outcomes|notes|source_paragraph)"
    r"([.\[][\w\[\].:*-]*)*$"
)

# Map dotted prefixes the agent will write to attribute paths on the draft.
# Returns the attribute value or a sentinel ("MISSING") meaning the field
# is in fact unset. Only covers the dotted scalar fields callers most
# commonly mark unspecified — list element ([N]) lookups are skipped
# because list emptiness is already obvious from the draft.
def _resolve_dotted(draft: ReactionDraft, path: str) -> object | str:
    if "[" in path:
        return "SKIP"  # don't try to resolve list/index paths
    parts = path.split(".")
    obj: object = draft
    for p in parts:
        if obj is None or not hasattr(obj, p):
            return "MISSING"
        obj = getattr(obj, p)
    return obj


@register_rule
class UnspecifiedFieldsAreValid(Rule):
    """NUM-003: ``unspecified_fields`` must be well-shaped paths AND
    point at fields that are actually empty in the draft.
    """

    id = "NUM-003"
    description = (
        "unspecified_fields entries must be valid paths and must point at "
        "fields that the draft has actually left empty."
    )

    def check(self, draft: ReactionDraft) -> list[RuleViolation]:
        violations: list[RuleViolation] = []
        for k, path in enumerate(draft.unspecified_fields):
            if not isinstance(path, str) or not path.strip():
                violations.append(
                    RuleViolation(
                        rule_id=self.id,
                        severity=Severity.WARNING,
                        message="unspecified_fields entry is empty or non-string.",
                        fix_hint="Remove the entry or replace it with a valid path.",
                        path=f"unspecified_fields[{k}]",
                        actual_value=repr(path),
                    )
                )
                continue
            if not _JSONPATH_RE.match(path):
                violations.append(
                    RuleViolation(
                        rule_id=self.id,
                        severity=Severity.WARNING,
                        message=(
                            "unspecified_fields entry is not a recognised "
                            "JSONPath-like string. Expected forms: "
                            "'conditions.duration_minutes', "
                            "'outcomes[0].reaction_time_minutes'."
                        ),
                        fix_hint=(
                            "Use a top-level key (identifiers/inputs/conditions/"
                            "workups/outcomes/notes) followed by dotted "
                            "attributes or bracketed indices."
                        ),
                        path=f"unspecified_fields[{k}]",
                        actual_value=path,
                    )
                )
                continue
            value = _resolve_dotted(draft, path)
            if value == "MISSING":
                # Path resolved to a missing attribute — slight typo. Don't
                # error; warn so the agent can correct or remove the entry.
                violations.append(
                    RuleViolation(
                        rule_id=self.id,
                        severity=Severity.WARNING,
                        message=(
                            "unspecified_fields path does not resolve to a "
                            "real attribute on the draft."
                        ),
                        fix_hint="Drop the entry or fix the path.",
                        path=f"unspecified_fields[{k}]",
                        actual_value=path,
                    )
                )
                continue
            if value == "SKIP":
                continue  # list-indexed paths are not introspected here
            if value not in (None, "", [], {}):
                # The path resolved to something non-empty — the agent is
                # both filling the field AND marking it unspecified.
                violations.append(
                    RuleViolation(
                        rule_id=self.id,
                        severity=Severity.WARNING,
                        message=(
                            "unspecified_fields claims a field is missing, "
                            "but the draft has populated it."
                        ),
                        fix_hint=(
                            "Either remove the entry from unspecified_fields "
                            "or clear the value if the paragraph truly didn't "
                            "supply it."
                        ),
                        path=f"unspecified_fields[{k}]",
                        actual_value=str(value),
                    )
                )
        return violations


