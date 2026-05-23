"""Completeness rules (CMP-*)."""

from __future__ import annotations

from eln_structurer.rules.base import Rule, RuleViolation, Severity, register_rule
from eln_structurer.schema import ReactionDraft


@register_rule
class HasReactant(Rule):
    id = "CMP-001"
    description = "At least one input must have role REACTANT."

    def check(self, draft: ReactionDraft) -> list[RuleViolation]:
        for inp in draft.inputs:
            for comp in inp.components:
                if comp.reaction_role == "REACTANT":
                    return []
        return [
            RuleViolation(
                rule_id=self.id,
                severity=Severity.ERROR,
                message="No input component is flagged with reaction_role=REACTANT.",
                fix_hint=(
                    "At least one compound must have reaction_role='REACTANT'. "
                    "Identify the main starting material from the paragraph and "
                    "mark it accordingly."
                ),
                path="inputs[*].components[*].reaction_role",
            )
        ]


@register_rule
class HasProduct(Rule):
    id = "CMP-002"
    description = "At least one outcome with at least one product."

    def check(self, draft: ReactionDraft) -> list[RuleViolation]:
        for outcome in draft.outcomes:
            if outcome.products:
                return []
        return [
            RuleViolation(
                rule_id=self.id,
                severity=Severity.ERROR,
                message="No product was captured in any outcome.",
                fix_hint=(
                    "Add a ProductModel under outcomes[0].products describing the "
                    "isolated product (compound name and/or SMILES, plus yield "
                    "measurement if stated)."
                ),
                path="outcomes[0].products",
            )
        ]


@register_rule
class HasTemperature(Rule):
    id = "CMP-003"
    description = "conditions.temperature must be set (even if AMBIENT)."

    def check(self, draft: ReactionDraft) -> list[RuleViolation]:
        if draft.conditions.temperature is None:
            return [
                RuleViolation(
                    rule_id=self.id,
                    severity=Severity.ERROR,
                    message="conditions.temperature is missing.",
                    fix_hint=(
                        "Set conditions.temperature. If the paragraph says 'rt' or "
                        "doesn't specify, use control_type='AMBIENT' with "
                        "setpoint_celsius=null."
                    ),
                    path="conditions.temperature",
                )
            ]
        return []


@register_rule
class HasDuration(Rule):
    id = "CMP-004"
    description = "Reaction duration should be set somewhere."

    def check(self, draft: ReactionDraft) -> list[RuleViolation]:
        if draft.conditions.duration_minutes is not None:
            return []
        for outcome in draft.outcomes:
            if outcome.reaction_time_minutes is not None:
                return []
        return [
            RuleViolation(
                rule_id=self.id,
                severity=Severity.WARNING,
                message="No reaction duration captured.",
                fix_hint=(
                    "If the paragraph mentions a duration (e.g. 'stirred for 16 h'), "
                    "set conditions.duration_minutes or outcomes[0].reaction_time_minutes."
                ),
                path="conditions.duration_minutes",
            )
        ]


@register_rule
class NotesCaptureSource(Rule):
    id = "CMP-005"
    description = "notes should capture provenance / source paragraph reference."

    def check(self, draft: ReactionDraft) -> list[RuleViolation]:
        if not draft.notes or not draft.notes.strip():
            return [
                RuleViolation(
                    rule_id=self.id,
                    severity=Severity.WARNING,
                    message="notes field is empty.",
                    fix_hint=(
                        "Use the notes field to record citation info or any "
                        "non-structured details from the paragraph that don't fit "
                        "elsewhere."
                    ),
                    path="notes",
                )
            ]
        return []


@register_rule
class TemperatureRangeSanity(Rule):
    id = "CMP-006"
    description = "Reaction temperature should be within a chemically plausible range."

    LOW_C = -100.0   # liquid-N2 cooling reaches ~-196, but rarely is the
                     # *setpoint* itself below -100
    HIGH_C = 300.0   # bench-scale chemistry rarely goes above 300 °C; flow
                     # / high-T reactors do, but those usually carry
                     # control_type=CUSTOM details

    def check(self, draft: ReactionDraft) -> list[RuleViolation]:
        temp = draft.conditions.temperature
        if temp is None or temp.setpoint_celsius is None:
            return []
        c = temp.setpoint_celsius
        if c < self.LOW_C or c > self.HIGH_C:
            return [
                RuleViolation(
                    rule_id=self.id,
                    severity=Severity.ERROR,
                    message=(
                        f"Temperature setpoint is outside the plausible "
                        f"range [{self.LOW_C}, {self.HIGH_C}] °C. Likely a unit "
                        "confusion (K vs. °C) or a transcription error."
                    ),
                    fix_hint=(
                        "Re-read the paragraph. If the source reports Kelvin, "
                        "convert: T(°C) = T(K) − 273.15. If the value is real "
                        "and intentional, set control_type='CUSTOM' and put the "
                        "reactor description in conditions.atmosphere or notes."
                    ),
                    path="conditions.temperature.setpoint_celsius",
                    actual_value=f"{c} °C",
                )
            ]
        return []


@register_rule
class DurationRangeSanity(Rule):
    id = "CMP-007"
    description = "Reaction duration should be within a plausible bench-scale range."

    MAX_MINUTES = 60 * 24 * 14   # two weeks — anything longer is almost
                                 # certainly an extraction error
    MIN_MINUTES = 0.0

    def check(self, draft: ReactionDraft) -> list[RuleViolation]:
        candidates: list[tuple[float, str]] = []
        if draft.conditions.duration_minutes is not None:
            candidates.append((draft.conditions.duration_minutes, "conditions.duration_minutes"))
        for oi, outcome in enumerate(draft.outcomes):
            if outcome.reaction_time_minutes is not None:
                candidates.append(
                    (outcome.reaction_time_minutes, f"outcomes[{oi}].reaction_time_minutes")
                )
        violations: list[RuleViolation] = []
        for minutes, path in candidates:
            if minutes < self.MIN_MINUTES:
                violations.append(
                    RuleViolation(
                        rule_id=self.id,
                        severity=Severity.ERROR,
                        message="Negative duration.",
                        fix_hint="Durations are non-negative; re-read the paragraph.",
                        path=path,
                        actual_value=f"{minutes} min",
                    )
                )
            elif minutes > self.MAX_MINUTES:
                violations.append(
                    RuleViolation(
                        rule_id=self.id,
                        severity=Severity.ERROR,
                        message=(
                            "Duration is implausibly long (>2 weeks). Likely "
                            "a unit confusion (s vs. min vs. h)."
                        ),
                        fix_hint=(
                            "Confirm the time unit. If the paragraph says 'h', "
                            "convert to minutes: minutes = hours * 60."
                        ),
                        path=path,
                        actual_value=f"{minutes} min",
                    )
                )
        return violations


