"""Completeness rules (CMP-*)."""

from __future__ import annotations

from eln_structurer.rules.base import Rule, RuleViolation, Severity
from eln_structurer.schema import ReactionDraft


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


CMP_RULES: list[Rule] = [
    HasReactant(),
    HasProduct(),
    HasTemperature(),
    HasDuration(),
    NotesCaptureSource(),
]
