"""Validation harness: runs the rule pack + ord-schema validation on a draft."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from eln_structurer.proto_bridge import ProtoBridgeError, draft_to_proto
from eln_structurer.rules import ALL_RULES, RuleViolation, Severity
from eln_structurer.schema import ReactionDraft


@dataclass
class ValidationReport:
    errors: list[RuleViolation] = field(default_factory=list)
    warnings: list[RuleViolation] = field(default_factory=list)
    ord_validation_errors: list[str] = field(default_factory=list)
    ord_validation_warnings: list[str] = field(default_factory=list)
    bridge_error: str | None = None

    @property
    def is_clean(self) -> bool:
        return (
            not self.errors
            and not self.ord_validation_errors
            and self.bridge_error is None
        )

    def all_violations(self) -> list[RuleViolation]:
        return [*self.errors, *self.warnings]

    def as_repair_prompt(self) -> str:
        if self.is_clean:
            return "VALIDATION OK."
        n_err = len(self.errors) + len(self.ord_validation_errors) + (1 if self.bridge_error else 0)
        n_warn = len(self.warnings) + len(self.ord_validation_warnings)
        lines = [f"VALIDATION FAILED — {n_err} error(s), {n_warn} warning(s)."]
        if self.bridge_error:
            lines.append(
                f"[ERROR BRIDGE]: Could not coerce draft to ORD proto: {self.bridge_error}"
            )
            lines.append("  Fix: re-emit the draft with valid types per the JSON schema.")
        for v in self.errors:
            lines.append(v.format_line())
        for msg in self.ord_validation_errors:
            lines.append(f"[ERROR ORD-SCHEMA]: {msg}")
            lines.append("  Fix: adjust the corresponding draft field to satisfy ord-schema validation.")
        for v in self.warnings:
            lines.append(v.format_line())
        for msg in self.ord_validation_warnings:
            lines.append(f"[WARN  ORD-SCHEMA]: {msg}")
        lines.append("\nNow produce a corrected draft and call validate_reaction again.")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_clean": self.is_clean,
            "errors": [
                {
                    "rule_id": v.rule_id,
                    "severity": v.severity.value,
                    "message": v.message,
                    "fix_hint": v.fix_hint,
                    "path": v.path,
                }
                for v in self.errors
            ],
            "warnings": [
                {
                    "rule_id": v.rule_id,
                    "severity": v.severity.value,
                    "message": v.message,
                    "fix_hint": v.fix_hint,
                    "path": v.path,
                }
                for v in self.warnings
            ],
            "ord_validation_errors": list(self.ord_validation_errors),
            "ord_validation_warnings": list(self.ord_validation_warnings),
            "bridge_error": self.bridge_error,
        }


def _run_ord_schema_validation(draft: ReactionDraft) -> tuple[list[str], list[str], str | None]:
    """Bridge to proto and run ord_schema.validations.validate_message."""
    try:
        reaction_pb = draft_to_proto(draft)
    except ProtoBridgeError as exc:
        return [], [], str(exc)
    except Exception as exc:  # pragma: no cover — defensive
        return [], [], f"Unexpected bridge error: {exc!r}"

    try:
        from ord_schema import validations
    except ImportError:  # pragma: no cover — declared in pyproject
        return [], [], "ord_schema is not installed; install via `uv sync`."

    output = validations.validate_message(reaction_pb, raise_on_error=False)
    errors = list(getattr(output, "errors", []) or [])
    warnings = list(getattr(output, "warnings", []) or [])
    return errors, warnings, None


def run_harness(draft: ReactionDraft) -> ValidationReport:
    """Run every rule in ALL_RULES + ord-schema validation, return a report."""
    report = ValidationReport()
    for rule in ALL_RULES:
        try:
            violations = rule.check(draft)
        except Exception as exc:  # pragma: no cover — defensive
            report.errors.append(
                RuleViolation(
                    rule_id=rule.id,
                    severity=Severity.ERROR,
                    message=f"Rule {rule.id} raised an exception: {exc!r}",
                    fix_hint="This is a tool bug; the rule itself is broken.",
                )
            )
            continue
        for v in violations:
            if v.severity is Severity.ERROR:
                report.errors.append(v)
            else:
                report.warnings.append(v)

    ord_errors, ord_warnings, bridge_error = _run_ord_schema_validation(draft)
    report.ord_validation_errors = ord_errors
    report.ord_validation_warnings = ord_warnings
    report.bridge_error = bridge_error
    return report
