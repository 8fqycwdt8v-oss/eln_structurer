"""Base types for the rule pack."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

from eln_structurer.schema import ReactionDraft


class Severity(str, Enum):
    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True)
class RuleViolation:
    rule_id: str
    severity: Severity
    message: str
    fix_hint: str
    path: str = ""
    # The actual offending value (stringified), surfaced in the repair
    # prompt so the agent doesn't have to scan the draft to know what
    # to change. Populated by rules where there's a single bad value to
    # quote; left empty when the rule is about structure/relationships.
    actual_value: str | None = None

    def format_line(self) -> str:
        sev = "ERROR" if self.severity is Severity.ERROR else "WARN "
        loc = f" at {self.path}" if self.path else ""
        head = f"[{sev} {self.rule_id}]{loc}: {self.message}"
        if self.actual_value:
            head += f"\n  actual value: {self.actual_value}"
        return f"{head}\n  Fix: {self.fix_hint}"


class Rule(ABC):
    id: str
    description: str

    @abstractmethod
    def check(self, draft: ReactionDraft) -> list[RuleViolation]:
        ...
