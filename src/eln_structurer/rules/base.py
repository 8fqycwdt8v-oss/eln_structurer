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

    def format_line(self) -> str:
        sev = "ERROR" if self.severity is Severity.ERROR else "WARN "
        loc = f" at {self.path}" if self.path else ""
        return f"[{sev} {self.rule_id}]{loc}: {self.message}\n  Fix: {self.fix_hint}"


class Rule(ABC):
    id: str
    description: str

    @abstractmethod
    def check(self, draft: ReactionDraft) -> list[RuleViolation]:
        ...
