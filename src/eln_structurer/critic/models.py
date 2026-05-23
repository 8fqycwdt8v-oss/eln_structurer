"""Critic Pydantic + dataclass result types.

The internal Pydantic models enforce the critic LLM's output shape; the
public dataclasses are what the agent loop consumes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel


# Pydantic-validated finding shape — the critic's output MUST match this.
class _CriticFindingModel(BaseModel):
    path: str
    severity: Literal["ERROR", "WARNING"]
    message: str


class _CriticResponseModel(BaseModel):
    findings: list[_CriticFindingModel]


@dataclass(frozen=True)
class CriticFinding:
    path: str
    severity: str
    message: str


@dataclass
class CriticReport:
    findings: list[CriticFinding] = field(default_factory=list)
    raw_text: str = ""
    parse_error: str | None = None

    @property
    def is_clean(self) -> bool:
        return self.parse_error is None and not any(
            f.severity == "ERROR" for f in self.findings
        )

    def as_repair_prompt(self) -> str:
        if not self.findings:
            return "CRITIC FOUND NO ISSUES."
        lines = [f"CRITIC FOUND {len(self.findings)} FINDING(S):"]
        for f in self.findings:
            lines.append(f"[{f.severity} {f.path}] {f.message}")
        lines.append(
            "\nFix every finding above, then call validate_reaction and "
            "finalize_reaction again. Do not introduce new fields the "
            "paragraph does not support."
        )
        return "\n".join(lines)
