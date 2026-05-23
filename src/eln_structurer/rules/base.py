"""Base types and infrastructure for the rule pack.

Provides:
- ``Severity`` enum, ``RuleViolation`` dataclass, ``Rule`` abstract base.
- ``@register_rule`` decorator + ``get_all_rules`` — replaces the
  per-module ``*_RULES`` list aggregation pattern.
- ``walk_compounds`` traversal helper used by every rule that needs to
  iterate the (input, component) grid.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum

from eln_structurer.schema import CompoundModel, ReactionDraft


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


# ---------------------------------------------------------------------------
# Rule registry — decorator-driven replacement for per-module *_RULES lists.
# ---------------------------------------------------------------------------

_RULE_REGISTRY: list[type[Rule]] = []


def register_rule(cls: type[Rule]) -> type[Rule]:
    """Class decorator: every Rule subclass marked with this is auto-added
    to ``ALL_RULES`` (via ``get_all_rules``). Replaces the boilerplate
    pattern of declaring ``CMP_RULES = [...]`` at the bottom of each
    rules/*.py module."""
    if cls in _RULE_REGISTRY:
        return cls
    _RULE_REGISTRY.append(cls)
    return cls


def get_all_rules() -> list[Rule]:
    """Instantiate every registered rule once. Stable order = registration order."""
    return [cls() for cls in _RULE_REGISTRY]


# ---------------------------------------------------------------------------
# Shared traversal helper — replaces the
# ``for inp in draft.inputs: for comp in inp.components`` pattern that
# appeared dozens of times across the rule pack.
# ---------------------------------------------------------------------------


def walk_compounds(
    draft: ReactionDraft,
    *,
    role: str | None = None,
) -> Iterator[tuple[int, int, CompoundModel, str]]:
    """Yield ``(input_idx, component_idx, compound, path)`` for every
    compound in every input.

    ``path`` is a JSONPath-like string suitable for ``RuleViolation.path``.
    ``role`` filters by ``compound.reaction_role`` when supplied.
    """
    for i, inp in enumerate(draft.inputs):
        for j, comp in enumerate(inp.components):
            if role is not None and comp.reaction_role != role:
                continue
            yield i, j, comp, f"inputs[{i}].components[{j}]"
