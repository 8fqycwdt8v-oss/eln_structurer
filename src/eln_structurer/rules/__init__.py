"""Rule pack registry.

Every Rule subclass marks itself with ``@register_rule`` from
``rules.base``. Importing the rule modules below populates the registry;
``ALL_RULES`` is the canonical aggregated list. There are no per-module
``*_RULES`` lists anymore — the registry replaces them.
"""

from __future__ import annotations

# Import every rule module to trigger @register_rule decoration. The
# import order determines the iteration order of ALL_RULES; we keep the
# completeness → structure → stoichiometry → ordering → class-specific →
# numeric-grounding order that prior commits established so downstream
# tests that check rule-id sequence stay green.
from eln_structurer.rules import (  # noqa: F401
    class_specific,
    completeness,
    numeric_grounding,
    ordering,
    stoichiometry,
    structure,
)
from eln_structurer.rules.base import (
    Rule,
    RuleViolation,
    Severity,
    get_all_rules,
    register_rule,
    walk_compounds,
)

ALL_RULES: list[Rule] = get_all_rules()

__all__ = [
    "ALL_RULES",
    "Rule",
    "RuleViolation",
    "Severity",
    "register_rule",
    "walk_compounds",
]
