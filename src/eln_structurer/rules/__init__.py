"""Rule pack registry.

Importing this module exposes ``ALL_RULES`` — every Rule instance that the
harness runs against a ReactionDraft.
"""

from __future__ import annotations

from eln_structurer.rules.base import Rule, RuleViolation, Severity
from eln_structurer.rules.class_specific import CLS_RULES
from eln_structurer.rules.completeness import CMP_RULES
from eln_structurer.rules.numeric_grounding import NUM_RULES
from eln_structurer.rules.ordering import ORD_RULES
from eln_structurer.rules.stoichiometry import STO_RULES
from eln_structurer.rules.structure import STR_RULES

ALL_RULES: list[Rule] = [
    *CMP_RULES,
    *STR_RULES,
    *STO_RULES,
    *ORD_RULES,
    *CLS_RULES,
    *NUM_RULES,
]

__all__ = ["ALL_RULES", "Rule", "RuleViolation", "Severity"]
