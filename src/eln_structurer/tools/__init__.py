"""In-process MCP tools exposed to the Anthropic Agent SDK.

Public surface:
- ``@tool``-decorated handlers used by ``agent.extract``:
  validate_reaction, validate_smiles, finalize_reaction, compute_mw,
  expand_abbreviation
- pure-Python core functions for callers that want the validator logic
  without the SDK marshaling.

Internal slot-binding helpers used by ``agent.extract`` are NOT re-exported;
import them from ``eln_structurer.tools.finalize_reaction`` directly.
"""

from eln_structurer.tools.compute_mw import MwResult, compute_mw, compute_mw_from_smiles
from eln_structurer.tools.detect_reaction_class import (
    ClassifyResult,
    classify_from_payload,
    detect_reaction_class,
)
from eln_structurer.tools.expand_abbreviation import (
    AbbreviationLookup,
    expand_abbreviation,
    lookup_abbreviation,
)
from eln_structurer.tools.finalize_reaction import finalize_reaction
from eln_structurer.tools.validate_reaction import (
    DraftValidation,
    validate_draft_payload,
    validate_reaction,
)
from eln_structurer.tools.validate_smiles import (
    SmilesCheck,
    check_smiles,
    validate_smiles,
)

__all__ = [
    # SDK tool handlers
    "validate_reaction",
    "validate_smiles",
    "finalize_reaction",
    "compute_mw",
    "expand_abbreviation",
    "detect_reaction_class",
    # Pure core functions + result types
    "check_smiles",
    "SmilesCheck",
    "validate_draft_payload",
    "DraftValidation",
    "compute_mw_from_smiles",
    "MwResult",
    "lookup_abbreviation",
    "AbbreviationLookup",
    "classify_from_payload",
    "ClassifyResult",
]
