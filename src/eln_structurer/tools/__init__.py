"""In-process MCP tools exposed to the Anthropic Agent SDK.

Public surface:
- the three ``@tool``-decorated handlers used by ``agent.extract``
- their pure-Python counterparts (``check_smiles`` / ``validate_draft_payload``)
  for callers that want the validator logic without the SDK marshaling.

Internal slot-binding helpers used by ``agent.extract`` are NOT re-exported;
import them from ``eln_structurer.tools.finalize_reaction`` directly.
"""

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
    # Pure core functions + result types
    "check_smiles",
    "SmilesCheck",
    "validate_draft_payload",
    "DraftValidation",
]
