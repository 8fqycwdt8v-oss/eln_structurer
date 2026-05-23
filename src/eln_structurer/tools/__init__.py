"""In-process MCP tools exposed to the Anthropic Agent SDK.

Public surface = the SDK ``@tool``-decorated handlers used by
``agent.extract``. Pure-Python helpers live in ``eln_structurer.tools.core``.
Internal slot-binding helpers used by ``agent.extract`` are not re-exported;
import them from ``eln_structurer.tools.finalize_reaction`` directly.
"""

from eln_structurer.tools.compute_mw import compute_mw
from eln_structurer.tools.detect_reaction_class import detect_reaction_class
from eln_structurer.tools.expand_abbreviation import expand_abbreviation
from eln_structurer.tools.finalize_reaction import finalize_reaction
from eln_structurer.tools.validate_reaction import validate_reaction
from eln_structurer.tools.validate_smiles import validate_smiles
from eln_structurer.tools.verify_quote import verify_quote

__all__ = [
    "validate_reaction",
    "validate_smiles",
    "finalize_reaction",
    "compute_mw",
    "expand_abbreviation",
    "detect_reaction_class",
    "verify_quote",
]
