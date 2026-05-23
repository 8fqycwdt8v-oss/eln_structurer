"""Pure-Python core functions and result types.

Separated from ``tools/__init__.py`` (which now contains ONLY the
SDK ``@tool`` handlers) so callers can pick the layer they need:

    from eln_structurer.tools import validate_reaction   # SDK handler
    from eln_structurer.tools.core import validate_draft_payload, check_smiles
"""

from __future__ import annotations

from eln_structurer.tools.compute_mw import MwResult, compute_mw_from_smiles
from eln_structurer.tools.detect_reaction_class import (
    ClassifyResult,
    classify_from_payload,
)
from eln_structurer.tools.expand_abbreviation import (
    AbbreviationLookup,
    lookup_abbreviation,
)
from eln_structurer.tools.validate_reaction import (
    DraftValidation,
    validate_draft_payload,
)
from eln_structurer.tools.validate_smiles import SmilesCheck, check_smiles
from eln_structurer.tools.verify_quote import QuoteCheck, verify_quote_against

__all__ = [
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
    "verify_quote_against",
    "QuoteCheck",
]
