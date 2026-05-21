"""Direct unit tests of the MCP tool handlers.

The @tool decorator wraps each function into an ``SdkMcpTool``; the original
async function is exposed on ``.handler``. Tests invoke that directly so
they don't need the SDK runtime.
"""

from __future__ import annotations

import asyncio

from eln_structurer.chemistry import smiles_of
from eln_structurer.schema import (
    CompoundIdentifierModel,
    CompoundModel,
    ReactionDraft,
)
from eln_structurer.tools import (
    finalize_reaction,
    validate_reaction,
    validate_smiles,
)
from eln_structurer.tools.finalize_reaction import (
    FinalizedReaction,
    bind_finalized_slot,
    unbind_finalized_slot,
)


def _run(coro):
    return asyncio.run(coro)


def test_validate_smiles_valid() -> None:
    result = _run(validate_smiles.handler({"smiles": "CCO"}))
    assert result.get("isError") is not True
    text = result["content"][0]["text"]
    assert "VALID" in text and "Canonical" in text


def test_validate_smiles_invalid() -> None:
    result = _run(validate_smiles.handler({"smiles": "this-is-not-smiles[[["}))
    assert result.get("isError") is True
    assert "INVALID" in result["content"][0]["text"]


def test_validate_smiles_empty_string() -> None:
    result = _run(validate_smiles.handler({"smiles": ""}))
    assert result.get("isError") is True
    assert "non-empty" in result["content"][0]["text"]


def test_validate_smiles_missing_arg() -> None:
    result = _run(validate_smiles.handler({}))
    assert result.get("isError") is True


def test_validate_reaction_passes_clean_draft(aspirin_draft: ReactionDraft) -> None:
    result = _run(
        validate_reaction.handler(
            {"draft_json": aspirin_draft.model_dump(mode="json")}
        )
    )
    # The aspirin fixture should be clean against the rule pack.
    assert result.get("isError") is not True, result["content"][0]["text"]


def test_validate_reaction_rejects_invalid_shape() -> None:
    result = _run(
        validate_reaction.handler({"draft_json": {"not": "a draft"}})
    )
    assert result.get("isError") is True
    assert "SCHEMA ERROR" in result["content"][0]["text"]


def test_validate_reaction_missing_arg() -> None:
    result = _run(validate_reaction.handler({}))
    assert result.get("isError") is True


def test_finalize_reaction_refuses_with_unclean_draft(aspirin_draft: ReactionDraft) -> None:
    # Break CMP-001 deliberately, then try to finalize inside a bound slot.
    for inp in aspirin_draft.inputs:
        for comp in inp.components:
            comp.reaction_role = "REAGENT"
    slot = FinalizedReaction()
    token = bind_finalized_slot(slot)
    try:
        result = _run(
            finalize_reaction.handler(
                {"draft_json": aspirin_draft.model_dump(mode="json")}
            )
        )
    finally:
        unbind_finalized_slot(token)
    assert result.get("isError") is True
    assert "REFUSED" in result["content"][0]["text"]
    assert slot.pbtxt == ""  # nothing was written


def test_smiles_of_returns_first_smiles() -> None:
    comp = CompoundModel(
        identifiers=[
            CompoundIdentifierModel(type="NAME", value="benzene"),
            CompoundIdentifierModel(type="SMILES", value="c1ccccc1"),
            CompoundIdentifierModel(type="SMILES", value="C1=CC=CC=C1"),
        ]
    )
    # smiles_of returns the FIRST SMILES, not the canonical one.
    assert smiles_of(comp) == "c1ccccc1"
