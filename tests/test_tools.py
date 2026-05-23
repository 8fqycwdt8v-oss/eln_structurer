"""Direct unit tests of the MCP tool handlers.

The @tool decorator wraps each function into an ``SdkMcpTool``; the original
async function is exposed on ``.handler``. Tests invoke that directly so
they don't need the SDK runtime. ``asyncio_mode = "auto"`` in pyproject.toml
turns bare ``async def`` test functions into coroutines pytest-asyncio
schedules automatically — no per-test decorators needed.
"""

from __future__ import annotations

from eln_structurer.chemistry import smiles_of
from eln_structurer.schema import (
    CompoundIdentifierModel,
    CompoundModel,
    ReactionDraft,
)
from eln_structurer.tools import (
    compute_mw,
    expand_abbreviation,
    finalize_reaction,
    validate_reaction,
    validate_smiles,
)
from eln_structurer.tools.core import compute_mw_from_smiles, lookup_abbreviation
from eln_structurer.tools.finalize_reaction import (
    FinalizedReaction,
    bind_finalized_slot,
    unbind_finalized_slot,
)


async def test_validate_smiles_valid() -> None:
    result = await validate_smiles.handler({"smiles": "CCO"})
    assert result.get("isError") is not True
    text = result["content"][0]["text"]
    assert "VALID" in text and "Canonical" in text


async def test_validate_smiles_invalid() -> None:
    result = await validate_smiles.handler({"smiles": "this-is-not-smiles[[["})
    assert result.get("isError") is True
    assert "INVALID" in result["content"][0]["text"]


async def test_validate_smiles_empty_string() -> None:
    result = await validate_smiles.handler({"smiles": ""})
    assert result.get("isError") is True
    assert "non-empty" in result["content"][0]["text"]


async def test_validate_smiles_missing_arg() -> None:
    result = await validate_smiles.handler({})
    assert result.get("isError") is True


async def test_validate_reaction_passes_clean_draft(aspirin_draft: ReactionDraft) -> None:
    result = await validate_reaction.handler(
        {"draft_json": aspirin_draft.model_dump(mode="json")}
    )
    assert result.get("isError") is not True, result["content"][0]["text"]


async def test_validate_reaction_rejects_invalid_shape() -> None:
    result = await validate_reaction.handler({"draft_json": {"not": "a draft"}})
    assert result.get("isError") is True
    assert "SCHEMA ERROR" in result["content"][0]["text"]


async def test_validate_reaction_missing_arg() -> None:
    result = await validate_reaction.handler({})
    assert result.get("isError") is True


async def test_finalize_reaction_refuses_with_unclean_draft(
    aspirin_draft: ReactionDraft,
) -> None:
    # Break CMP-001 deliberately, then try to finalize inside a bound slot.
    for inp in aspirin_draft.inputs:
        for comp in inp.components:
            comp.reaction_role = "REAGENT"
    slot = FinalizedReaction()
    token = bind_finalized_slot(slot)
    try:
        result = await finalize_reaction.handler(
            {"draft_json": aspirin_draft.model_dump(mode="json")}
        )
    finally:
        unbind_finalized_slot(token)
    assert result.get("isError") is True
    assert "REFUSED" in result["content"][0]["text"]
    assert slot.pbtxt == ""


async def test_compute_mw_valid_smiles() -> None:
    result = await compute_mw.handler({"smiles": "c1ccccc1"})
    assert result.get("isError") is not True
    text = result["content"][0]["text"]
    # Benzene MW ≈ 78.11
    assert "78" in text
    assert "Heavy atoms: 6" in text


async def test_compute_mw_invalid_smiles() -> None:
    result = await compute_mw.handler({"smiles": "totally-invalid["})
    assert result.get("isError") is True
    assert "ERROR" in result["content"][0]["text"]


def test_compute_mw_from_smiles_pure() -> None:
    """The pure-core helper returns a typed result without SDK marshaling."""
    ok = compute_mw_from_smiles("CCO")  # ethanol, MW≈46
    assert ok.ok
    assert ok.heavy_atom_count == 3
    assert 45 < ok.mw_g_per_mol < 47

    bad = compute_mw_from_smiles("not[a]smiles")
    assert not bad.ok
    assert bad.error is not None


async def test_expand_abbreviation_known_token() -> None:
    result = await expand_abbreviation.handler({"token": "thf"})
    assert result.get("isError") is not True
    text = result["content"][0]["text"]
    assert "tetrahydrofuran" in text.lower()
    assert "66" in text  # bp


async def test_expand_abbreviation_solvent_only() -> None:
    """A solvent name not in the abbreviation dict but in the bp table
    still returns the bp."""
    result = await expand_abbreviation.handler({"token": "toluene"})
    assert result.get("isError") is not True
    assert "111" in result["content"][0]["text"]


async def test_expand_abbreviation_unknown_token() -> None:
    result = await expand_abbreviation.handler({"token": "absolutely_not_a_real_abbrev"})
    assert "UNKNOWN" in result["content"][0]["text"]


def test_lookup_abbreviation_pure() -> None:
    r = lookup_abbreviation("rt")
    assert r.expansion is not None
    assert "room temperature" in r.expansion.lower()
    assert r.bp_celsius is None


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
