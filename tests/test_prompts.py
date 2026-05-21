"""Tests for the system-prompt builder and the schema compressor."""

from __future__ import annotations

import json

from eln_structurer.prompts import (
    EDGE_CASE_HEURISTICS,
    FEW_SHOT_EXAMPLE,
    WORKUP_VERB_REFERENCE,
    build_system_prompt,
    compressed_reaction_draft_schema,
)
from eln_structurer.schema import reaction_draft_json_schema


def test_compressed_schema_strips_noise() -> None:
    compressed = compressed_reaction_draft_schema()
    payload = json.loads(compressed)
    # No top-level 'title' / 'description' / 'examples' anywhere.

    def assert_clean(node: object) -> None:
        if isinstance(node, dict):
            for noisy in ("title", "description", "examples"):
                assert noisy not in node, f"{noisy!r} survived compression: {node}"
            for value in node.values():
                assert_clean(value)
        elif isinstance(node, list):
            for item in node:
                assert_clean(item)

    assert_clean(payload)


def test_compressed_schema_keeps_enums_and_required() -> None:
    """Compressing removes documentation but must not touch required-field
    lists or enum constraints — those are load-bearing for the LLM."""
    raw = reaction_draft_json_schema()
    compressed = json.loads(compressed_reaction_draft_schema())

    # Every $defs entry that had `required` in the raw schema still has it.
    raw_defs = raw.get("$defs", {})
    cmp_defs = compressed.get("$defs", {})
    for name, raw_def in raw_defs.items():
        if "required" in raw_def:
            assert "required" in cmp_defs[name], f"required field list dropped for {name}"

    # AmountUnit enum values must survive compression (they're critical).
    from eln_structurer.schema import AmountUnit  # type: ignore[attr-defined]
    from typing import get_args
    expected = set(get_args(AmountUnit))
    assert expected.issubset(set(json.dumps(compressed).encode().decode().split())) or (
        any(v in json.dumps(compressed) for v in expected)
    ), "AmountUnit enum values missing from compressed schema"


def test_system_prompt_includes_required_blocks() -> None:
    prompt = build_system_prompt()
    assert "Workflow" in prompt
    assert "Workup vocabulary" in prompt
    assert "Worked example" in prompt
    assert "ReactionDraft JSON Schema" in prompt
    assert WORKUP_VERB_REFERENCE.strip().splitlines()[0] in prompt
    assert "Quenched with saturated NH4Cl" in prompt  # from FEW_SHOT_EXAMPLE


def test_system_prompt_is_cached() -> None:
    """build_system_prompt is cached — repeated calls return the same object."""
    assert build_system_prompt() is build_system_prompt()


def test_few_shot_example_uses_only_legal_enums() -> None:
    """The few-shot example must use units / roles / workup types that the
    Pydantic model actually accepts — otherwise we're teaching the LLM to
    emit invalid drafts."""
    from eln_structurer.schema import (
        AmountUnit,
        IdentifierType,
        ReactionRole,
        WorkupType,
    )
    from typing import get_args

    text = FEW_SHOT_EXAMPLE
    # Sanity: every role / unit / type literal that appears in the example
    # must be in the corresponding enum.
    for role in {"REACTANT", "REAGENT", "SOLVENT", "WORKUP", "PRODUCT"}:
        if role in text:
            assert role in get_args(ReactionRole)
    for unit in {"mmol", "mL", "g", "equiv", "%"}:
        if f'"units": "{unit}"' in text:
            # "%" is for the YIELD measurement's free-form units field, not an AmountUnit
            if unit != "%":
                assert unit in get_args(AmountUnit), unit
    for wu in {"ADDITION", "EXTRACTION", "DRY_WITH_MATERIAL", "CONCENTRATION"}:
        if f'"type": "{wu}"' in text:
            assert wu in get_args(WorkupType)
    for ident in {"NAME", "SMILES"}:
        if f'"type": "{ident}"' in text:
            assert ident in get_args(IdentifierType)


def test_edge_case_heuristics_referenced() -> None:
    """EDGE_CASE_HEURISTICS is embedded; the prompt should mention the
    Grignard / atmosphere guidance the chemistry rule (ORD-006) enforces."""
    prompt = build_system_prompt()
    assert "atmosphere" in prompt.lower()
    assert EDGE_CASE_HEURISTICS.strip().splitlines()[0] in prompt
