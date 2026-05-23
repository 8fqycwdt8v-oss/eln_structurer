"""Prompts package.

Public surface:
- ``build_system_prompt()`` — cached system prompt assembly for the agent loop.
- ``USER_PROMPT_TEMPLATE`` — format-string for the per-extraction user message.
- ``WORKUP_VERB_REFERENCE``, ``EDGE_CASE_HEURISTICS``, ``FEW_SHOT_EXAMPLE`` —
  reusable blocks for callers (critic, naive-LLM adapter) that compose their
  own prompts and want parity with the primary system prompt.
- ``compressed_reaction_draft_schema`` / ``reaction_draft_json_schema`` —
  schema helpers used by the prompt assembler AND the critic.
"""

from __future__ import annotations

from functools import lru_cache

from eln_structurer.prompts.examples import (
    EDGE_CASE_HEURISTICS,
    FEW_SHOT_EXAMPLE,
    WORKUP_VERB_REFERENCE,
)
from eln_structurer.prompts.schema import (
    compressed_reaction_draft_schema,
    reaction_draft_json_schema,
)
from eln_structurer.prompts.template import SYSTEM_PROMPT_TEMPLATE, USER_PROMPT_TEMPLATE


@lru_cache(maxsize=1)
def build_system_prompt() -> str:
    """Build the system prompt. Cached because every embedded block is static."""
    return SYSTEM_PROMPT_TEMPLATE.format(
        schema_json=compressed_reaction_draft_schema(),
        workup_verbs=WORKUP_VERB_REFERENCE,
        heuristics=EDGE_CASE_HEURISTICS,
        example=FEW_SHOT_EXAMPLE,
    )


__all__ = [
    "build_system_prompt",
    "USER_PROMPT_TEMPLATE",
    "SYSTEM_PROMPT_TEMPLATE",
    "WORKUP_VERB_REFERENCE",
    "EDGE_CASE_HEURISTICS",
    "FEW_SHOT_EXAMPLE",
    "compressed_reaction_draft_schema",
    "reaction_draft_json_schema",
]
