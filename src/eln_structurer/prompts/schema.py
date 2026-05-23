"""ReactionDraft JSON-schema factory + LLM-facing compression.

Pulled out of the prompts package's __init__ so callers (critic, agent)
can import the compressed schema without dragging in the full template
machinery.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any


# Pydantic's model_json_schema() output carries metadata the LLM doesn't
# need: per-field title/description duplication, top-level title,
# default values that match the type signature, etc. The compressor strips
# this noise while preserving the structural information that actually
# guides the agent (types, enums, required lists, $defs structure).
_NOISE_KEYS = {"title", "description", "examples"}


def _compress_schema_node(node: Any) -> Any:
    """Recursively remove documentation-only keys from a JSON-schema node."""
    if isinstance(node, dict):
        return {
            k: _compress_schema_node(v)
            for k, v in node.items()
            if k not in _NOISE_KEYS
        }
    if isinstance(node, list):
        return [_compress_schema_node(x) for x in node]
    return node


@lru_cache(maxsize=1)
def reaction_draft_json_schema() -> dict:
    """Return the raw Pydantic-generated JSON schema for ReactionDraft."""
    from eln_structurer.schema import ReactionDraft

    return ReactionDraft.model_json_schema()


@lru_cache(maxsize=1)
def compressed_reaction_draft_schema() -> str:
    """Return a stripped JSON-Schema string suitable for the system prompt.

    Same field/type/enum content as ``reaction_draft_json_schema()``; just
    without the documentation noise. Cached so repeated calls return the
    same string.
    """
    compressed = _compress_schema_node(reaction_draft_json_schema())
    return json.dumps(compressed, indent=2)
