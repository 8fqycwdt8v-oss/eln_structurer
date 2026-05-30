"""SDK tools that expose Tier-1 predictor primitives to the agent.

Three handlers:

- ``retrieve_exact_reaction`` — same-reaction channel (channel C).
- ``retrieve_similar_reactions`` — K-NN over the corpus (channel D).
- ``safety_screen`` — layered safety check before any protocol output.

All three operate against a process-global ``LocalCorpus`` set up at
package import time by ``bootstrap_default_corpus``. Tests can swap the
corpus by calling ``set_active_corpus(...)``; production callers will
typically pre-seed the global corpus before constructing the agent.
"""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from claude_agent_sdk import tool

from eln_structurer.predict import (
    LocalCorpus,
    hard_constraint_filter,
    retrieve_exact,
    retrieve_knn,
    safety_screen,
)
from eln_structurer.schema import ReactionDraft


# Process-global corpus. ``None`` means no corpus is bound; the tools
# return empty result sets with an explanatory note. Production callers
# inject a populated corpus before the agent runs.
_ACTIVE_CORPUS: LocalCorpus | None = None


def set_active_corpus(corpus: LocalCorpus | None) -> None:
    """Bind a corpus for the predict tools to query."""
    global _ACTIVE_CORPUS
    _ACTIVE_CORPUS = corpus


def get_active_corpus() -> LocalCorpus | None:
    return _ACTIVE_CORPUS


# --- retrieval ----------------------------------------------------------


def _format_hit(hit) -> dict[str, Any]:
    return {
        "reaction_smiles": hit.record.reaction_smiles,
        "similarity": round(hit.similarity, 3),
        "source": hit.record.source.value,
        "source_id": hit.record.source_id,
        "year": hit.record.year,
        "conditions": hit.record.conditions,
        "yield_percent": hit.record.yield_percent,
    }


@tool(
    "retrieve_exact_reaction",
    (
        "Look up records whose canonical reaction SMILES matches the "
        "target exactly. Returns every matching record across all "
        "sources (literature, ORD, HTE, industrial). Multiple matches "
        "are normal — different labs run the same reaction under "
        "different conditions. Use this BEFORE calling "
        "retrieve_similar_reactions; an exact match is the strongest "
        "possible retrieval signal."
    ),
    {"reaction_smiles": str},
)
async def retrieve_exact_reaction(args: dict[str, Any]) -> dict[str, Any]:
    smi = args.get("reaction_smiles", "")
    if not isinstance(smi, str) or not smi.strip():
        return {
            "content": [{"type": "text", "text": "ERROR: reaction_smiles must be a non-empty string."}],
            "isError": True,
        }
    if _ACTIVE_CORPUS is None:
        return {
            "content": [{"type": "text", "text": "NO_CORPUS: no reaction corpus is currently bound; retrieve_exact returned 0 hits."}]
        }
    hits = retrieve_exact(_ACTIVE_CORPUS, smi)
    if not hits:
        return {"content": [{"type": "text", "text": "0 exact matches."}]}
    return {
        "content": [
            {"type": "text", "text": f"{len(hits)} exact match(es):\n"
             + "\n".join(f"- {h['source']}:{h['source_id']} ({h['year']}) "
                         f"solvents={h['conditions'].get('solvents', [])} "
                         f"yield={h['yield_percent']}"
                         for h in (_format_hit(h) for h in hits))}
        ]
    }


@tool(
    "retrieve_similar_reactions",
    (
        "K-nearest-neighbour retrieval over the reaction corpus by "
        "Morgan-difference reaction fingerprint (Tanimoto similarity). "
        "Use this to find chemically related precedents when no exact "
        "match exists. Pass `constraints` to apply hard filters at "
        "query time — recognised keys: `no_halogenated_solvents` "
        "(bool), `min_year` (int), `allowed_sources` (list of "
        "'literature'|'ord'|'hte'|'industrial')."
    ),
    {
        "reaction_smiles": str,
        "k": int,
        "constraints": dict,
    },
)
async def retrieve_similar_reactions(args: dict[str, Any]) -> dict[str, Any]:
    smi = args.get("reaction_smiles", "")
    k = int(args.get("k", 5))
    constraints = args.get("constraints") or {}
    if not isinstance(smi, str) or not smi.strip():
        return {
            "content": [{"type": "text", "text": "ERROR: reaction_smiles must be a non-empty string."}],
            "isError": True,
        }
    if _ACTIVE_CORPUS is None:
        return {
            "content": [{"type": "text", "text": "NO_CORPUS: no reaction corpus is currently bound."}]
        }
    filters = hard_constraint_filter(constraints if isinstance(constraints, dict) else None)
    hits = retrieve_knn(_ACTIVE_CORPUS, smi, k=k, filters=filters)
    if not hits:
        return {"content": [{"type": "text", "text": "0 nearest neighbours found (constraints may be too tight)."}]}
    rendered = "\n".join(
        f"- sim={h['similarity']:.2f} {h['source']}:{h['source_id']} ({h['year']}) "
        f"solvents={h['conditions'].get('solvents', [])} yield={h['yield_percent']}"
        for h in (_format_hit(h) for h in hits)
    )
    return {"content": [{"type": "text", "text": f"{len(hits)} neighbour(s):\n{rendered}"}]}


# --- safety screen -----------------------------------------------------


@tool(
    "safety_screen",
    (
        "Run the layered safety screen against a draft (Pydantic "
        "ReactionDraft JSON shape). Returns BLOCKED when a controlled "
        "chemical is named, WARN for high-risk substructures or "
        "peroxide-former solvents, OK otherwise. ALWAYS call this "
        "before emitting a final proposed protocol — BLOCKED outputs "
        "must not be finalized."
    ),
    {"draft_json": dict},
)
async def safety_screen_tool(args: dict[str, Any]) -> dict[str, Any]:
    raw = args.get("draft_json")
    if raw is None:
        return {
            "content": [{"type": "text", "text": "ERROR: missing required argument `draft_json`."}],
            "isError": True,
        }
    try:
        draft = ReactionDraft.model_validate(raw)
    except ValidationError as exc:
        return {
            "content": [{"type": "text", "text": f"SCHEMA ERROR: {exc}"}],
            "isError": True,
        }
    report = safety_screen(draft)
    text = f"verdict: {report.verdict.value}"
    if report.flags:
        text += "\n" + "\n".join(f"  - {f}" for f in report.flags)
    return {
        "content": [{"type": "text", "text": text}],
        "isError": report.verdict.value == "blocked",
    }


__all__ = [
    "retrieve_exact_reaction",
    "retrieve_similar_reactions",
    "safety_screen_tool",
    "set_active_corpus",
    "get_active_corpus",
]
