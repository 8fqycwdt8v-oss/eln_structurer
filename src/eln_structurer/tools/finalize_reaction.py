"""finalize_reaction tool — emits the final ORD pbtxt and JSON.

The tool stores the finalized output in a ``ContextVar`` slot scoped to the
calling extract() task. This keeps concurrent extract() calls isolated
(unlike a module-level singleton) while still letting the agent runner pick
up the result after the SDK session ends. The agent itself sees only a
"FINALIZED" confirmation in the tool result.
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from claude_agent_sdk import tool

from eln_structurer.harness import run_harness
from eln_structurer.proto_bridge import draft_to_proto, serialize_reaction
from eln_structurer.schema import ReactionDraft


@dataclass
class FinalizedReaction:
    draft: ReactionDraft | None = None
    pbtxt: str = ""
    json_text: str = ""
    validation_summary: dict = field(default_factory=dict)


# Per-task slot. Set by ``agent.extract`` before opening the SDK session,
# read back by ``agent.extract`` after the session closes. Concurrent
# extract() calls in the same process see independent slots because
# ContextVar copies its value across asyncio tasks.
_CURRENT_FINALIZED: ContextVar[FinalizedReaction | None] = ContextVar(
    "eln_structurer_finalized", default=None
)


def bind_finalized_slot(slot: FinalizedReaction):
    """Bind a fresh FinalizedReaction container to the current task.

    Returns the token that must be passed to ``unbind_finalized_slot`` so the
    previous value is restored when the extraction completes.
    """
    return _CURRENT_FINALIZED.set(slot)


def unbind_finalized_slot(token) -> None:
    _CURRENT_FINALIZED.reset(token)


def get_finalized() -> FinalizedReaction | None:
    """Return the current task's finalized slot, or None if no slot is bound."""
    return _CURRENT_FINALIZED.get()


@tool(
    "finalize_reaction",
    (
        "Call this ONCE after validate_reaction reports clean. Pass the final draft "
        "as `draft_json`. The tool runs one last validation, serializes to ORD pbtxt "
        "and JSON, and stores the result. After calling this, end your response."
    ),
    {"draft_json": dict},
)
async def finalize_reaction(args: dict[str, Any]) -> dict[str, Any]:
    raw = args.get("draft_json")
    if raw is None:
        return {
            "content": [
                {"type": "text", "text": "ERROR: missing required argument `draft_json`."}
            ],
            "isError": True,
        }
    try:
        draft = ReactionDraft.model_validate(raw)
    except ValidationError as exc:
        return {
            "content": [
                {
                    "type": "text",
                    "text": f"SCHEMA ERROR — finalize_reaction received invalid draft:\n{exc}",
                }
            ],
            "isError": True,
        }
    report = run_harness(draft)
    if not report.is_clean:
        return {
            "content": [
                {
                    "type": "text",
                    "text": (
                        "REFUSED: validation still failing; call validate_reaction "
                        "and fix all errors before finalize_reaction.\n\n"
                        + report.as_repair_prompt()
                    ),
                }
            ],
            "isError": True,
        }

    slot = _CURRENT_FINALIZED.get()
    if slot is None:
        return {
            "content": [
                {
                    "type": "text",
                    "text": (
                        "ERROR: finalize_reaction called outside an extract() "
                        "context. This is a tool wiring bug, not a draft problem."
                    ),
                }
            ],
            "isError": True,
        }

    reaction_pb = draft_to_proto(draft)
    slot.draft = draft
    slot.pbtxt = serialize_reaction(reaction_pb, fmt="pbtxt")
    slot.json_text = serialize_reaction(reaction_pb, fmt="json")
    slot.validation_summary = report.to_dict()

    return {
        "content": [
            {
                "type": "text",
                "text": (
                    "FINALIZED. Reaction has been serialized to ORD pbtxt and JSON. "
                    "End your response now."
                ),
            }
        ]
    }
