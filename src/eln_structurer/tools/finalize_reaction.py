"""finalize_reaction tool — emits the final ORD pbtxt and JSON.

The tool stores the finalized output in a module-level slot that the calling
agent runner picks up after the session ends. The agent itself sees only a
short "FINALIZED" confirmation; the CLI is responsible for actually writing
the bytes to disk or stdout.
"""

from __future__ import annotations

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


FINALIZED_REACTION = FinalizedReaction()


def get_finalized() -> FinalizedReaction:
    return FINALIZED_REACTION


def clear_finalized() -> None:
    FINALIZED_REACTION.draft = None
    FINALIZED_REACTION.pbtxt = ""
    FINALIZED_REACTION.json_text = ""
    FINALIZED_REACTION.validation_summary = {}


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

    reaction_pb = draft_to_proto(draft)
    FINALIZED_REACTION.draft = draft
    FINALIZED_REACTION.pbtxt = serialize_reaction(reaction_pb, fmt="pbtxt")
    FINALIZED_REACTION.json_text = serialize_reaction(reaction_pb, fmt="json")
    FINALIZED_REACTION.validation_summary = report.to_dict()

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
