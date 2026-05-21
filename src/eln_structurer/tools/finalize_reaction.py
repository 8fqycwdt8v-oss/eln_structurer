"""finalize_reaction tool — emits the final ORD pbtxt and JSON.

The tool stores the finalized output in a ``ContextVar`` slot scoped to the
calling extract() task. This keeps concurrent extract() calls isolated
(unlike a module-level singleton) while still letting the agent runner pick
up the result after the SDK session ends. The agent itself sees only a
"FINALIZED" confirmation in the tool result.

The slot also carries cross-tool state that other validators consult:
- last_clean_signature: the JSON-canonical signature of the draft most
  recently validated as clean; ``finalize_reaction`` skips the redundant
  rule-pack re-run if the incoming draft matches.
- rule_history: a counter of rule_ids seen across repair iterations; used
  by ``validate_reaction`` to escalate the repair prompt when the same
  rule keeps firing.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
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
    # The signature of the most-recently-validated-clean draft. Lets
    # finalize_reaction skip the redundant run_harness pass.
    last_clean_signature: str | None = None
    # rule_id → number of times this rule has fired during the loop.
    # validate_reaction uses this to escalate the repair message.
    rule_history: Counter = field(default_factory=Counter)
    # iteration counter (validate_reaction calls)
    iterations: int = 0


_CURRENT_FINALIZED: ContextVar[FinalizedReaction | None] = ContextVar(
    "eln_structurer_finalized", default=None
)


def bind_finalized_slot(slot: FinalizedReaction):
    return _CURRENT_FINALIZED.set(slot)


def unbind_finalized_slot(token) -> None:
    _CURRENT_FINALIZED.reset(token)


def get_finalized() -> FinalizedReaction | None:
    return _CURRENT_FINALIZED.get()


def draft_signature(draft: ReactionDraft) -> str:
    """Stable hash for change detection — independent of dict ordering."""
    canonical = json.dumps(draft.model_dump(mode="json"), sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@tool(
    "finalize_reaction",
    (
        "Call this ONCE after validate_reaction reports clean. Pass the final draft "
        "as `draft_json`. The tool serializes to ORD pbtxt and JSON and stores the "
        "result. After calling this, end your response."
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

    # If the incoming draft is byte-identical to the one validate_reaction
    # just blessed as clean, we can trust that result and skip the full
    # rule-pack re-run. Saves a meaningful chunk of latency on every
    # successful extraction.
    sig = draft_signature(draft)
    trusted_clean = (
        slot.last_clean_signature is not None
        and slot.last_clean_signature == sig
    )

    if not trusted_clean:
        report = run_harness(draft)
        if not report.is_clean:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "REFUSED: validation still failing; call "
                            "validate_reaction and fix all errors before "
                            "finalize_reaction.\n\n" + report.as_repair_prompt()
                        ),
                    }
                ],
                "isError": True,
            }
        slot.validation_summary = report.to_dict()
    # else: slot.validation_summary already holds the previous clean report

    reaction_pb = draft_to_proto(draft)
    slot.draft = draft
    slot.pbtxt = serialize_reaction(reaction_pb, fmt="pbtxt")
    slot.json_text = serialize_reaction(reaction_pb, fmt="json")

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
