"""The validate_reaction tool — the heart of the self-repair loop."""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from claude_agent_sdk import tool

from eln_structurer.harness import run_harness
from eln_structurer.schema import ReactionDraft


@tool(
    "validate_reaction",
    (
        "Validate a draft Reaction (Pydantic JSON shape) against ord-schema and "
        "the local chemistry rule pack. Pass the entire draft as a JSON object in "
        "`draft_json`. Returns a human-readable report; if errors are present you "
        "MUST fix them and call this tool again. When clean, call finalize_reaction."
    ),
    {"draft_json": dict},
)
async def validate_reaction(args: dict[str, Any]) -> dict[str, Any]:
    raw = args.get("draft_json")
    if raw is None:
        return {
            "content": [
                {
                    "type": "text",
                    "text": "ERROR: missing required argument `draft_json`.",
                }
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
                    "text": (
                        "SCHEMA ERROR — draft_json does not match ReactionDraft shape.\n"
                        f"{exc}\n\n"
                        "Re-emit the draft conforming to the ReactionDraft JSON schema "
                        "shown in the system prompt."
                    ),
                }
            ],
            "isError": True,
        }

    report = run_harness(draft)
    return {
        "content": [{"type": "text", "text": report.as_repair_prompt()}],
        "isError": not report.is_clean,
    }
