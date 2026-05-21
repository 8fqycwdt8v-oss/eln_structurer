"""validate_reaction — the SDK tool driving the self-repair loop.

Pure core function ``validate_draft_payload`` runs the harness on a raw
JSON-shaped payload and returns a ``(report, parse_error)`` pair. The
``@tool``-decorated handler marshals that pair into the SDK's
``content``/``isError`` shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from claude_agent_sdk import tool

from eln_structurer.harness import ValidationReport, run_harness
from eln_structurer.schema import ReactionDraft


@dataclass(frozen=True)
class DraftValidation:
    report: ValidationReport | None
    parse_error: str | None

    @property
    def is_clean(self) -> bool:
        return self.report is not None and self.report.is_clean


def validate_draft_payload(payload: Any) -> DraftValidation:
    """Validate a raw JSON-shaped payload against the rule pack.

    Returns ``DraftValidation`` carrying either a ValidationReport (when the
    payload is shape-valid) or a parse_error string (when Pydantic rejects
    the input shape).
    """
    if not isinstance(payload, dict):
        return DraftValidation(report=None, parse_error="draft_json must be an object")
    try:
        draft = ReactionDraft.model_validate(payload)
    except ValidationError as exc:
        return DraftValidation(report=None, parse_error=str(exc))
    return DraftValidation(report=run_harness(draft), parse_error=None)


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
                {"type": "text", "text": "ERROR: missing required argument `draft_json`."}
            ],
            "isError": True,
        }
    result = validate_draft_payload(raw)
    if result.parse_error is not None:
        return {
            "content": [
                {
                    "type": "text",
                    "text": (
                        "SCHEMA ERROR — draft_json does not match ReactionDraft shape.\n"
                        f"{result.parse_error}\n\n"
                        "Re-emit the draft conforming to the ReactionDraft JSON schema "
                        "shown in the system prompt."
                    ),
                }
            ],
            "isError": True,
        }
    assert result.report is not None  # parse_error is None ⇒ report is set
    return {
        "content": [{"type": "text", "text": result.report.as_repair_prompt()}],
        "isError": not result.report.is_clean,
    }
