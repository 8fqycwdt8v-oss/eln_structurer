"""detect_reaction_class — expose the heuristic classifier as an MCP tool.

Lets the agent ask "what reaction class did I just extract?" instead of
inferring it from the paragraph text. The classifier is the same one the
rule pack uses internally — calling this tool gives the agent the same
view the rules will have when they run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from claude_agent_sdk import tool

from eln_structurer.reaction_class import classify_reaction
from eln_structurer.schema import ReactionDraft


@dataclass(frozen=True)
class ClassifyResult:
    cls: str
    confidence: float
    rationale: str

    @classmethod
    def from_draft(cls, draft: ReactionDraft) -> "ClassifyResult":
        result = classify_reaction(draft)
        return cls(
            cls=result.cls.value,
            confidence=result.confidence,
            rationale=result.rationale,
        )


def classify_from_payload(payload: Any) -> ClassifyResult | str:
    """Pure-core entry point. Returns ClassifyResult or an error string."""
    if not isinstance(payload, dict):
        return "draft_json must be a JSON object"
    try:
        draft = ReactionDraft.model_validate(payload)
    except ValidationError as exc:
        return f"schema error: {exc}"
    return ClassifyResult.from_draft(draft)


@tool(
    "detect_reaction_class",
    (
        "Classify the current draft into a reaction class (Suzuki coupling, "
        "amide formation, Grignard, reduction, etc.) using the same "
        "heuristic the rule pack uses internally. Call this BEFORE finalize "
        "to confirm the extraction matches the paragraph's reaction type. "
        "Returns the class name, a 0-1 confidence, and the rationale. "
        "Returns UNKNOWN if no class pattern matches — that's a valid answer."
    ),
    {"draft_json": dict},
)
async def detect_reaction_class(args: dict[str, Any]) -> dict[str, Any]:
    raw = args.get("draft_json")
    if raw is None:
        return {
            "content": [
                {"type": "text", "text": "ERROR: missing required argument `draft_json`."}
            ],
            "isError": True,
        }
    result = classify_from_payload(raw)
    if isinstance(result, str):
        return {
            "content": [{"type": "text", "text": f"ERROR: {result}"}],
            "isError": True,
        }
    return {
        "content": [
            {
                "type": "text",
                "text": (
                    f"class: {result.cls}\n"
                    f"confidence: {result.confidence:.2f}\n"
                    f"rationale: {result.rationale}"
                ),
            }
        ]
    }
