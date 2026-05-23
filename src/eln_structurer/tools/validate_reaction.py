"""validate_reaction — the SDK tool driving the self-repair loop.

Pure-core ``validate_draft_payload`` returns a ``DraftValidation``; the
``@tool``-decorated handler marshals that into the SDK's content/isError
shape. The handler also bookkeeps the per-extract slot:

- counts iterations and rule_id repeats; when the same rule fires three
  times in a row, the repair-prompt gets a stronger directive appended.
- records the signature of a draft that was just validated as clean so
  ``finalize_reaction`` can skip the redundant rule-pack pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from claude_agent_sdk import tool

from eln_structurer.harness import ValidationReport, run_harness
from eln_structurer.schema import ReactionDraft
from eln_structurer.tools.finalize_reaction import (
    draft_signature,
    get_finalized,
)


from eln_structurer.config import DEFAULT_EXTRACTOR_CONFIG as _CFG

# Tunables read from config.DEFAULT_EXTRACTOR_CONFIG. Kept as module-level
# names so existing tests that patch them continue to work.
_DIVERGENCE_THRESHOLD = _CFG.divergence_threshold
_DEFAULT_ITER_BUDGET = _CFG.default_iter_budget


@dataclass(frozen=True)
class DraftValidation:
    report: ValidationReport | None
    parse_error: str | None

    @property
    def is_clean(self) -> bool:
        return self.report is not None and self.report.is_clean


def validate_draft_payload(payload: Any) -> DraftValidation:
    """Validate a raw JSON-shaped payload against the rule pack."""
    if not isinstance(payload, dict):
        return DraftValidation(report=None, parse_error="draft_json must be an object")
    try:
        draft = ReactionDraft.model_validate(payload)
    except ValidationError as exc:
        return DraftValidation(report=None, parse_error=str(exc))
    return DraftValidation(report=run_harness(draft), parse_error=None)


def _escalation_message(slot, rule_id: str, count: int) -> str:
    return (
        f"\n\n!!! ESCALATION: You have failed rule {rule_id} {count} times "
        "in a row. Re-read the original paragraph from scratch and rebuild "
        f"the affected field(s) without copying from your last attempt. "
        "If you cannot satisfy this rule with the data given, stop, do NOT "
        "call finalize_reaction, and explain in plain text why this rule "
        "cannot be satisfied."
    )


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
    assert result.report is not None
    report = result.report
    slot = get_finalized()
    if slot is not None:
        slot.iterations += 1
        if report.is_clean:
            # Cache signature so finalize_reaction can trust this verdict.
            try:
                draft = ReactionDraft.model_validate(raw)
                slot.last_clean_signature = draft_signature(draft)
                slot.validation_summary = report.to_dict()
            except ValidationError:  # pragma: no cover — already shape-validated
                pass
        else:
            for v in report.errors:
                slot.rule_history[v.rule_id] += 1

    text = report.as_repair_prompt()
    if slot is not None:
        # Tell the agent how much of the iteration budget it has used so it
        # can self-limit instead of looping silently until the SDK caps it.
        used = slot.iterations
        remaining = max(0, _DEFAULT_ITER_BUDGET - used)
        if not report.is_clean:
            if remaining == 0:
                text += (
                    "\n\n!!! BUDGET EXHAUSTED: This was iteration "
                    f"{used}/{_DEFAULT_ITER_BUDGET}. Do NOT call "
                    "validate_reaction again — explain in plain text which "
                    "errors remain and why they cannot be satisfied from "
                    "the paragraph alone, then stop. Do not call "
                    "finalize_reaction with an unclean draft."
                )
            elif remaining <= 2:
                text += (
                    f"\n\n[BUDGET WARNING: iteration {used} of "
                    f"{_DEFAULT_ITER_BUDGET}; only {remaining} more "
                    "validate→fix cycles before the agent must stop.]"
                )
            else:
                text += f"\n\n[iteration {used} of {_DEFAULT_ITER_BUDGET}]"

    if (
        slot is not None
        and not report.is_clean
        and slot.rule_history
        and (most_common := slot.rule_history.most_common(1))
        and most_common[0][1] >= _DIVERGENCE_THRESHOLD
        and most_common[0][0] in {v.rule_id for v in report.errors}
    ):
        text += _escalation_message(slot, most_common[0][0], most_common[0][1])

    return {
        "content": [{"type": "text", "text": text}],
        "isError": not report.is_clean,
    }
