"""Agent SDK wiring: drives Claude through the draft → validate → repair loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    TextBlock,
    create_sdk_mcp_server,
)

from eln_structurer.preprocess import normalize_paragraph
from eln_structurer.prompts import USER_PROMPT_TEMPLATE, build_system_prompt
from eln_structurer.tools import (
    compute_mw,
    expand_abbreviation,
    finalize_reaction,
    validate_reaction,
    validate_smiles,
)
from eln_structurer.tools.finalize_reaction import (
    FinalizedReaction,
    bind_finalized_slot,
    unbind_finalized_slot,
)


# Default model is Sonnet 4.6 — Sonnet's quality is close to Opus on this
# task at a fraction of the cost. Users wanting maximum quality pass
# ``--quality`` (CLI) or model="claude-opus-4-7" (programmatic).
DEFAULT_MODEL = "claude-sonnet-4-6"
HIGH_QUALITY_MODEL = "claude-opus-4-7"


@dataclass
class ExtractResult:
    success: bool
    pbtxt: str
    json_text: str
    validation_summary: dict
    transcript: list[str]
    # When success=False, a structured account of what's still wrong so the
    # caller can render a useful message instead of "extraction failed".
    failure_summary: dict[str, Any] = field(default_factory=dict)


def _build_failure_summary(slot: FinalizedReaction) -> dict[str, Any]:
    return {
        "iterations": slot.iterations,
        "rule_history": dict(slot.rule_history),
        "last_validation_summary": slot.validation_summary,
        "explanation": (
            "Agent did not converge to a clean draft within the iteration "
            "budget. The most-frequent rule failures and the last validation "
            "report are included so you can intervene manually."
        ),
    }


async def extract(
    paragraph: str,
    *,
    model: str = DEFAULT_MODEL,
    max_iters: int = 5,
    debug: bool = False,
) -> ExtractResult:
    """Run the agent against a single paragraph, returning the finalized output."""
    normalization = normalize_paragraph(paragraph)
    finalized = FinalizedReaction()
    token = bind_finalized_slot(finalized)
    transcript: list[str] = []
    try:
        server = create_sdk_mcp_server(
            name="eln",
            version="0.1.0",
            tools=[
                validate_reaction,
                validate_smiles,
                finalize_reaction,
                compute_mw,
                expand_abbreviation,
            ],
        )

        options = ClaudeAgentOptions(
            model=model,
            mcp_servers={"eln": server},
            allowed_tools=[
                "mcp__eln__validate_reaction",
                "mcp__eln__validate_smiles",
                "mcp__eln__finalize_reaction",
                "mcp__eln__compute_mw",
                "mcp__eln__expand_abbreviation",
            ],
            system_prompt=build_system_prompt(),
            # Three turns per repair iteration (think + tool + think).
            max_turns=max_iters * 3,
        )

        async with ClaudeSDKClient(options=options) as client:
            await client.query(
                USER_PROMPT_TEMPLATE.format(paragraph=normalization.normalized)
            )
            async for message in client.receive_response():
                if isinstance(message, AssistantMessage) and debug:
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            transcript.append(block.text)
    finally:
        unbind_finalized_slot(token)

    success = bool(finalized.pbtxt)
    return ExtractResult(
        success=success,
        pbtxt=finalized.pbtxt,
        json_text=finalized.json_text,
        validation_summary=finalized.validation_summary,
        transcript=transcript,
        failure_summary={} if success else _build_failure_summary(finalized),
    )
