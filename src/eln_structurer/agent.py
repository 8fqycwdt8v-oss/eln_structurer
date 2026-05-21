"""Agent SDK wiring: drives Claude through the draft → validate → repair loop."""

from __future__ import annotations

from dataclasses import dataclass

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    TextBlock,
    create_sdk_mcp_server,
)

from eln_structurer.prompts import USER_PROMPT_TEMPLATE, build_system_prompt
from eln_structurer.tools import (
    clear_finalized,
    finalize_reaction,
    get_finalized,
    validate_reaction,
    validate_smiles,
)


DEFAULT_MODEL = "claude-opus-4-7"


@dataclass
class ExtractResult:
    success: bool
    pbtxt: str
    json_text: str
    validation_summary: dict
    transcript: list[str]


async def extract(
    paragraph: str,
    *,
    model: str = DEFAULT_MODEL,
    max_iters: int = 5,
    debug: bool = False,
) -> ExtractResult:
    """Run the agent against a single paragraph, returning the finalized output."""
    clear_finalized()
    transcript: list[str] = []

    server = create_sdk_mcp_server(
        name="eln",
        version="0.1.0",
        tools=[validate_reaction, validate_smiles, finalize_reaction],
    )

    options = ClaudeAgentOptions(
        model=model,
        mcp_servers={"eln": server},
        allowed_tools=[
            "mcp__eln__validate_reaction",
            "mcp__eln__validate_smiles",
            "mcp__eln__finalize_reaction",
        ],
        system_prompt=build_system_prompt(),
        # Allow up to three turns per repair iteration (think + tool + think).
        max_turns=max_iters * 3,
    )

    async with ClaudeSDKClient(options=options) as client:
        await client.query(USER_PROMPT_TEMPLATE.format(paragraph=paragraph))
        async for message in client.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        if debug:
                            transcript.append(block.text)
            # ResultMessage signals end of session — receive_response yields it last.

    finalized = get_finalized()
    return ExtractResult(
        success=bool(finalized.pbtxt),
        pbtxt=finalized.pbtxt,
        json_text=finalized.json_text,
        validation_summary=finalized.validation_summary,
        transcript=transcript,
    )
