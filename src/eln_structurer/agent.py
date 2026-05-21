"""Agent SDK wiring: drives Claude through the draft → validate → repair loop.

The pipeline is:

    paragraph
       │
       ▼
    normalize → build system prompt
       │
       ▼
    Primary agent loop  (ClaudeSDKClient + 5 MCP tools)
       │  iterates validate_reaction / fix until clean,
       │  then finalize_reaction.
       ▼
    [optional] Critic subagent  (fresh query(), no tools)
       │  reads the original paragraph + the finalized JSON,
       │  emits structured findings.
       ▼
    [if critic found errors] one more primary-agent round
       with critic feedback prepended to the user prompt.
       ▼
    return ExtractResult
"""

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

from eln_structurer.critic import CriticReport, critic_available, run_critic
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
    critic_findings: list[dict] = field(default_factory=list)
    failure_summary: dict[str, Any] = field(default_factory=dict)


def _build_failure_summary(
    slot: FinalizedReaction, critic: CriticReport | None
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "iterations": slot.iterations,
        "rule_history": dict(slot.rule_history),
        "last_validation_summary": slot.validation_summary,
        "explanation": (
            "Agent did not converge to a clean draft within the iteration "
            "budget. The most-frequent rule failures and the last validation "
            "report are included so you can intervene manually."
        ),
    }
    if critic is not None:
        summary["critic_findings"] = [
            {"path": f.path, "severity": f.severity, "message": f.message}
            for f in critic.findings
        ]
    return summary


def _critic_findings_dicts(critic: CriticReport | None) -> list[dict]:
    if critic is None:
        return []
    return [
        {"path": f.path, "severity": f.severity, "message": f.message}
        for f in critic.findings
    ]


async def _run_primary_loop(
    user_prompt: str,
    *,
    model: str,
    max_turns: int,
    debug: bool,
    transcript: list[str],
) -> None:
    """Run one pass of the primary tool-using agent against the user prompt."""
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
        max_turns=max_turns,
    )
    async with ClaudeSDKClient(options=options) as client:
        await client.query(user_prompt)
        async for message in client.receive_response():
            if isinstance(message, AssistantMessage) and debug:
                for block in message.content:
                    if isinstance(block, TextBlock):
                        transcript.append(block.text)


async def extract(
    paragraph: str,
    *,
    model: str = DEFAULT_MODEL,
    max_iters: int = 5,
    debug: bool = False,
    enable_critic: bool = True,
) -> ExtractResult:
    """Run the agent against a single paragraph, returning the finalized output.

    With ``enable_critic=True`` (default) a fresh critic LLM call follows
    a successful primary pass; if the critic finds issues, one extra
    primary-agent round runs with the critic's feedback included.
    """
    normalization = normalize_paragraph(paragraph)
    finalized = FinalizedReaction()
    token = bind_finalized_slot(finalized)
    transcript: list[str] = []
    critic_report: CriticReport | None = None

    try:
        user_prompt = USER_PROMPT_TEMPLATE.format(paragraph=normalization.normalized)
        await _run_primary_loop(
            user_prompt,
            model=model,
            max_turns=max_iters * 3,
            debug=debug,
            transcript=transcript,
        )

        # If the primary loop produced a clean output and the critic is
        # available, run a single review pass.
        if (
            finalized.pbtxt
            and enable_critic
            and critic_available()
        ):
            critic_report = await run_critic(
                paragraph=normalization.original,
                draft_json=finalized.json_text,
                model=model,
            )
            # If the critic flagged ERRORs, run one more primary round with
            # the findings prepended to the original prompt.
            if not critic_report.is_clean and any(
                f.severity == "ERROR" for f in critic_report.findings
            ):
                # Reset finalized state for the revision round; if the
                # revision fails we'll know it via slot.pbtxt being empty.
                prior_pbtxt = finalized.pbtxt
                finalized.pbtxt = ""
                finalized.json_text = ""
                revision_prompt = (
                    critic_report.as_repair_prompt()
                    + "\n\n"
                    + user_prompt
                )
                await _run_primary_loop(
                    revision_prompt,
                    model=model,
                    max_turns=max_iters * 3,
                    debug=debug,
                    transcript=transcript,
                )
                # If the revision didn't produce a finalized output, fall
                # back to the pre-critic output so we don't silently lose
                # a valid extraction.
                if not finalized.pbtxt:
                    finalized.pbtxt = prior_pbtxt
                    # json_text was cleared too; the slot already has a
                    # validation_summary, so just refill the JSON below.
                    from eln_structurer.proto_bridge import (
                        draft_to_proto,
                        serialize_reaction,
                    )
                    if finalized.draft is not None:
                        proto = draft_to_proto(finalized.draft)
                        finalized.json_text = serialize_reaction(proto, fmt="json")
    finally:
        unbind_finalized_slot(token)

    success = bool(finalized.pbtxt)
    return ExtractResult(
        success=success,
        pbtxt=finalized.pbtxt,
        json_text=finalized.json_text,
        validation_summary=finalized.validation_summary,
        transcript=transcript,
        critic_findings=_critic_findings_dicts(critic_report),
        failure_summary={} if success else _build_failure_summary(finalized, critic_report),
    )
