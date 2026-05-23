"""Agent SDK wiring: drives Claude through the draft → validate → repair loop.

Pipeline:

    paragraph
       │
       ▼
    normalize (preprocess.py) → build system prompt
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
    [if critic found ERRORs] one more primary-agent round
       with critic feedback prepended to the user prompt.
       │
       ▼
    return ExtractResult (carries metrics for downstream observability)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    TextBlock,
    create_sdk_mcp_server,
)

from eln_structurer.critic import CriticReport, critic_available, run_critic
from eln_structurer.preprocess import normalize_paragraph
from eln_structurer.prompts import USER_PROMPT_TEMPLATE, build_system_prompt
from eln_structurer.tools import (
    compute_mw,
    detect_reaction_class,
    expand_abbreviation,
    finalize_reaction,
    validate_reaction,
    validate_smiles,
    verify_quote,
)
from eln_structurer.tools.finalize_reaction import FinalizedReaction, finalized_slot


DEFAULT_MODEL = "claude-sonnet-4-6"
HIGH_QUALITY_MODEL = "claude-opus-4-7"

# Backwards-compatible alias — ``config.DEFAULT_EXTRACTOR_CONFIG`` is the
# canonical source. Kept so existing imports keep working.
from eln_structurer.config import DEFAULT_EXTRACTOR_CONFIG as _CFG  # noqa: E402
MAX_PARAGRAPH_CHARS = _CFG.max_paragraph_chars


@dataclass
class UsageStats:
    """LLM usage and cost. Aggregated across the primary loop AND any
    critic / revision rounds in this extraction."""
    total_cost_usd: float = 0.0
    duration_ms: int = 0
    duration_api_ms: int = 0
    num_turns: int = 0
    usage_blob: dict[str, Any] = field(default_factory=dict)

    def merge(self, other: ResultMessage) -> None:
        if other.total_cost_usd is not None:
            self.total_cost_usd += other.total_cost_usd
        self.duration_ms += other.duration_ms
        self.duration_api_ms += other.duration_api_ms
        self.num_turns += other.num_turns
        if other.usage:
            # Sum token counts from concurrent calls when keys overlap.
            for k, v in other.usage.items():
                if isinstance(v, (int, float)):
                    self.usage_blob[k] = self.usage_blob.get(k, 0) + v
                else:
                    self.usage_blob.setdefault(k, v)


@dataclass
class ExtractResult:
    success: bool
    pbtxt: str
    json_text: str
    validation_summary: dict
    transcript: list[str]
    critic_findings: list[dict] = field(default_factory=list)
    failure_summary: dict[str, Any] = field(default_factory=dict)
    # Observability — populated even on success so callers can track cost
    # and convergence:
    iterations: int = 0
    rule_history: dict[str, int] = field(default_factory=dict)
    critic_ran: bool = False
    revision_triggered: bool = False
    usage: UsageStats = field(default_factory=UsageStats)


def _build_failure_summary(
    slot: FinalizedReaction, critic: CriticReport | None, reason: str
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "reason": reason,
        "iterations": slot.iterations,
        "rule_history": dict(slot.rule_history),
        "last_validation_summary": slot.validation_summary,
    }
    if critic is not None:
        summary["critic_findings"] = [
            {"path": f.path, "severity": f.severity, "message": f.message}
            for f in critic.findings
        ]
        summary["critic_parse_error"] = critic.parse_error
    return summary


def _critic_findings_dicts(critic: CriticReport | None) -> list[dict]:
    if critic is None:
        return []
    return [
        {"path": f.path, "severity": f.severity, "message": f.message}
        for f in critic.findings
    ]


async def _run_agent_pass(
    user_prompt: str,
    *,
    model: str,
    max_turns: int,
    debug: bool,
    transcript: list[str],
    usage: UsageStats,
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
            detect_reaction_class,
            verify_quote,
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
            "mcp__eln__detect_reaction_class",
            "mcp__eln__verify_quote",
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
            elif isinstance(message, ResultMessage):
                usage.merge(message)


async def extract(
    paragraph: str,
    *,
    model: str = DEFAULT_MODEL,
    max_iters: int = 5,
    debug: bool = False,
    enable_critic: bool = True,
) -> ExtractResult:
    """Run the agent against a single paragraph, returning the finalized output."""
    if len(paragraph) > MAX_PARAGRAPH_CHARS:
        return ExtractResult(
            success=False,
            pbtxt="",
            json_text="",
            validation_summary={},
            transcript=[],
            failure_summary={
                "reason": "paragraph too large",
                "size_chars": len(paragraph),
                "limit_chars": MAX_PARAGRAPH_CHARS,
                "explanation": (
                    "Paragraph exceeds the input size cap. The schema is "
                    "single-reaction; multi-step or multi-page procedures "
                    "should be split before extraction."
                ),
            },
        )

    normalization = normalize_paragraph(paragraph)
    finalized = FinalizedReaction()
    transcript: list[str] = []
    critic_report: CriticReport | None = None
    revision_triggered = False
    usage = UsageStats()

    with finalized_slot(finalized):
        user_prompt = USER_PROMPT_TEMPLATE.format(paragraph=normalization.normalized)
        await _run_agent_pass(
            user_prompt,
            model=model,
            max_turns=max_iters * 3,
            debug=debug,
            transcript=transcript,
            usage=usage,
        )

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
            if not critic_report.is_clean and any(
                f.severity == "ERROR" for f in critic_report.findings
            ):
                # Snapshot pre-revision state so we can roll back cleanly
                # if the revision fails to finalize. Also clear the
                # rule_history so the divergence counter measures only the
                # revision round, not the cumulative loop.
                revision_triggered = True
                prior_pbtxt = finalized.pbtxt
                prior_json = finalized.json_text
                prior_validation = finalized.validation_summary
                prior_history = dict(finalized.rule_history)
                finalized.pbtxt = ""
                finalized.json_text = ""
                finalized.rule_history.clear()

                revision_prompt = (
                    critic_report.as_repair_prompt()
                    + "\n\n"
                    + user_prompt
                )
                await _run_agent_pass(
                    revision_prompt,
                    model=model,
                    max_turns=max_iters * 3,
                    debug=debug,
                    transcript=transcript,
                    usage=usage,
                )
                # If the revision round didn't finalize, restore the
                # pre-critic clean state (and its rule_history) so the
                # returned result is internally consistent.
                if not finalized.pbtxt:
                    finalized.pbtxt = prior_pbtxt
                    finalized.json_text = prior_json
                    finalized.validation_summary = prior_validation
                    finalized.rule_history.clear()
                    finalized.rule_history.update(prior_history)

    success = bool(finalized.pbtxt)
    return ExtractResult(
        success=success,
        pbtxt=finalized.pbtxt,
        json_text=finalized.json_text,
        validation_summary=finalized.validation_summary,
        transcript=transcript,
        critic_findings=_critic_findings_dicts(critic_report),
        iterations=finalized.iterations,
        rule_history=dict(finalized.rule_history),
        critic_ran=critic_report is not None,
        revision_triggered=revision_triggered,
        usage=usage,
        failure_summary=(
            {}
            if success
            else _build_failure_summary(
                finalized,
                critic_report,
                reason=(
                    "agent did not finalize a draft within the iteration "
                    "budget; see rule_history for the most-frequent failures"
                ),
            )
        ),
    )
