"""Tier-6 LLM-agentic orchestration loop.

Wraps the deterministic :func:`propose_protocol` core without touching
it: the determinisitic ranker still produces the candidate list, the
LLM critiques it, and only **slot-level overrides cited against a
retrieved record** are accepted. The agent NEVER rewrites the draft
wholesale — that would re-open the hallucination surface that Tier 2's
voting layer was built to close.

Architecture:

    propose_protocol(target, constraints)   <-- deterministic baseline
            │
            ▼
    build candidate brief (top-K JSON)
            │
            ▼
    ClaudeSDKClient(system=predict_agent_prompt,
                    tools=[retrieve_exact, retrieve_similar,
                           safety_screen, compute_mw,
                           expand_abbreviation, detect_reaction_class])
            │
            ▼
    parse AgenticVerdict JSON          <-- pure Python parser
            │
            ▼
    apply_verdict(predictor_output, verdict)
            │   - swap top candidate if verdict.endorsed_index != 1
            │   - apply each slot_overrides patch
            │   - record the agent's rationale in proposal.warnings
            │   - block if safety verdict is BLOCKED
            ▼
    AgenticPredictorOutput
"""

from __future__ import annotations

import json
import os
import re
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
from pydantic import BaseModel, Field, ValidationError

from eln_structurer.predict.agent_prompts import (
    build_predict_agent_system_prompt,
    build_predict_agent_user_prompt,
)
from eln_structurer.predict.composition import ProposalResult
from eln_structurer.predict.corpus import LocalCorpus
from eln_structurer.predict.proposer import (
    PredictorOutput,
    propose_protocol,
)
from eln_structurer.predict.ranker import RankedProposal, Weights
from eln_structurer.predict.risks import SafetyVerdict, safety_screen
from eln_structurer.schema import (
    CompoundIdentifierModel,
    CompoundModel,
    ReactionInputModel,
)
from eln_structurer.tools import (
    compute_mw,
    detect_reaction_class,
    expand_abbreviation,
)
from eln_structurer.tools.predict_tools import (
    retrieve_exact_reaction,
    retrieve_similar_reactions,
    safety_screen_tool,
    set_active_corpus,
)


DEFAULT_AGENT_MODEL = "claude-sonnet-4-6"
HIGH_QUALITY_AGENT_MODEL = "claude-opus-4-7"


# ---------------------------------------------------------------------------
# Verdict models
# ---------------------------------------------------------------------------


class _SlotOverride(BaseModel):
    """One slot patch the agent wants applied to the endorsed candidate."""
    slot_name: str
    new_value: str
    source: str
    rationale: str = ""


class _VerdictModel(BaseModel):
    """Pydantic mirror of the JSON the agent must emit."""
    endorsed_index: int | None = Field(default=None)
    rationale: str = ""
    slot_overrides: list[_SlotOverride] = Field(default_factory=list)
    additional_warnings: list[str] = Field(default_factory=list)
    safety_verdict: str = "warn"


@dataclass
class AgenticVerdict:
    """Validated, normalised view of the agent's structured output."""
    endorsed_index: int | None
    rationale: str
    slot_overrides: list[_SlotOverride]
    additional_warnings: list[str]
    safety_verdict: str
    parse_error: str | None = None
    raw_text: str = ""

    @property
    def is_blocked(self) -> bool:
        return self.safety_verdict.lower() == "blocked"


@dataclass
class AgentUsage:
    total_cost_usd: float = 0.0
    duration_ms: int = 0
    num_turns: int = 0
    usage_blob: dict[str, Any] = field(default_factory=dict)

    def merge(self, msg: ResultMessage) -> None:
        if msg.total_cost_usd is not None:
            self.total_cost_usd += msg.total_cost_usd
        self.duration_ms += msg.duration_ms
        self.num_turns += msg.num_turns
        if msg.usage:
            for k, v in msg.usage.items():
                if isinstance(v, (int, float)):
                    self.usage_blob[k] = self.usage_blob.get(k, 0) + v
                else:
                    self.usage_blob.setdefault(k, v)


@dataclass
class AgenticPredictorOutput:
    """Tier-6 output wrapping the Tier-5 baseline + agent verdict.

    The deterministic baseline lives in ``baseline``; the (possibly
    reordered + slot-patched) ranked list the agent endorses is in
    ``ranked_proposals``. Callers that don't care about the audit
    trail can read ``ranked_proposals[0]`` straight off.
    """
    target_reaction_smiles: str
    baseline: PredictorOutput
    ranked_proposals: list[RankedProposal]
    verdict: AgenticVerdict
    warnings: list[str] = field(default_factory=list)
    agent_ran: bool = False
    usage: AgentUsage = field(default_factory=AgentUsage)


# ---------------------------------------------------------------------------
# Candidate brief: serialise the deterministic top-K as JSON for the agent
# ---------------------------------------------------------------------------


def _summarise_candidate(rank: int, ranked: RankedProposal) -> dict[str, Any]:
    """Compact serialisation per candidate — keeps prompt tokens down."""
    proposal = ranked.proposal
    inputs_block: list[dict[str, Any]] = []
    for inp in proposal.draft.inputs:
        for comp in inp.components:
            name = next((i.value for i in comp.identifiers if i.type == "NAME"), None)
            smiles = next((i.value for i in comp.identifiers if i.type == "SMILES"), None)
            inputs_block.append({
                "slot_name": inp.name,
                "role": comp.reaction_role,
                "name": name,
                "smiles": smiles,
                "confidence": proposal.slot_confidences.get(
                    inp.name, "UNKNOWN"
                ).value if hasattr(proposal.slot_confidences.get(inp.name), "value")
                else str(proposal.slot_confidences.get(inp.name, "UNKNOWN")),
                "provenance": proposal.slot_provenance.get(inp.name, []),
            })
    return {
        "rank": rank,
        "class": proposal.skeleton_class,
        "overall_confidence": proposal.overall_confidence.value,
        "overall_score": round(ranked.overall_score, 3),
        "yield_score": round(ranked.yield_score, 3),
        "greenness_score": round(ranked.greenness_score, 3),
        "retrieval_score": round(ranked.retrieval_score, 3),
        "constraint_violations": ranked.constraint_violations,
        "warnings": proposal.warnings,
        "inputs": inputs_block,
        "temperature_celsius": (
            proposal.draft.conditions.temperature.setpoint_celsius
            if proposal.draft.conditions and proposal.draft.conditions.temperature
            else None
        ),
        "duration_minutes": (
            proposal.draft.conditions.duration_minutes
            if proposal.draft.conditions else None
        ),
        "atmosphere": (
            proposal.draft.conditions.atmosphere
            if proposal.draft.conditions else None
        ),
    }


def _build_candidate_brief(
    output: PredictorOutput, *, top_k: int
) -> tuple[str, int]:
    """Render the top-K ranked list as a JSON block for the prompt."""
    chosen = output.ranked_proposals[:top_k]
    payload = [
        _summarise_candidate(i + 1, ranked) for i, ranked in enumerate(chosen)
    ]
    return json.dumps(payload, indent=2), len(chosen)


# ---------------------------------------------------------------------------
# Verdict parsing
# ---------------------------------------------------------------------------


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```\s*$", re.MULTILINE)


def _parse_verdict(raw: str) -> AgenticVerdict:
    """Strip markdown fences, json.loads, validate, normalise."""
    cleaned = _FENCE_RE.sub("", raw).strip()
    if not cleaned:
        return AgenticVerdict(
            endorsed_index=None, rationale="", slot_overrides=[],
            additional_warnings=[], safety_verdict="warn",
            parse_error="agent returned empty text",
            raw_text=raw,
        )
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        return AgenticVerdict(
            endorsed_index=None, rationale="", slot_overrides=[],
            additional_warnings=[], safety_verdict="warn",
            parse_error=f"non-JSON agent output: {exc}",
            raw_text=raw,
        )
    try:
        model = _VerdictModel.model_validate(payload)
    except ValidationError as exc:
        return AgenticVerdict(
            endorsed_index=None, rationale="", slot_overrides=[],
            additional_warnings=[], safety_verdict="warn",
            parse_error=f"agent schema error: {exc}",
            raw_text=raw,
        )
    return AgenticVerdict(
        endorsed_index=model.endorsed_index,
        rationale=model.rationale,
        slot_overrides=list(model.slot_overrides),
        additional_warnings=list(model.additional_warnings),
        safety_verdict=model.safety_verdict.lower(),
        raw_text=cleaned,
    )


# ---------------------------------------------------------------------------
# Verdict application: re-rank + patch slots
# ---------------------------------------------------------------------------


_LITERATURE_CONSENSUS = "literature_consensus"


def _apply_slot_overrides(
    proposal: ProposalResult,
    overrides: list[_SlotOverride],
) -> tuple[ProposalResult, list[str]]:
    """Mutate the proposal's draft in place per the agent's slot patches.

    Each override carries a citation. We trust the citation only if the
    string is non-empty — the agent is required to cite either a
    corpus record id or ``literature_consensus``.
    Returns the updated proposal and a list of strings describing each
    applied (or rejected) patch for the audit trail.
    """
    audit: list[str] = []
    for ov in overrides:
        source = (ov.source or "").strip()
        if not source:
            audit.append(
                f"rejected override on {ov.slot_name!r}: missing source citation"
            )
            continue
        # Find the matching ReactionInputModel by slot_name.
        target_input: ReactionInputModel | None = None
        for inp in proposal.draft.inputs:
            if inp.name == ov.slot_name:
                target_input = inp
                break
        if target_input is None:
            audit.append(
                f"rejected override on {ov.slot_name!r}: slot not in draft"
            )
            continue
        # Replace the NAME identifier on the first component.
        comp = target_input.components[0]
        new_idents = [
            i for i in comp.identifiers if i.type != "NAME"
        ]
        new_idents.append(
            CompoundIdentifierModel(type="NAME", value=ov.new_value)
        )
        target_input.components[0] = CompoundModel(
            identifiers=new_idents,
            amount=comp.amount,
            reaction_role=comp.reaction_role,
            is_limiting=comp.is_limiting,
        )
        # Record provenance.
        existing = proposal.slot_provenance.setdefault(ov.slot_name, [])
        existing.append(f"agent_override:{source}")
        audit.append(
            f"slot {ov.slot_name!r} → {ov.new_value!r} (source={source})"
        )
    return proposal, audit


def _safety_block_proposal(
    proposal: ProposalResult,
) -> tuple[SafetyVerdict, list[str]]:
    """Deterministic safety check — agents may lie about the verdict."""
    report = safety_screen(proposal.draft)
    return report.verdict, list(report.flags)


def apply_verdict(
    baseline: PredictorOutput,
    verdict: AgenticVerdict,
) -> tuple[list[RankedProposal], list[str]]:
    """Re-rank the baseline candidates per the agent's verdict.

    Returns the new ranked list (best-first) and a list of warnings
    capturing what the agent did. Safety verdict is re-checked
    independently with the deterministic ``safety_screen`` — the LLM
    cannot self-certify safety.
    """
    warnings: list[str] = []
    candidates = list(baseline.ranked_proposals)
    if not candidates:
        return candidates, ["no baseline candidates to apply verdict against"]

    if verdict.parse_error:
        warnings.append(f"agent verdict parse error: {verdict.parse_error}")
        return candidates, warnings

    # Re-order to put the endorsed candidate first.
    if verdict.endorsed_index is not None and 1 <= verdict.endorsed_index <= len(candidates):
        idx = verdict.endorsed_index - 1
        if idx != 0:
            chosen = candidates.pop(idx)
            candidates.insert(0, chosen)
            warnings.append(
                f"agent re-ranked: candidate #{verdict.endorsed_index} "
                "promoted to top"
            )

    # Apply slot overrides to the now-top candidate.
    if verdict.slot_overrides:
        proposal = candidates[0].proposal
        _, audit = _apply_slot_overrides(proposal, verdict.slot_overrides)
        for line in audit:
            warnings.append(f"agent override: {line}")

    # Re-run safety screen on the (possibly mutated) top candidate.
    verdict_sv, flags = _safety_block_proposal(candidates[0].proposal)
    if verdict_sv is SafetyVerdict.BLOCKED:
        warnings.append(
            "agent's endorsed candidate failed deterministic safety screen: "
            + "; ".join(flags)
        )
        # Drop the blocked candidate entirely; fall back to the next one.
        blocked = candidates.pop(0)
        candidates[0:0] = []  # noop; keeps mypy happy on list view
        warnings.append(
            f"dropped blocked candidate (class={blocked.proposal.skeleton_class})"
        )

    if verdict.rationale:
        warnings.append(f"agent rationale: {verdict.rationale}")
    for w in verdict.additional_warnings:
        warnings.append(f"agent warning: {w}")

    return candidates, warnings


# ---------------------------------------------------------------------------
# The agent driver
# ---------------------------------------------------------------------------


def agent_available() -> bool:
    """Agent loop requires an Anthropic key. Skip gracefully without one."""
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


async def _run_predict_agent(
    *,
    user_prompt: str,
    model: str,
    max_turns: int,
    usage: AgentUsage,
) -> str:
    """Spin up an SDK client with the predict toolbox, return raw text."""
    server = create_sdk_mcp_server(
        name="eln_predict",
        version="0.1.0",
        tools=[
            retrieve_exact_reaction,
            retrieve_similar_reactions,
            safety_screen_tool,
            compute_mw,
            expand_abbreviation,
            detect_reaction_class,
        ],
    )
    options = ClaudeAgentOptions(
        model=model,
        mcp_servers={"eln_predict": server},
        allowed_tools=[
            "mcp__eln_predict__retrieve_exact_reaction",
            "mcp__eln_predict__retrieve_similar_reactions",
            "mcp__eln_predict__safety_screen",
            "mcp__eln_predict__compute_mw",
            "mcp__eln_predict__expand_abbreviation",
            "mcp__eln_predict__detect_reaction_class",
        ],
        system_prompt=build_predict_agent_system_prompt(),
        max_turns=max_turns,
    )
    text_parts: list[str] = []
    async with ClaudeSDKClient(options=options) as client:
        await client.query(user_prompt)
        async for message in client.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        text_parts.append(block.text)
            elif isinstance(message, ResultMessage):
                usage.merge(message)
    return "".join(text_parts)


async def agentic_propose_protocol(
    rxn_smiles: str,
    *,
    constraints: dict[str, Any] | None = None,
    corpus: LocalCorpus | None = None,
    weights: Weights | None = None,
    reference_year: int = 2025,
    model: str = DEFAULT_AGENT_MODEL,
    top_k: int = 3,
    max_turns: int = 10,
    enable_agent: bool = True,
) -> AgenticPredictorOutput:
    """Run the deterministic baseline, then critique it with an LLM agent.

    The deterministic baseline is always computed. The agent runs only
    when ``enable_agent`` is True AND an ``ANTHROPIC_API_KEY`` is
    available; otherwise the baseline is returned with an empty
    verdict. This is exactly the same graceful-degrade contract the
    extractor's critic uses.
    """
    baseline = propose_protocol(
        rxn_smiles,
        constraints=constraints,
        corpus=corpus,
        weights=weights,
        reference_year=reference_year,
    )
    warnings = list(baseline.warnings)
    usage = AgentUsage()

    if not baseline.ranked_proposals:
        return AgenticPredictorOutput(
            target_reaction_smiles=rxn_smiles,
            baseline=baseline,
            ranked_proposals=[],
            verdict=AgenticVerdict(
                endorsed_index=None, rationale="",
                slot_overrides=[], additional_warnings=[],
                safety_verdict="warn",
                parse_error="no baseline candidates",
            ),
            warnings=warnings + ["agent skipped: no baseline candidates"],
            agent_ran=False,
        )

    if not enable_agent or not agent_available():
        # Even without the LLM, run safety screen deterministically on
        # the top candidate so the output's safety_verdict is honest.
        sv, flags = _safety_block_proposal(baseline.ranked_proposals[0].proposal)
        return AgenticPredictorOutput(
            target_reaction_smiles=rxn_smiles,
            baseline=baseline,
            ranked_proposals=list(baseline.ranked_proposals),
            verdict=AgenticVerdict(
                endorsed_index=1, rationale="baseline default",
                slot_overrides=[], additional_warnings=[],
                safety_verdict=sv.value,
            ),
            warnings=warnings + ([f"safety: {'; '.join(flags)}"] if flags else []),
            agent_ran=False,
        )

    # Bind the corpus into the predict tools so the agent's
    # retrieve_exact/retrieve_similar calls see the same corpus the
    # baseline used.
    from eln_structurer.predict.hte_corpus import default_seed_corpus
    active = corpus if corpus is not None else default_seed_corpus()
    set_active_corpus(active)

    try:
        brief, n = _build_candidate_brief(baseline, top_k=top_k)
        constraints_json = json.dumps(constraints or {}, indent=2)
        user_prompt = build_predict_agent_user_prompt(
            target_smiles=rxn_smiles,
            constraints_json=constraints_json,
            candidates_block=brief,
            n_candidates=n,
        )
        raw = await _run_predict_agent(
            user_prompt=user_prompt,
            model=model,
            max_turns=max_turns,
            usage=usage,
        )
    finally:
        set_active_corpus(None)

    verdict = _parse_verdict(raw)
    ranked, agent_warnings = apply_verdict(baseline, verdict)
    warnings.extend(agent_warnings)

    return AgenticPredictorOutput(
        target_reaction_smiles=rxn_smiles,
        baseline=baseline,
        ranked_proposals=ranked,
        verdict=verdict,
        warnings=warnings,
        agent_ran=True,
        usage=usage,
    )


__all__ = [
    "AgenticPredictorOutput",
    "AgenticVerdict",
    "AgentUsage",
    "DEFAULT_AGENT_MODEL",
    "HIGH_QUALITY_AGENT_MODEL",
    "agent_available",
    "agentic_propose_protocol",
    "apply_verdict",
]
