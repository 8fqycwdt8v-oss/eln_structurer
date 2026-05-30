"""Multi-objective ranking for candidate protocols.

Takes a list of ``ProposalResult`` objects produced by the composer,
scores each across the user-weighted objectives, and returns them
sorted best-first. Each score component is exposed in the
``RankedProposal`` so the chemist sees WHY one beat another.

Objectives surfaced in this Tier-3 implementation:

  yield         — conservative yield score from yield_model
  greenness     — CHEM21-derived solvent score
  confidence    — overall composer confidence (HIGH=1.0 → SPECULATIVE=0.0)
  retrieval     — average similarity across all channels (reward
                  candidates backed by strong neighbours)
  constraint    — hard penalty: candidates that violate a user
                  hard-constraint are penalised but never auto-dropped
                  (the composer already filtered at query time; this
                  is the belt-and-braces second pass)

Weights default to 1.0 across the four positive objectives. Pass a
custom ``Weights`` instance to bias the ranking toward a particular
user preference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from eln_structurer.predict.composition import ProposalResult
from eln_structurer.predict.confidence import ConfidenceLevel
from eln_structurer.predict.greenness import protocol_solvent_score
from eln_structurer.predict.retrieval import Hit
from eln_structurer.predict.yield_model import (
    YieldEstimate,
    conservative_yield_score,
    estimate_yield,
)


_CONFIDENCE_TO_SCORE = {
    ConfidenceLevel.HIGH: 1.0,
    ConfidenceLevel.MEDIUM: 0.66,
    ConfidenceLevel.LOW: 0.33,
    ConfidenceLevel.SPECULATIVE: 0.0,
}


@dataclass(frozen=True)
class Weights:
    """User-tunable weights across the ranking objectives."""
    yield_: float = 1.0
    greenness: float = 1.0
    confidence: float = 1.0
    retrieval: float = 0.5
    constraint_penalty: float = 2.0   # multiplier on the penalty term


@dataclass(frozen=True)
class RankedProposal:
    """One ranked candidate with the score breakdown."""
    proposal: ProposalResult
    overall_score: float
    yield_score: float
    greenness_score: float
    confidence_score: float
    retrieval_score: float
    constraint_penalty: float
    yield_estimate: YieldEstimate | None
    constraint_violations: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Constraint check (belt-and-braces; primary filtering happens at query time)
# ---------------------------------------------------------------------------


_HALOGENATED_SOLVENT_NAMES = {
    "dichloromethane", "dcm", "ch2cl2",
    "chloroform", "chcl3",
    "carbon tetrachloride", "ccl4",
    "1,2-dichloroethane", "dce",
}


def _solvent_names(proposal: ProposalResult) -> list[str]:
    out: list[str] = []
    for inp in proposal.draft.inputs:
        for comp in inp.components:
            if comp.reaction_role != "SOLVENT":
                continue
            for ident in comp.identifiers:
                if ident.type == "NAME":
                    out.append(ident.value)
    return out


def _check_constraint_violations(
    proposal: ProposalResult, constraints: dict[str, Any] | None
) -> list[str]:
    if not constraints:
        return []
    violations: list[str] = []
    solvent_lc = {s.lower() for s in _solvent_names(proposal)}
    if constraints.get("no_halogenated_solvents"):
        for name in solvent_lc:
            if name in _HALOGENATED_SOLVENT_NAMES:
                violations.append(f"halogenated solvent {name!r}")
    if (cap := constraints.get("max_temperature_c")) is not None:
        t = proposal.draft.conditions.temperature
        if t and t.setpoint_celsius and t.setpoint_celsius > cap:
            violations.append(
                f"temperature {t.setpoint_celsius} > cap {cap}"
            )
    if (cap := constraints.get("max_duration_minutes")) is not None:
        d = proposal.draft.conditions.duration_minutes
        if d and d > cap:
            violations.append(f"duration {d} > cap {cap}")
    return violations


# ---------------------------------------------------------------------------
# Retrieval score
# ---------------------------------------------------------------------------


def _retrieval_score(hits_by_channel: dict[str, list[Hit]]) -> float:
    """Average similarity across all channel hits, in [0, 1]."""
    sims: list[float] = []
    for hits in hits_by_channel.values():
        for h in hits:
            sims.append(h.similarity)
    if not sims:
        return 0.0
    return sum(sims) / len(sims)


# ---------------------------------------------------------------------------
# Rank
# ---------------------------------------------------------------------------


def rank_proposals(
    proposals: list[ProposalResult],
    *,
    hits_by_proposal: list[dict[str, list[Hit]]] | None = None,
    constraints: dict[str, Any] | None = None,
    weights: Weights | None = None,
) -> list[RankedProposal]:
    """Score and order proposals best-first.

    ``hits_by_proposal`` parallels ``proposals`` and supplies the hits
    used to estimate yield + retrieval support per candidate. Pass an
    empty dict for any candidate that was built from scratch (no
    channel evidence).
    """
    if hits_by_proposal is None:
        hits_by_proposal = [{} for _ in proposals]
    if len(hits_by_proposal) != len(proposals):
        raise ValueError("hits_by_proposal length must match proposals")
    weights = weights or Weights()

    out: list[RankedProposal] = []
    for proposal, hits_dict in zip(proposals, hits_by_proposal, strict=False):
        all_hits = [h for hits in hits_dict.values() for h in hits]
        ye = estimate_yield(all_hits)
        yield_score = conservative_yield_score(ye)
        greenness = protocol_solvent_score(_solvent_names(proposal))
        confidence = _CONFIDENCE_TO_SCORE[proposal.overall_confidence]
        retrieval = _retrieval_score(hits_dict)
        violations = _check_constraint_violations(proposal, constraints)
        penalty = len(violations) * weights.constraint_penalty

        overall = (
            weights.yield_ * yield_score
            + weights.greenness * greenness
            + weights.confidence * confidence
            + weights.retrieval * retrieval
            - penalty
        )

        out.append(RankedProposal(
            proposal=proposal,
            overall_score=overall,
            yield_score=yield_score,
            greenness_score=greenness,
            confidence_score=confidence,
            retrieval_score=retrieval,
            constraint_penalty=penalty,
            yield_estimate=ye,
            constraint_violations=violations,
        ))

    out.sort(key=lambda r: r.overall_score, reverse=True)
    return out


__all__ = [
    "Weights",
    "RankedProposal",
    "rank_proposals",
]
