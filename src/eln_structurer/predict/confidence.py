"""Per-channel confidence + multi-source consensus.

The single most important risk mitigation in the predictor: never
trust one channel alone. ``multi_source_vote`` looks at proposals from
multiple sources and tells the composition layer how strongly they
agree. A "high" verdict needs at least two independent sources voting
the same way; otherwise the verdict steps down to "medium" or "low".

These types are public — they appear in the ``ProposalResult`` returned
to callers, so a chemist reading the output can see exactly which
channels supported each slot and where they disagreed.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Generic, TypeVar

from eln_structurer.predict.corpus import CorpusSource


T = TypeVar("T")


class ConfidenceLevel(str, Enum):
    """Four-value confidence enum surfaced on every proposed slot.

    ``LOW`` and ``SPECULATIVE`` trigger "human review required"
    annotations downstream. ``SPECULATIVE`` is reserved for the
    LLM-priors-only path (no retrieval support, no class skeleton hit).
    """
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    SPECULATIVE = "speculative"


@dataclass(frozen=True)
class SlotProposal(Generic[T]):
    """One channel's vote for a single slot in the protocol.

    ``source`` identifies the channel that produced the proposal so the
    vote can require *independent* sources to agree (HTE + literature,
    not literature + literature).
    """
    value: T
    source: CorpusSource
    source_id: str
    weight: float = 1.0


@dataclass(frozen=True)
class ConsensusResult(Generic[T]):
    """Output of ``multi_source_vote`` for one slot."""
    value: T | None
    confidence: ConfidenceLevel
    agreeing_sources: list[CorpusSource]
    dissent: list[SlotProposal[T]]
    # Provenance trail: which proposal won and which lost. Renderable as
    # human-readable text by the caller for the reasoning trail.
    chosen_proposal: SlotProposal[T] | None = None


# --- vote ----------------------------------------------------------


def multi_source_vote(
    proposals: list[SlotProposal[T]],
    *,
    independent_sources_for_high: int = 2,
) -> ConsensusResult[T]:
    """Vote across proposals; require ``independent_sources_for_high``
    distinct sources to agree before returning HIGH confidence.

    No proposals → ``LOW`` with value None.
    All proposals agree from one source → ``MEDIUM`` (still better than
        no support, but not multi-source).
    """
    if not proposals:
        return ConsensusResult(
            value=None,
            confidence=ConfidenceLevel.LOW,
            agreeing_sources=[],
            dissent=[],
            chosen_proposal=None,
        )

    # Group by value (string-coerced) — values that compare equal
    # contribute to the same tally.
    tallies: dict = {}
    for p in proposals:
        key = repr(p.value)
        bucket = tallies.setdefault(key, {"value": p.value, "weight": 0.0,
                                          "sources": set(), "proposals": []})
        bucket["weight"] += p.weight
        bucket["sources"].add(p.source)
        bucket["proposals"].append(p)

    # Pick winner by total weight; ties broken by source-diversity.
    winner_key = max(
        tallies,
        key=lambda k: (tallies[k]["weight"], len(tallies[k]["sources"])),
    )
    winner = tallies[winner_key]
    n_distinct_sources = len(winner["sources"])

    if n_distinct_sources >= independent_sources_for_high:
        level = ConfidenceLevel.HIGH
    elif winner["weight"] >= 2.0:
        # Multiple proposals from one source — better than nothing.
        level = ConfidenceLevel.MEDIUM
    else:
        level = ConfidenceLevel.LOW

    dissent = [p for p in proposals if repr(p.value) != winner_key]
    # The chosen proposal is the highest-weight one from the winning bucket;
    # callers may want it for provenance display.
    chosen = max(winner["proposals"], key=lambda p: p.weight)

    return ConsensusResult(
        value=winner["value"],
        confidence=level,
        agreeing_sources=sorted(winner["sources"], key=lambda s: s.value),
        dissent=dissent,
        chosen_proposal=chosen,
    )


# --- summary helper -------------------------------------------------


@dataclass(frozen=True)
class ChannelReport:
    """Per-channel summary surfaced in the reasoning trail."""
    channel: str
    n_proposals: int
    confidence: ConfidenceLevel
    note: str = ""


def summarise_channels(by_channel: dict[str, list[SlotProposal]]) -> list[ChannelReport]:
    out: list[ChannelReport] = []
    for name, props in by_channel.items():
        if not props:
            out.append(ChannelReport(channel=name, n_proposals=0,
                                     confidence=ConfidenceLevel.LOW,
                                     note="no proposals from this channel"))
            continue
        unique_sources = len({p.source for p in props})
        if unique_sources >= 2:
            level = ConfidenceLevel.HIGH
        elif len(props) >= 2:
            level = ConfidenceLevel.MEDIUM
        else:
            level = ConfidenceLevel.LOW
        out.append(ChannelReport(channel=name, n_proposals=len(props),
                                 confidence=level))
    return out


__all__ = [
    "ConfidenceLevel",
    "SlotProposal",
    "ConsensusResult",
    "multi_source_vote",
    "ChannelReport",
    "summarise_channels",
]
