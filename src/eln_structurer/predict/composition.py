"""Slot-filling composition layer.

Given a class skeleton + channel evidence (exact-match hits, K-NN
neighbours, HTE hits), produce a ``ProposalResult`` carrying the
filled ``ReactionDraft`` together with per-slot confidence and the
reasoning trail.

Compose is **deterministic** — no LLM call. The LLM enters later (Tier 5)
as the agentic orchestrator that decides which channels to call and
how to handle conflicts the composer surfaces. Keeping this layer
deterministic means tests can pin its behaviour against synthetic
corpora without burning API tokens.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from eln_structurer.predict.confidence import (
    ChannelReport,
    ConfidenceLevel,
    ConsensusResult,
    SlotProposal,
    multi_source_vote,
)
from eln_structurer.predict.retrieval import Hit
from eln_structurer.predict.skeleton import ProtocolSkeleton, Slot
from eln_structurer.schema import (
    AmountModel,
    CompoundIdentifierModel,
    CompoundModel,
    ConditionsModel,
    OutcomeModel,
    ProductModel,
    ReactionDraft,
    ReactionInputModel,
    StirringModel,
    TemperatureModel,
)


@dataclass
class ProposalResult:
    """A proposed protocol + the reasoning trail that produced it."""
    draft: ReactionDraft
    skeleton_class: str
    slot_confidences: dict[str, ConfidenceLevel]
    slot_provenance: dict[str, list[str]]
    channel_reports: list[ChannelReport] = field(default_factory=list)
    overall_confidence: ConfidenceLevel = ConfidenceLevel.LOW
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal: pull condition fields out of a hit's conditions dict
# ---------------------------------------------------------------------------


def _solvent_names_from_hit(hit: Hit) -> list[str]:
    conds = hit.record.conditions or {}
    return [str(s) for s in (conds.get("solvents") or [])]


def _catalyst_names_from_hit(hit: Hit) -> list[str]:
    conds = hit.record.conditions or {}
    return [str(s) for s in (conds.get("catalysts") or [])]


def _reagent_names_from_hit(hit: Hit) -> list[str]:
    conds = hit.record.conditions or {}
    return [str(s) for s in (conds.get("reagents") or [])]


def _temperature_from_hit(hit: Hit) -> float | None:
    conds = hit.record.conditions or {}
    t = conds.get("temperature_celsius")
    return float(t) if t is not None else None


def _duration_from_hit(hit: Hit) -> float | None:
    conds = hit.record.conditions or {}
    d = conds.get("duration_minutes")
    return float(d) if d is not None else None


def _atmosphere_from_hit(hit: Hit) -> str | None:
    conds = hit.record.conditions or {}
    a = conds.get("atmosphere")
    return str(a) if a else None


# ---------------------------------------------------------------------------
# Vote per slot
# ---------------------------------------------------------------------------


def _vote_for_slot(
    slot: Slot,
    hits_by_channel: dict[str, list[Hit]],
) -> ConsensusResult[str]:
    """Build SlotProposal list from channel hits and vote.

    The simple v1 heuristic: a literature/HTE hit "votes" for any of
    its catalysts/reagents/solvents based on slot.role. The composer
    doesn't know which named compound goes in which slot — it expects
    the corpus records to be pre-tagged by role. Hits without
    sufficient metadata simply don't vote on that slot.
    """
    proposals: list[SlotProposal[str]] = []
    for channel_name, hits in hits_by_channel.items():
        for hit in hits:
            candidates: list[str]
            if slot.role == "SOLVENT":
                candidates = _solvent_names_from_hit(hit)
            elif slot.role == "CATALYST":
                candidates = _catalyst_names_from_hit(hit)
            elif slot.role == "REAGENT":
                candidates = _reagent_names_from_hit(hit)
            else:
                # REACTANTs are usually known from the user's target SMILES;
                # we don't vote them here.
                candidates = []
            for c in candidates:
                proposals.append(SlotProposal(
                    value=c,
                    source=hit.record.source,
                    source_id=f"{channel_name}:{hit.record.source_id}",
                    weight=max(hit.similarity, 0.1),
                ))
    return multi_source_vote(proposals)


# ---------------------------------------------------------------------------
# Compose a draft
# ---------------------------------------------------------------------------


def compose_protocol(
    *,
    target_reaction_smiles: str,
    skeleton: ProtocolSkeleton,
    hits_by_channel: dict[str, list[Hit]],
    user_constraints: dict[str, Any] | None = None,
) -> ProposalResult:
    """Fill the skeleton's slots by voting across channel evidence.

    ``hits_by_channel`` is a mapping from channel name (e.g. ``"exact"``,
    ``"knn"``, ``"hte"``) to retrieved hits. The composer aggregates
    them through ``multi_source_vote`` per slot, falling back to the
    slot's ``fallback_names[0]`` when no hit votes for that slot
    (confidence then drops to SPECULATIVE).
    """
    slot_confidences: dict[str, ConfidenceLevel] = {}
    slot_provenance: dict[str, list[str]] = {}
    warnings: list[str] = []

    # Build the inputs list: one ReactionInputModel per slot that got filled.
    inputs: list[ReactionInputModel] = []

    # Always carry the target reaction SMILES into source_paragraph so
    # downstream tools can audit and so the schema's required field is
    # satisfied. Real provenance lives in the proposal result, not the
    # draft.
    reactant_smiles = _extract_reactant_smiles(target_reaction_smiles)
    product_smiles = _extract_product_smiles(target_reaction_smiles)

    for i, smi in enumerate(reactant_smiles):
        inputs.append(ReactionInputModel(
            name=f"reactant_{i}",
            components=[CompoundModel(
                identifiers=[CompoundIdentifierModel(type="SMILES", value=smi)],
                reaction_role="REACTANT",
                is_limiting=(i == 0),
            )],
        ))

    for slot in skeleton.slots:
        # Reactants are filled from the user's input, not voted from the corpus.
        if slot.role == "REACTANT":
            slot_confidences[slot.name] = ConfidenceLevel.HIGH
            slot_provenance[slot.name] = ["target_smiles"]
            continue
        consensus = _vote_for_slot(slot, hits_by_channel)
        chosen: str | None = consensus.value
        if chosen is None and slot.fallback_names:
            chosen = slot.fallback_names[0]
            slot_confidences[slot.name] = ConfidenceLevel.SPECULATIVE
            slot_provenance[slot.name] = [f"fallback:{chosen}"]
            warnings.append(
                f"slot {slot.name!r}: no channel evidence; used "
                f"skeleton fallback {chosen!r} (SPECULATIVE)"
            )
        elif chosen is None and slot.required:
            slot_confidences[slot.name] = ConfidenceLevel.LOW
            slot_provenance[slot.name] = []
            warnings.append(
                f"slot {slot.name!r}: REQUIRED but no evidence and no "
                "fallback — output incomplete"
            )
            continue
        elif chosen is None:
            # Optional slot, no evidence — skip it entirely.
            continue
        else:
            slot_confidences[slot.name] = consensus.confidence
            slot_provenance[slot.name] = [
                f"{p.source.value}:{p.source_id}" for p in
                (consensus.dissent + ([consensus.chosen_proposal] if consensus.chosen_proposal else []))
            ]

        # Materialise the slot as a ReactionInputModel.
        equiv_value: float | None = None
        if slot.typical_equiv_range:
            lo, hi = slot.typical_equiv_range
            equiv_value = (lo + hi) / 2 if lo > 0 else None

        amount = (
            AmountModel(value=equiv_value, units="equiv")
            if equiv_value and equiv_value > 0 else None
        )

        inputs.append(ReactionInputModel(
            name=slot.name,
            components=[CompoundModel(
                identifiers=[CompoundIdentifierModel(type="NAME", value=chosen)],
                amount=amount,
                reaction_role=slot.role,
            )],
        ))

    # Aggregate conditions: vote on temperature, duration, atmosphere.
    temperature = _vote_temperature(hits_by_channel) or _midpoint(
        skeleton.typical_temperature_c
    )
    duration = _vote_duration(hits_by_channel) or _midpoint(
        skeleton.typical_duration_minutes
    )
    atmosphere = _vote_atmosphere(hits_by_channel) or (
        "nitrogen" if skeleton.inert_atmosphere_required else None
    )

    conditions = ConditionsModel(
        temperature=TemperatureModel(
            setpoint_celsius=temperature,
            control_type="HEATER" if temperature and temperature > 30 else "AMBIENT",
        ),
        stirring=StirringModel(type="MAGNETIC"),
        atmosphere=atmosphere,
        duration_minutes=duration,
    )

    # Workups follow the skeleton sequence.
    from eln_structurer.schema import WorkupModel
    workups = [
        WorkupModel(type=t, description=f"Skeleton step #{i + 1}: {t}",
                    order=i + 1)
        for i, t in enumerate(skeleton.workup_sequence)
    ]

    outcomes = [OutcomeModel(products=[
        ProductModel(compound=CompoundModel(
            identifiers=[CompoundIdentifierModel(type="SMILES", value=s)]
                        if s else
                        [CompoundIdentifierModel(type="NAME", value="product")],
            reaction_role="PRODUCT",
        ))
        for s in product_smiles
    ] or [ProductModel(compound=CompoundModel(
        identifiers=[CompoundIdentifierModel(type="NAME", value="product")],
        reaction_role="PRODUCT",
    ))])]

    draft = ReactionDraft(
        identifiers=[CompoundIdentifierModel(type="SMILES", value=target_reaction_smiles)],
        inputs=inputs,
        conditions=conditions,
        workups=workups,
        outcomes=outcomes,
        notes=(
            f"Auto-composed from class skeleton {skeleton.reaction_class.value}. "
            f"Channels: {sorted(hits_by_channel.keys())}."
        ),
        source_paragraph=f"(auto-proposed from target reaction: {target_reaction_smiles})",
    )

    # Overall confidence: the WEAKEST required-slot confidence governs.
    required_slot_confs = [
        slot_confidences[s.name] for s in skeleton.slots
        if s.required and s.name in slot_confidences
    ]
    overall = _aggregate_confidence(required_slot_confs)

    # Channel reports — one per channel with proposal count.
    channel_reports: list[ChannelReport] = []
    for ch, hits in hits_by_channel.items():
        unique_sources = len({h.record.source for h in hits})
        if not hits:
            lvl = ConfidenceLevel.LOW
            note = "channel returned 0 hits"
        elif unique_sources >= 2:
            lvl = ConfidenceLevel.HIGH
            note = ""
        elif len(hits) >= 2:
            lvl = ConfidenceLevel.MEDIUM
            note = ""
        else:
            lvl = ConfidenceLevel.LOW
            note = "single-hit channel; consider broader retrieval"
        channel_reports.append(ChannelReport(channel=ch, n_proposals=len(hits),
                                             confidence=lvl, note=note))

    return ProposalResult(
        draft=draft,
        skeleton_class=skeleton.reaction_class.value,
        slot_confidences=slot_confidences,
        slot_provenance=slot_provenance,
        channel_reports=channel_reports,
        overall_confidence=overall,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_reactant_smiles(rxn_smi: str) -> list[str]:
    if rxn_smi.count(">") != 2:
        return []
    left, _middle, _right = rxn_smi.split(">")
    return [s for s in left.split(".") if s]


def _extract_product_smiles(rxn_smi: str) -> list[str]:
    if rxn_smi.count(">") != 2:
        return []
    _left, _middle, right = rxn_smi.split(">")
    return [s for s in right.split(".") if s]


def _midpoint(rng: tuple[float, float] | None) -> float | None:
    if rng is None:
        return None
    return (rng[0] + rng[1]) / 2


def _vote_temperature(hits_by_channel: dict[str, list[Hit]]) -> float | None:
    proposals: list[SlotProposal[float]] = []
    for ch_name, hits in hits_by_channel.items():
        for hit in hits:
            t = _temperature_from_hit(hit)
            if t is not None:
                proposals.append(SlotProposal(
                    value=t, source=hit.record.source,
                    source_id=f"{ch_name}:{hit.record.source_id}",
                    weight=max(hit.similarity, 0.1),
                ))
    if not proposals:
        return None
    # Weighted average for numeric slots.
    total_w = sum(p.weight for p in proposals)
    return sum(p.value * p.weight for p in proposals) / total_w if total_w else None


def _vote_duration(hits_by_channel: dict[str, list[Hit]]) -> float | None:
    proposals: list[SlotProposal[float]] = []
    for ch_name, hits in hits_by_channel.items():
        for hit in hits:
            d = _duration_from_hit(hit)
            if d is not None:
                proposals.append(SlotProposal(
                    value=d, source=hit.record.source,
                    source_id=f"{ch_name}:{hit.record.source_id}",
                    weight=max(hit.similarity, 0.1),
                ))
    if not proposals:
        return None
    total_w = sum(p.weight for p in proposals)
    return sum(p.value * p.weight for p in proposals) / total_w if total_w else None


def _vote_atmosphere(hits_by_channel: dict[str, list[Hit]]) -> str | None:
    proposals: list[SlotProposal[str]] = []
    for ch_name, hits in hits_by_channel.items():
        for hit in hits:
            a = _atmosphere_from_hit(hit)
            if a:
                proposals.append(SlotProposal(
                    value=a.lower(), source=hit.record.source,
                    source_id=f"{ch_name}:{hit.record.source_id}",
                    weight=max(hit.similarity, 0.1),
                ))
    consensus = multi_source_vote(proposals)
    return consensus.value


def _aggregate_confidence(levels: list[ConfidenceLevel]) -> ConfidenceLevel:
    """Overall confidence = the worst per-slot required confidence."""
    if not levels:
        return ConfidenceLevel.LOW
    order = [
        ConfidenceLevel.SPECULATIVE,
        ConfidenceLevel.LOW,
        ConfidenceLevel.MEDIUM,
        ConfidenceLevel.HIGH,
    ]
    return min(levels, key=order.index)


__all__ = [
    "ProposalResult",
    "compose_protocol",
]
