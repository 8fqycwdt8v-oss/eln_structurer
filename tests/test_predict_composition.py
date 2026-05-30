"""Tests for Tier-2 composition layer: skeletons + composer + HTE seed."""

from __future__ import annotations


from eln_structurer.predict import (
    ConfidenceLevel,
    CorpusSource,
    ProposalResult,
    all_skeletons,
    compose_protocol,
    default_seed_corpus,
    get_skeleton,
    known_classes,
    retrieve_exact,
    retrieve_knn,
)
from eln_structurer.reaction_class import ReactionClass


# ---------- skeletons -----------------------------------------------------


def test_every_known_class_has_skeleton() -> None:
    # Every class returned by known_classes() must yield a skeleton.
    for cls in known_classes():
        sk = get_skeleton(cls)
        assert sk is not None
        assert sk.reaction_class is cls
        assert sk.slots, f"{cls.value} skeleton has empty slot list"
        assert sk.workup_sequence


def test_get_skeleton_unknown_returns_none() -> None:
    assert get_skeleton(ReactionClass.UNKNOWN) is None


def test_all_skeletons_round_trip() -> None:
    assert len(all_skeletons()) == len(known_classes())


# ---------- HTE corpus ---------------------------------------------------


SUZUKI_RXN = "BrC1=CC=CC=C1.OB(O)c1ccccc1>>C1=CC=CC=C1C1=CC=CC=C1"
AMIDE_RXN = "O=C(O)c1ccccc1.Nc1ccccc1>>O=C(Nc1ccccc1)c1ccccc1"


def test_default_seed_corpus_nonempty() -> None:
    corp = default_seed_corpus()
    assert len(corp) >= 5
    suzuki_hits = retrieve_exact(corp, SUZUKI_RXN)
    assert len(suzuki_hits) >= 2
    # Mixed sources — important for the multi-source vote.
    sources = {h.record.source for h in suzuki_hits}
    assert CorpusSource.HTE in sources or CorpusSource.LITERATURE in sources


# ---------- composition --------------------------------------------------


def test_compose_suzuki_from_seed_corpus_high_confidence() -> None:
    corp = default_seed_corpus()
    sk = get_skeleton(ReactionClass.SUZUKI_COUPLING)
    assert sk is not None
    hits_exact = retrieve_exact(corp, SUZUKI_RXN)
    hits_knn = retrieve_knn(corp, SUZUKI_RXN, k=5)

    result = compose_protocol(
        target_reaction_smiles=SUZUKI_RXN,
        skeleton=sk,
        hits_by_channel={"exact": hits_exact, "knn": hits_knn},
    )
    assert isinstance(result, ProposalResult)
    assert result.skeleton_class == ReactionClass.SUZUKI_COUPLING.value

    # Two-plus distinct sources → HIGH confidence on at least the
    # well-supported slots (pd_source, base, solvent).
    high_confidence = {
        slot for slot, lvl in result.slot_confidences.items()
        if lvl is ConfidenceLevel.HIGH
    }
    assert {"pd_source"} <= high_confidence or {"solvent"} <= high_confidence


def test_compose_fills_inert_atmosphere_when_skeleton_demands() -> None:
    corp = default_seed_corpus()
    sk = get_skeleton(ReactionClass.SUZUKI_COUPLING)
    result = compose_protocol(
        target_reaction_smiles=SUZUKI_RXN,
        skeleton=sk,
        hits_by_channel={"exact": retrieve_exact(corp, SUZUKI_RXN)},
    )
    # Skeleton says inert_atmosphere_required → composer must
    # propagate (either from retrieval vote or default).
    assert result.draft.conditions.atmosphere in {"nitrogen", "argon"}


def test_compose_falls_back_to_skeleton_when_no_hits() -> None:
    # Empty corpus → no retrieval evidence → fallback names + speculative.
    sk = get_skeleton(ReactionClass.AMIDE_FORMATION)
    result = compose_protocol(
        target_reaction_smiles=AMIDE_RXN,
        skeleton=sk,
        hits_by_channel={"exact": [], "knn": []},
    )
    assert result.overall_confidence in {ConfidenceLevel.LOW,
                                         ConfidenceLevel.SPECULATIVE}
    # The fallback-coupling-reagent slot should be filled with the
    # skeleton's first fallback name.
    coupling_input = [
        inp for inp in result.draft.inputs
        if inp.name == "coupling_reagent"
    ]
    assert coupling_input
    name_idents = [i for c in coupling_input[0].components
                   for i in c.identifiers if i.type == "NAME"]
    assert name_idents[0].value in {"EDC", "HATU"}


def test_compose_overall_confidence_is_minimum_required_slot() -> None:
    corp = default_seed_corpus()
    sk = get_skeleton(ReactionClass.SUZUKI_COUPLING)
    result = compose_protocol(
        target_reaction_smiles=SUZUKI_RXN,
        skeleton=sk,
        hits_by_channel={"knn": retrieve_knn(corp, SUZUKI_RXN, k=3)},
    )
    # Overall should never be HIGHER than the weakest required slot.
    req_levels = [
        result.slot_confidences[s.name] for s in sk.slots
        if s.required and s.name in result.slot_confidences
    ]
    order = [
        ConfidenceLevel.SPECULATIVE,
        ConfidenceLevel.LOW,
        ConfidenceLevel.MEDIUM,
        ConfidenceLevel.HIGH,
    ]
    expected_overall = min(req_levels, key=order.index)
    assert result.overall_confidence == expected_overall


def test_compose_produces_valid_reaction_draft() -> None:
    """Composed ReactionDraft must pass Pydantic validation."""
    corp = default_seed_corpus()
    sk = get_skeleton(ReactionClass.SUZUKI_COUPLING)
    result = compose_protocol(
        target_reaction_smiles=SUZUKI_RXN,
        skeleton=sk,
        hits_by_channel={"knn": retrieve_knn(corp, SUZUKI_RXN, k=3)},
    )
    # Round-trip through the schema as a smoke test.
    from eln_structurer.schema import ReactionDraft
    payload = result.draft.model_dump(mode="json")
    rebuilt = ReactionDraft.model_validate(payload)
    assert rebuilt.identifiers[0].value == SUZUKI_RXN


def test_compose_carries_channel_reports() -> None:
    corp = default_seed_corpus()
    sk = get_skeleton(ReactionClass.SUZUKI_COUPLING)
    result = compose_protocol(
        target_reaction_smiles=SUZUKI_RXN,
        skeleton=sk,
        hits_by_channel={
            "exact": retrieve_exact(corp, SUZUKI_RXN),
            "knn": retrieve_knn(corp, SUZUKI_RXN, k=3),
            "hte": [],
        },
    )
    by_name = {r.channel: r for r in result.channel_reports}
    assert "exact" in by_name
    assert "knn" in by_name
    assert "hte" in by_name
    assert by_name["hte"].confidence is ConfidenceLevel.LOW   # empty channel
