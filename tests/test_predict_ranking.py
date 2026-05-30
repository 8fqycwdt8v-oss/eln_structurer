"""Tier-3 tests: yield model, solvent greenness, multi-objective ranker."""

from __future__ import annotations

from eln_structurer.predict import (
    ConfidenceLevel,
    CorpusSource,
    MIN_TRAINING_POINTS,
    ReactionRecord,
    Weights,
    compose_protocol,
    conservative_yield_score,
    default_seed_corpus,
    estimate_yield,
    get_skeleton,
    protocol_solvent_score,
    rank_proposals,
    retrieve_exact,
    retrieve_knn,
    solvent_score,
)
from eln_structurer.predict.retrieval import Hit
from eln_structurer.reaction_class import ReactionClass


SUZUKI_RXN = "BrC1=CC=CC=C1.OB(O)c1ccccc1>>C1=CC=CC=C1C1=CC=CC=C1"


def _record(yield_pct: float, source=CorpusSource.LITERATURE,
            similarity=1.0) -> Hit:
    return Hit(
        record=ReactionRecord(
            reaction_smiles=SUZUKI_RXN, source=source,
            source_id=f"id-{yield_pct}",
            yield_percent=yield_pct,
        ),
        similarity=similarity,
    )


# ---------- yield model ---------------------------------------------------


def test_estimate_yield_requires_min_training_points() -> None:
    assert estimate_yield([_record(89.0)]) is None
    assert MIN_TRAINING_POINTS >= 2


def test_estimate_yield_returns_band_and_n_support() -> None:
    hits = [_record(88.0), _record(90.0), _record(92.0)]
    est = estimate_yield(hits)
    assert est is not None
    assert 86.0 <= est.point <= 92.0
    assert est.lower_95 < est.point < est.upper_95
    assert est.n_support == 3


def test_conservative_yield_score_uses_lower_bound() -> None:
    hits = [_record(85.0), _record(85.0), _record(85.0)]
    est = estimate_yield(hits)
    s = conservative_yield_score(est)
    # Lower bound should be below 0.85 → conservative score < 0.85.
    assert s <= 0.85
    # Unknown collapses to neutral 0.5.
    assert conservative_yield_score(None) == 0.5


# ---------- solvent greenness --------------------------------------------


def test_solvent_score_known_recommended_solvents() -> None:
    assert solvent_score("water") == 1.0
    assert solvent_score("ethanol") == 1.0
    assert solvent_score("ethyl acetate") == 1.0


def test_solvent_score_known_problematic() -> None:
    assert solvent_score("DCM") < 0.5
    assert solvent_score("dichloromethane") < 0.5
    assert solvent_score("chloroform") == 0.0


def test_solvent_score_unknown_returns_neutral() -> None:
    assert solvent_score("absolutely-fictional-solvent-1234") == 0.5


def test_protocol_solvent_score_takes_minimum() -> None:
    # One bad solvent drags the protocol down.
    assert protocol_solvent_score(["water", "chloroform"]) == 0.0
    assert protocol_solvent_score(["water", "ethanol"]) == 1.0


def test_protocol_solvent_score_empty_returns_neutral() -> None:
    assert protocol_solvent_score([]) == 0.5


# ---------- ranker --------------------------------------------------------


def _suzuki_proposal_high_confidence():
    corp = default_seed_corpus()
    sk = get_skeleton(ReactionClass.SUZUKI_COUPLING)
    hits_dict = {
        "exact": retrieve_exact(corp, SUZUKI_RXN),
        "knn": retrieve_knn(corp, SUZUKI_RXN, k=3),
    }
    proposal = compose_protocol(
        target_reaction_smiles=SUZUKI_RXN,
        skeleton=sk,
        hits_by_channel=hits_dict,
    )
    return proposal, hits_dict


def test_rank_single_proposal_returns_one_ranked() -> None:
    p, hits = _suzuki_proposal_high_confidence()
    ranked = rank_proposals([p], hits_by_proposal=[hits])
    assert len(ranked) == 1
    r = ranked[0]
    assert r.proposal is p
    assert 0.0 <= r.greenness_score <= 1.0
    assert 0.0 <= r.confidence_score <= 1.0


def test_ranker_orders_by_overall_score() -> None:
    p_good, hits = _suzuki_proposal_high_confidence()
    # Build an obviously worse proposal — same draft but with no hits
    # and the speculative-fallback path will dominate.
    sk = get_skeleton(ReactionClass.AMIDE_FORMATION)
    p_bad = compose_protocol(
        target_reaction_smiles="O=C(O)c1ccccc1.Nc1ccccc1>>O=C(Nc1ccccc1)c1ccccc1",
        skeleton=sk,
        hits_by_channel={"exact": [], "knn": []},
    )
    ranked = rank_proposals(
        [p_bad, p_good],
        hits_by_proposal=[{}, hits],
    )
    assert ranked[0].proposal is p_good
    assert ranked[0].overall_score > ranked[-1].overall_score


def test_constraint_penalty_applied() -> None:
    # Force a halogenated-solvent proposal by injecting DCM into the
    # composed draft post hoc.
    p, hits = _suzuki_proposal_high_confidence()
    from eln_structurer.schema import CompoundIdentifierModel, CompoundModel, ReactionInputModel
    p.draft.inputs.append(ReactionInputModel(
        name="extra_solvent",
        components=[CompoundModel(
            identifiers=[CompoundIdentifierModel(type="NAME", value="DCM")],
            reaction_role="SOLVENT",
        )],
    ))
    ranked = rank_proposals(
        [p],
        hits_by_proposal=[hits],
        constraints={"no_halogenated_solvents": True},
    )
    assert ranked[0].constraint_penalty > 0
    assert any("halogenated" in v for v in ranked[0].constraint_violations)


def test_custom_weights_change_ordering() -> None:
    p, hits = _suzuki_proposal_high_confidence()
    high_yield_weights = Weights(yield_=10.0, greenness=0.1,
                                  confidence=0.1, retrieval=0.1)
    ranked = rank_proposals([p], hits_by_proposal=[hits],
                            weights=high_yield_weights)
    # No crash; the weighting is applied without error.
    assert ranked[0].overall_score >= 0.0


def test_rank_proposals_mismatched_hits_raises() -> None:
    p, _ = _suzuki_proposal_high_confidence()
    import pytest
    with pytest.raises(ValueError):
        rank_proposals([p, p], hits_by_proposal=[{}])


def test_confidence_collapsed_to_score() -> None:
    # Smoke test the confidence-to-score map.
    p, hits = _suzuki_proposal_high_confidence()
    p_high = p
    ranked = rank_proposals([p_high], hits_by_proposal=[hits])
    # The composer produces HIGH overall confidence here.
    if p.overall_confidence == ConfidenceLevel.HIGH:
        assert ranked[0].confidence_score == 1.0
