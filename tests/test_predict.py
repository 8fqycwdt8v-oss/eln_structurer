"""Offline tests for the Tier-1 predictor primitives.

No LLM calls, no network. Exercises corpus / fingerprint / retrieval /
multi-source vote / safety screen / recency check.
"""

from __future__ import annotations

import pytest

from eln_structurer.predict import (
    ConfidenceLevel,
    CorpusSource,
    Hit,
    LocalCorpus,
    ReactionRecord,
    SafetyVerdict,
    SlotProposal,
    classifier_must_be_confident,
    hard_constraint_filter,
    multi_source_vote,
    reaction_fingerprint,
    recency_summary,
    retrieve_exact,
    retrieve_knn,
    safety_screen,
    summarise_channels,
    tanimoto,
)
from eln_structurer.predict.retrieval import min_year, no_halogenated_solvents
from eln_structurer.reaction_class import ReactionClass


# ---------- fingerprint ----------------------------------------------------


SUZUKI = "BrC1=CC=CC=C1.OB(O)c1ccccc1>>C1=CC=CC=C1C1=CC=CC=C1"
SUZUKI_VAR = "Brc1ccc(C)cc1.OB(O)c1ccccc1>>C(c1ccccc1)c1ccc(C)cc1"
GRIGNARD = "BrC1=CC=CC=C1.[Mg]>>BrMgC1=CC=CC=C1"


def test_fingerprint_self_tanimoto_is_one() -> None:
    fp = reaction_fingerprint(SUZUKI)
    assert fp  # non-empty
    assert tanimoto(fp, fp) == pytest.approx(1.0)


def test_fingerprint_similar_reactions_score_above_unrelated() -> None:
    a = reaction_fingerprint(SUZUKI)
    b = reaction_fingerprint(SUZUKI_VAR)
    c = reaction_fingerprint(GRIGNARD)
    assert tanimoto(a, b) > tanimoto(a, c)


def test_fingerprint_malformed_returns_empty() -> None:
    assert reaction_fingerprint("not a reaction") == {}


# ---------- corpus + retrieval --------------------------------------------


def _record(smi: str, source_id: str, *, source=CorpusSource.LITERATURE,
            year=2020, solvents=None) -> ReactionRecord:
    return ReactionRecord(
        reaction_smiles=smi,
        source=source,
        source_id=source_id,
        year=year,
        conditions={"solvents": solvents or []},
        procedure_text=None,
        yield_percent=None,
    )


def test_corpus_idempotent_add() -> None:
    corp = LocalCorpus()
    r = _record(SUZUKI, "lit:1")
    corp.add(r)
    corp.add(r)  # silent no-op
    assert len(corp) == 1


def test_retrieve_exact_finds_all_records_same_smiles() -> None:
    corp = LocalCorpus()
    corp.add(_record(SUZUKI, "lit:1"))
    corp.add(_record(SUZUKI, "hte:doyle:42", source=CorpusSource.HTE))
    corp.add(_record(GRIGNARD, "lit:2"))
    hits = retrieve_exact(corp, SUZUKI)
    assert len(hits) == 2
    assert all(h.similarity == 1.0 for h in hits)
    assert {h.record.source for h in hits} == {CorpusSource.LITERATURE, CorpusSource.HTE}


def test_retrieve_knn_orders_similar_first() -> None:
    corp = LocalCorpus()
    corp.add(_record(SUZUKI_VAR, "lit:1"))
    corp.add(_record(GRIGNARD, "lit:2"))
    hits = retrieve_knn(corp, SUZUKI, k=2)
    assert len(hits) == 2
    # The Suzuki variant must come first.
    assert hits[0].record.source_id == "lit:1"
    assert hits[0].similarity > hits[1].similarity


def test_retrieve_knn_respects_hard_constraints() -> None:
    corp = LocalCorpus()
    corp.add(_record(SUZUKI_VAR, "lit:dcm",
                     solvents=["DCM"]))                # halogenated
    corp.add(_record(SUZUKI_VAR, "lit:tol",
                     solvents=["toluene"]))            # not halogenated
    hits = retrieve_knn(corp, SUZUKI, k=5,
                       filters=[no_halogenated_solvents])
    assert {h.record.source_id for h in hits} == {"lit:tol"}


def test_retrieve_knn_respects_year_filter() -> None:
    corp = LocalCorpus()
    corp.add(_record(SUZUKI_VAR, "old", year=1995))
    corp.add(_record(SUZUKI_VAR, "new", year=2024))
    hits = retrieve_knn(corp, SUZUKI, k=5, filters=[min_year(2010)])
    assert {h.record.source_id for h in hits} == {"new"}


def test_hard_constraint_filter_builder_from_dict() -> None:
    filters = hard_constraint_filter({
        "no_halogenated_solvents": True,
        "min_year": 2010,
        "allowed_sources": ["literature"],
    })
    assert len(filters) == 3


# ---------- multi-source vote ---------------------------------------------


def test_multi_source_vote_no_proposals_is_low() -> None:
    c = multi_source_vote([])
    assert c.confidence is ConfidenceLevel.LOW
    assert c.value is None


def test_multi_source_vote_two_independent_sources_is_high() -> None:
    props = [
        SlotProposal(value="toluene", source=CorpusSource.LITERATURE, source_id="lit:1"),
        SlotProposal(value="toluene", source=CorpusSource.HTE,        source_id="hte:1"),
    ]
    c = multi_source_vote(props)
    assert c.confidence is ConfidenceLevel.HIGH
    assert c.value == "toluene"
    assert set(c.agreeing_sources) == {CorpusSource.LITERATURE, CorpusSource.HTE}


def test_multi_source_vote_single_source_multiple_proposals_is_medium() -> None:
    props = [
        SlotProposal(value="toluene", source=CorpusSource.LITERATURE, source_id="lit:1"),
        SlotProposal(value="toluene", source=CorpusSource.LITERATURE, source_id="lit:2"),
    ]
    c = multi_source_vote(props)
    assert c.confidence is ConfidenceLevel.MEDIUM


def test_multi_source_vote_dissent_recorded() -> None:
    props = [
        SlotProposal(value="toluene", source=CorpusSource.LITERATURE, source_id="lit:1"),
        SlotProposal(value="toluene", source=CorpusSource.HTE,        source_id="hte:1"),
        SlotProposal(value="dioxane", source=CorpusSource.LITERATURE, source_id="lit:2"),
    ]
    c = multi_source_vote(props)
    assert c.value == "toluene"
    assert len(c.dissent) == 1
    assert c.dissent[0].value == "dioxane"


def test_summarise_channels_reports_per_channel_state() -> None:
    by_channel = {
        "exact": [],
        "knn": [SlotProposal(value="dmf", source=CorpusSource.LITERATURE, source_id="x")],
        "hte":  [
            SlotProposal(value="dmf", source=CorpusSource.HTE, source_id="h"),
            SlotProposal(value="dmf", source=CorpusSource.LITERATURE, source_id="y"),
        ],
    }
    reports = {r.channel: r for r in summarise_channels(by_channel)}
    assert reports["exact"].n_proposals == 0
    assert reports["knn"].confidence is ConfidenceLevel.LOW
    assert reports["hte"].confidence is ConfidenceLevel.HIGH


# ---------- recency --------------------------------------------------------


def test_recency_summary_warns_when_stale() -> None:
    hits = [Hit(record=_record(SUZUKI, f"id:{i}", year=1990), similarity=1.0)
            for i in range(3)]
    summary = recency_summary(hits, reference_year=2025)
    assert summary.warning is not None
    assert "median" in summary.warning


def test_recency_summary_silent_when_fresh() -> None:
    hits = [Hit(record=_record(SUZUKI, f"id:{i}", year=2022), similarity=1.0)
            for i in range(3)]
    summary = recency_summary(hits, reference_year=2025)
    assert summary.warning is None
    assert summary.median_year == 2022


def test_recency_summary_handles_missing_years() -> None:
    hits = [Hit(record=_record(SUZUKI, "id:1", year=None), similarity=1.0)]
    summary = recency_summary(hits, reference_year=2025)
    assert summary.n_with_year == 0
    assert "no publication-year metadata" in (summary.warning or "")


# ---------- classifier confidence gate ------------------------------------


def test_classifier_must_be_confident_passes_for_suzuki(aspirin_draft) -> None:
    # aspirin_draft is not a Suzuki — classifier should return UNKNOWN
    # with low confidence, so the gate must report not-confident.
    cls, confident, score = classifier_must_be_confident(aspirin_draft)
    assert cls is ReactionClass.UNKNOWN
    assert confident is False
    assert score == 0.0


# ---------- safety screen --------------------------------------------------


def test_safety_screen_blocks_controlled_chemical(aspirin_draft) -> None:
    # Inject a controlled chemical name.
    from eln_structurer.schema import CompoundIdentifierModel
    aspirin_draft.inputs[0].components[0].identifiers.append(
        CompoundIdentifierModel(type="NAME", value="potassium cyanide")
    )
    report = safety_screen(aspirin_draft)
    assert report.verdict is SafetyVerdict.BLOCKED
    assert any("CONTROLLED" in f for f in report.flags)


def test_safety_screen_warns_on_peroxide_former(aspirin_draft) -> None:
    from eln_structurer.schema import CompoundIdentifierModel
    aspirin_draft.inputs[0].components[0].identifiers.append(
        CompoundIdentifierModel(type="NAME", value="THF")
    )
    report = safety_screen(aspirin_draft)
    assert report.verdict is SafetyVerdict.WARN
    assert any("PEROXIDE_FORMER" in f for f in report.flags)


def test_safety_screen_warns_on_high_risk_substring(aspirin_draft) -> None:
    from eln_structurer.schema import CompoundIdentifierModel
    aspirin_draft.inputs[0].components[0].identifiers.append(
        CompoundIdentifierModel(type="NAME", value="sodium azide")
    )
    report = safety_screen(aspirin_draft)
    assert report.verdict is SafetyVerdict.WARN
    assert any("azide" in f for f in report.flags)


def test_safety_screen_silent_on_clean_draft(aspirin_draft) -> None:
    report = safety_screen(aspirin_draft)
    assert report.verdict is SafetyVerdict.OK
    assert report.flags == []


# ---------- MCP tool wrappers ---------------------------------------------


async def test_retrieve_exact_tool_with_no_corpus_explains() -> None:
    from eln_structurer.tools.predict_tools import (
        retrieve_exact_reaction,
        set_active_corpus,
    )
    set_active_corpus(None)
    result = await retrieve_exact_reaction.handler({"reaction_smiles": SUZUKI})
    assert "NO_CORPUS" in result["content"][0]["text"]


async def test_retrieve_exact_tool_finds_record() -> None:
    from eln_structurer.tools.predict_tools import (
        retrieve_exact_reaction,
        set_active_corpus,
    )
    corp = LocalCorpus()
    corp.add(_record(SUZUKI, "lit:42"))
    set_active_corpus(corp)
    try:
        result = await retrieve_exact_reaction.handler({"reaction_smiles": SUZUKI})
        text = result["content"][0]["text"]
        assert "exact match" in text
        assert "lit:42" in text
    finally:
        set_active_corpus(None)


async def test_retrieve_similar_tool_applies_constraints() -> None:
    from eln_structurer.tools.predict_tools import (
        retrieve_similar_reactions,
        set_active_corpus,
    )
    corp = LocalCorpus()
    corp.add(_record(SUZUKI_VAR, "lit:dcm", solvents=["DCM"]))
    corp.add(_record(SUZUKI_VAR, "lit:tol", solvents=["toluene"]))
    set_active_corpus(corp)
    try:
        result = await retrieve_similar_reactions.handler({
            "reaction_smiles": SUZUKI,
            "k": 5,
            "constraints": {"no_halogenated_solvents": True},
        })
        text = result["content"][0]["text"]
        assert "lit:tol" in text
        assert "lit:dcm" not in text
    finally:
        set_active_corpus(None)


async def test_safety_tool_blocks_controlled_chemical(aspirin_draft) -> None:
    from eln_structurer.schema import CompoundIdentifierModel
    from eln_structurer.tools.predict_tools import safety_screen_tool

    aspirin_draft.inputs[0].components[0].identifiers.append(
        CompoundIdentifierModel(type="NAME", value="phosgene")
    )
    payload = aspirin_draft.model_dump(mode="json")
    result = await safety_screen_tool.handler({"draft_json": payload})
    assert result.get("isError") is True
    assert "blocked" in result["content"][0]["text"]


async def test_safety_tool_ok_for_clean_draft(aspirin_draft) -> None:
    from eln_structurer.tools.predict_tools import safety_screen_tool

    payload = aspirin_draft.model_dump(mode="json")
    result = await safety_screen_tool.handler({"draft_json": payload})
    assert result.get("isError") is not True
    assert "ok" in result["content"][0]["text"]
