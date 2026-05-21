"""Unit tests for the benchmark canonical projection, scoring, and runner.

All tests are offline: no LLM calls, no model downloads.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from eln_structurer.benchmarks.adapters import ADAPTERS
from eln_structurer.benchmarks.adapters.base import (
    Adapter,
)
from eln_structurer.benchmarks.canonical import (
    CanonicalReaction,
    canonicalize_draft,
    canonicalize_ord_json,
    load_gold,
    normalize_name,
)
from eln_structurer.benchmarks.runner import (
    CaseRun,
    discover_fixtures,
    render_markdown_report,
    run_benchmark,
)
from eln_structurer.benchmarks.scoring import macro_f1, score_against_gold
from eln_structurer.proto_bridge import draft_to_proto, serialize_reaction
from eln_structurer.schema import ReactionDraft

FIXTURES = Path(__file__).parent / "fixtures"


def test_normalize_name() -> None:
    assert normalize_name("Salicylic Acid!") == "salicylic acid"
    assert normalize_name("  K2CO3 ") == "k2co3"
    assert normalize_name("4-Bromoanisole") == "4-bromoanisole"


def test_load_gold_files_parse() -> None:
    for path in (FIXTURES / "golden").glob("*.gold.json"):
        gold = load_gold(path)
        assert isinstance(gold, CanonicalReaction)


def test_canonicalize_draft_round_trip(aspirin_draft: ReactionDraft) -> None:
    canon = canonicalize_draft(aspirin_draft)
    assert "salicylic acid" in canon.reactant_names
    assert any("OC(=O)c1ccccc1O".lower() in s.lower() or "O=C(O)c1ccccc1O".lower() in s.lower()
               for s in canon.reactant_smiles)
    # Acetic anhydride is the acyl donor (REAGENT), not the SOLVENT.
    assert "acetic anhydride" in canon.reagent_names
    assert "sulfuric acid" in canon.catalyst_names
    assert canon.yield_percent == 90.0
    assert canon.temperature_celsius == 85.0
    assert canon.duration_minutes == 30.0
    assert canon.workup_verbs == ["ADDITION", "FILTRATION", "WASH"]


def test_canonicalize_ord_json_matches_draft(aspirin_draft: ReactionDraft) -> None:
    """Round-trip: draft → proto → JSON → canonical should match draft → canonical."""
    canon_from_draft = canonicalize_draft(aspirin_draft)
    proto = draft_to_proto(aspirin_draft)
    json_text = serialize_reaction(proto, fmt="json")
    canon_from_json = canonicalize_ord_json(json_text)
    assert canon_from_json.reactant_names == canon_from_draft.reactant_names
    assert canon_from_json.solvent_names == canon_from_draft.solvent_names
    assert canon_from_json.catalyst_names == canon_from_draft.catalyst_names
    assert canon_from_json.yield_percent == canon_from_draft.yield_percent
    assert canon_from_json.temperature_celsius == canon_from_draft.temperature_celsius


def test_scoring_perfect_match() -> None:
    a = CanonicalReaction(
        reactant_names={"foo", "bar"},
        product_names={"baz"},
        yield_percent=90.0,
        temperature_celsius=25.0,
        duration_minutes=60.0,
        workup_verbs=["WASH", "DRY_WITH_MATERIAL"],
    )
    scores = score_against_gold(a, a)
    assert macro_f1(scores) == pytest.approx(1.0)


def test_scoring_partial_match() -> None:
    predicted = CanonicalReaction(
        reactant_names={"foo"},
        product_names={"baz"},
        yield_percent=90.0,
    )
    gold = CanonicalReaction(
        reactant_names={"foo", "bar"},
        product_names={"baz"},
        yield_percent=90.0,
    )
    scores = score_against_gold(predicted, gold)
    by_name = {s.field_name: s for s in scores}
    assert by_name["reactant_names"].precision == pytest.approx(1.0)
    assert by_name["reactant_names"].recall == pytest.approx(0.5)
    assert by_name["product_names"].f1 == pytest.approx(1.0)
    assert by_name["yield_percent"].f1 == pytest.approx(1.0)


def test_scoring_scalar_tolerance() -> None:
    predicted = CanonicalReaction(yield_percent=91.0)
    gold = CanonicalReaction(yield_percent=90.0)
    scores = {s.field_name: s for s in score_against_gold(predicted, gold)}
    # 91 is within 5% of 90 → match.
    assert scores["yield_percent"].f1 == 1.0


def test_scoring_scalar_outside_tolerance() -> None:
    predicted = CanonicalReaction(yield_percent=80.0)
    gold = CanonicalReaction(yield_percent=90.0)
    scores = {s.field_name: s for s in score_against_gold(predicted, gold)}
    assert scores["yield_percent"].f1 == 0.0


def test_discover_fixtures_finds_three() -> None:
    cases = discover_fixtures(FIXTURES / "paragraphs", FIXTURES / "golden")
    names = {c.name for c in cases}
    assert {"aspirin", "suzuki_coupling", "grignard"} <= names


def test_paragraph2actions_unavailable_in_base_env() -> None:
    adapter = ADAPTERS["paragraph2actions"]()
    available = asyncio.run(adapter.is_available())
    # Should be False in our base venv per the documented constraint.
    assert available is False


def test_openchemie_unavailable_in_base_env() -> None:
    adapter = ADAPTERS["openchemie"]()
    available = asyncio.run(adapter.is_available())
    assert available is False


def test_runner_marks_unavailable_adapter() -> None:
    cases = discover_fixtures(FIXTURES / "paragraphs", FIXTURES / "golden")
    runs = asyncio.run(run_benchmark(cases[:1], ["paragraph2actions"]))
    assert len(runs) == 1
    assert runs[0].success is False
    assert "UNAVAILABLE" in runs[0].error


def test_render_markdown_report_no_runs() -> None:
    assert "No benchmark runs" in render_markdown_report([])


class _SyntheticAdapter(Adapter):
    name = "synthetic"

    def __init__(self, prediction: CanonicalReaction) -> None:
        self.prediction = prediction

    async def is_available(self) -> bool:
        return True

    async def extract(self, paragraph: str) -> CanonicalReaction:
        return self.prediction


def test_render_markdown_report_with_synthetic_runs() -> None:
    cases = discover_fixtures(FIXTURES / "paragraphs", FIXTURES / "golden")
    # Build a perfect-match CaseRun synthetically.
    case = cases[0]
    scores = score_against_gold(case.gold, case.gold)
    runs = [
        CaseRun(
            fixture=case.name,
            adapter="perfect",
            success=True,
            error=None,
            elapsed_seconds=0.0,
            macro_f1=macro_f1(scores),
            field_scores=scores,
        )
    ]
    report = render_markdown_report(runs)
    assert "Benchmark report" in report
    assert "perfect" in report
    assert case.name in report
