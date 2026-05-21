"""Tests for the paired-comparison module."""

from __future__ import annotations

from eln_structurer.benchmarks.compare import (
    paired_comparison,
    render_comparison_report,
)


def _run(fixture: str, adapter: str, scores: dict[str, float]) -> dict:
    return {
        "fixture": fixture,
        "adapter": adapter,
        "success": True,
        "field_scores": [
            {"field_name": k, "precision": 1.0, "recall": 1.0, "f1": v, "support": 1}
            for k, v in scores.items()
        ],
    }


def test_empty_comparison_no_paired_samples() -> None:
    deltas, summary = paired_comparison([], [])
    assert summary.n == 0
    assert summary.p_value_two_sided == 1.0
    report = render_comparison_report(deltas, summary)
    assert "No paired" in report


def test_zero_delta_when_runs_match() -> None:
    runs = [_run("aspirin", "eln_structurer", {"reactant_names": 0.9, "product_names": 0.85})]
    deltas, summary = paired_comparison(runs, runs)
    assert summary.n == 2
    assert summary.mean_delta == 0.0
    assert summary.significant_at_95 is False


def test_positive_delta_detected() -> None:
    baseline = [_run("aspirin", "x", {"a": 0.5, "b": 0.6, "c": 0.7})]
    current = [_run("aspirin", "x", {"a": 0.6, "b": 0.7, "c": 0.8})]
    deltas, summary = paired_comparison(baseline, current)
    assert summary.n == 3
    assert summary.mean_delta > 0
    # All three deltas are positive and equal — should be flagged.
    for d in deltas:
        assert d.delta > 0


def test_unmatched_keys_ignored() -> None:
    """A (fixture, adapter, field) triple present in one run but not the
    other must be excluded from the paired analysis."""
    baseline = [_run("aspirin", "x", {"a": 0.5, "b": 0.6})]
    current = [_run("aspirin", "x", {"a": 0.6})]  # 'b' is missing
    deltas, summary = paired_comparison(baseline, current)
    # Only 'a' is paired.
    assert summary.n == 1
    assert deltas[0].field_name == "a"


def test_low_n_warning_in_report() -> None:
    baseline = [_run("aspirin", "x", {"a": 0.5})]
    current = [_run("aspirin", "x", {"a": 0.7})]
    _, summary = paired_comparison(baseline, current)
    report = render_comparison_report([], summary) if summary.n == 0 else render_comparison_report(
        paired_comparison(baseline, current)[0], summary
    )
    # With n=1 the report should warn about low power.
    if summary.n < 10:
        assert "low" in report.lower() or "n ≥" in report.lower() or "N ≥" in report


def test_fixture_files_match_paragraphs() -> None:
    """The five new gold fixtures must each have a matching paragraph."""
    from pathlib import Path
    fixtures_dir = Path(__file__).parent / "fixtures"
    paragraphs = {p.stem for p in (fixtures_dir / "paragraphs").glob("*.txt")}
    golds = {p.stem.replace(".gold", "") for p in (fixtures_dir / "golden").glob("*.gold.json")}
    # New fixtures we just added must be paired on both sides.
    for name in {
        "amide_coupling",
        "buchwald_hartwig",
        "reductive_amination",
        "boc_deprotection",
        "dmp_oxidation",
    }:
        assert name in paragraphs, f"missing paragraph for {name}"
        assert name in golds, f"missing gold for {name}"
