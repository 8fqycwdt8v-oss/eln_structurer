"""Field-level precision/recall scoring against a gold CanonicalReaction."""

from __future__ import annotations

from dataclasses import dataclass

from eln_structurer.benchmarks.canonical import CanonicalReaction


@dataclass
class FieldScore:
    field_name: str
    precision: float
    recall: float
    f1: float
    support: int  # |gold|


def _prf(predicted: set, gold: set) -> tuple[float, float, float, int]:
    tp = len(predicted & gold)
    fp = len(predicted - gold)
    fn = len(gold - predicted)
    p = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if not gold else 0.0)
    r = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1, len(gold)


def _scalar_match(predicted: float | None, gold: float | None, tolerance: float) -> tuple[float, float, float, int]:
    if gold is None and predicted is None:
        return 1.0, 1.0, 1.0, 0
    if gold is None:
        # No gold expectation but we predicted something — count as 0 support.
        return 0.0, 1.0, 0.0, 0
    if predicted is None:
        return 0.0, 0.0, 0.0, 1
    diff = abs(predicted - gold)
    ok = diff <= tolerance * max(1.0, abs(gold))
    p = r = f = 1.0 if ok else 0.0
    return p, r, f, 1


def _sequence_match(predicted: list[str], gold: list[str]) -> tuple[float, float, float, int]:
    """Score ordered workup verbs as a (multi-)set match.

    We score as a multiset because rough sequence agreement matters more than
    exact ordering for cross-tool comparison.
    """
    from collections import Counter
    if not gold and not predicted:
        return 1.0, 1.0, 1.0, 0
    if not gold:
        return 0.0, 1.0, 0.0, 0
    pc = Counter(predicted)
    gc = Counter(gold)
    tp = sum((pc & gc).values())
    fp = sum((pc - gc).values())
    fn = sum((gc - pc).values())
    p = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f, sum(gc.values())


def score_against_gold(
    predicted: CanonicalReaction, gold: CanonicalReaction
) -> list[FieldScore]:
    """Return per-field FieldScore objects comparing predicted to gold."""
    scores: list[FieldScore] = []

    for name, get in (
        ("reactant_names", lambda c: c.reactant_names),
        ("reactant_smiles", lambda c: c.reactant_smiles),
        ("reagent_names", lambda c: c.reagent_names),
        ("solvent_names", lambda c: c.solvent_names),
        ("catalyst_names", lambda c: c.catalyst_names),
        ("product_names", lambda c: c.product_names),
        ("product_smiles", lambda c: c.product_smiles),
    ):
        p, r, f, n = _prf(get(predicted), get(gold))
        scores.append(FieldScore(name, p, r, f, n))

    from eln_structurer.config import DEFAULT_BENCHMARK_CONFIG as _BC

    for name, get, tol in (
        ("yield_percent", lambda c: c.yield_percent, _BC.yield_tolerance),
        ("temperature_celsius", lambda c: c.temperature_celsius, _BC.temperature_tolerance),
        ("duration_minutes", lambda c: c.duration_minutes, _BC.duration_tolerance),
    ):
        p, r, f, n = _scalar_match(get(predicted), get(gold), tol)
        scores.append(FieldScore(name, p, r, f, n))

    p, r, f, n = _sequence_match(predicted.workup_verbs, gold.workup_verbs)
    scores.append(FieldScore("workup_verbs", p, r, f, n))
    return scores


def macro_f1(scores: list[FieldScore]) -> float:
    if not scores:
        return 0.0
    return sum(s.f1 for s in scores) / len(scores)
