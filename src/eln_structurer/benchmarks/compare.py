"""Paired-difference comparison for benchmark snapshots.

Two benchmark runs against the same fixture set produce per-(fixture, field)
F1 vectors. The compare module computes the field-by-field delta, runs a
paired one-sample t-test on the differences, and emits a markdown report
with a verdict.

Stays self-contained — no scipy dependency. Implements the t-statistic and
a Student-t CDF approximation (Gauss-Legendre integration is overkill;
a small Cornish-Fisher style normal approximation is fine for N>=10 or
fallback to "insufficient samples" below that).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FieldComparison:
    fixture: str
    adapter: str
    field_name: str
    baseline_f1: float
    current_f1: float
    delta: float


@dataclass(frozen=True)
class StatisticalSummary:
    n: int
    mean_delta: float
    std_delta: float
    t_statistic: float
    p_value_two_sided: float
    significant_at_95: bool


def _student_t_two_sided_p(t: float, df: int) -> float:
    """Approximate two-sided p-value for a Student-t statistic.

    For df>=30 the standard normal is a good approximation. For df<30 we
    use a small-sample correction. Returns 1.0 when df<=0 (no inference).
    """
    if df <= 0 or not math.isfinite(t):
        return 1.0
    if df >= 30:
        # Standard normal tail; |t| → two-sided.
        return 2.0 * (1.0 - _phi(abs(t)))
    # Crude small-df adjustment — scale |t| toward the normal a bit.
    # Good enough for triage; explicit "insufficient samples" callers
    # should set N >= 10 anyway.
    adjusted = abs(t) * math.sqrt(df / (df + 1))
    return min(1.0, 2.0 * (1.0 - _phi(adjusted)))


def _phi(x: float) -> float:
    """Standard normal CDF via error function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def load_run_json(path: Path) -> list[dict]:
    """Load a serialized list of CaseRun dicts from a JSON file."""
    return json.loads(path.read_text())


def to_field_table(runs: list[dict]) -> dict[tuple[str, str, str], float]:
    """Flatten runs to {(fixture, adapter, field_name): f1}."""
    out: dict[tuple[str, str, str], float] = {}
    for r in runs:
        if not r.get("success"):
            continue
        fixture = r["fixture"]
        adapter = r["adapter"]
        for fs in r.get("field_scores", []):
            out[(fixture, adapter, fs["field_name"])] = fs["f1"]
    return out


def paired_comparison(
    baseline_runs: list[dict],
    current_runs: list[dict],
) -> tuple[list[FieldComparison], StatisticalSummary]:
    """Compute per-(fixture, adapter, field) deltas and aggregate stats.

    A field is included only when it has a successful score in BOTH runs
    for the same (fixture, adapter) pair — that's the paired requirement.
    """
    baseline = to_field_table(baseline_runs)
    current = to_field_table(current_runs)

    deltas: list[FieldComparison] = []
    for key, b_f1 in baseline.items():
        if key not in current:
            continue
        c_f1 = current[key]
        deltas.append(
            FieldComparison(
                fixture=key[0],
                adapter=key[1],
                field_name=key[2],
                baseline_f1=b_f1,
                current_f1=c_f1,
                delta=c_f1 - b_f1,
            )
        )

    if not deltas:
        return [], StatisticalSummary(
            n=0,
            mean_delta=0.0,
            std_delta=0.0,
            t_statistic=0.0,
            p_value_two_sided=1.0,
            significant_at_95=False,
        )

    values = [d.delta for d in deltas]
    n = len(values)
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / (n - 1) if n > 1 else 0.0
    std = math.sqrt(variance)
    se = std / math.sqrt(n) if n > 0 else 0.0
    t_stat = mean / se if se > 0 else 0.0
    p = _student_t_two_sided_p(t_stat, df=n - 1)
    return deltas, StatisticalSummary(
        n=n,
        mean_delta=mean,
        std_delta=std,
        t_statistic=t_stat,
        p_value_two_sided=p,
        significant_at_95=(p < 0.05) and (n >= 10),
    )


def render_comparison_report(
    deltas: list[FieldComparison],
    summary: StatisticalSummary,
    *,
    baseline_label: str = "baseline",
    current_label: str = "current",
) -> str:
    lines = [f"# Benchmark comparison: {baseline_label} → {current_label}", ""]
    if summary.n == 0:
        lines.append(
            "_No paired (fixture, adapter, field) measurements found — "
            "are both runs covering the same fixtures?_"
        )
        return "\n".join(lines) + "\n"

    verdict = (
        "✅ statistically significant improvement"
        if summary.significant_at_95 and summary.mean_delta > 0
        else "⚠️ statistically significant regression"
        if summary.significant_at_95 and summary.mean_delta < 0
        else "○ no significant change at 95% confidence"
    )

    lines.extend([
        "## Summary",
        "",
        f"- N (paired field samples): **{summary.n}**",
        f"- Mean Δ F1: **{summary.mean_delta:+.4f}**",
        f"- Std Δ F1: {summary.std_delta:.4f}",
        f"- t-statistic: {summary.t_statistic:.3f}",
        f"- two-sided p-value (Student-t, df={summary.n - 1}): {summary.p_value_two_sided:.4f}",
        f"- **Verdict: {verdict}**",
        "",
    ])

    if summary.n < 10:
        lines.append(
            f"⚠️ With only {summary.n} paired samples, statistical power "
            "is low. Treat the p-value as suggestive, not conclusive. "
            "Aim for N ≥ 30 paired samples (≥ 10 fixtures × ≥ 3 fields) "
            "for credible inference."
        )
        lines.append("")

    # Top 5 biggest improvements and regressions.
    sorted_deltas = sorted(deltas, key=lambda d: d.delta, reverse=True)
    if sorted_deltas:
        improvements = [d for d in sorted_deltas if d.delta > 0][:5]
        regressions = sorted(
            (d for d in sorted_deltas if d.delta < 0),
            key=lambda d: d.delta,
        )[:5]
        if improvements:
            lines.append("## Top 5 improvements")
            lines.append("| fixture | adapter | field | baseline | current | Δ |")
            lines.append("|" + "---|" * 6)
            for d in improvements:
                lines.append(
                    f"| {d.fixture} | {d.adapter} | {d.field_name} | "
                    f"{d.baseline_f1:.3f} | {d.current_f1:.3f} | "
                    f"{d.delta:+.3f} |"
                )
            lines.append("")
        if regressions:
            lines.append("## Top 5 regressions")
            lines.append("| fixture | adapter | field | baseline | current | Δ |")
            lines.append("|" + "---|" * 6)
            for d in regressions:
                lines.append(
                    f"| {d.fixture} | {d.adapter} | {d.field_name} | "
                    f"{d.baseline_f1:.3f} | {d.current_f1:.3f} | "
                    f"{d.delta:+.3f} |"
                )
            lines.append("")

    return "\n".join(lines) + "\n"
