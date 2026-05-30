"""Yield estimation for ranking proposed protocols.

This is a Tier-3 stub interface. The eventual implementation will
fine-tune an RxnFP-style regression head on combined HTE + literature
data; for now we ship a deterministic similarity-weighted heuristic
that estimates yield from retrieved hits and reports an uncertainty
band derived from disagreement across the sample.

Critically — and per the predictor's risk table — the estimator
reports a **conservative lower bound** alongside the point estimate.
The ranker is expected to use the lower bound, not the point, so a
poorly-supported high prediction can't dominate the ranking.

When fewer than ``MIN_TRAINING_POINTS`` hits carry a yield, the
estimator refuses (returns ``None``) rather than guessing — explicitly
addressing the "yield-model overconfidence" risk.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

from eln_structurer.predict.retrieval import Hit


# Minimum number of yield-bearing nearest neighbours before we are
# willing to make any prediction at all. Below this we return None and
# the ranker treats yield as "unknown" rather than guessing.
MIN_TRAINING_POINTS = 3


@dataclass(frozen=True)
class YieldEstimate:
    point: float                    # weighted mean across hits, in %
    lower_95: float                 # conservative lower bound
    upper_95: float                 # upper bound
    n_support: int                  # how many hits contributed
    sources: list[str]              # distinct CorpusSource values seen


def estimate_yield(hits: list[Hit]) -> YieldEstimate | None:
    """Return a yield estimate or ``None`` when support is insufficient.

    The point estimate is the similarity-weighted mean across hits that
    carry a ``yield_percent``. The 95% band is symmetric two-sigma
    around the weighted mean of the yields, using the weighted standard
    deviation; when there is only a single hit we fall back to ±15 pp
    (a conservative blanket for the noise floor of HTE data).
    """
    yield_hits = [(h, h.record.yield_percent) for h in hits
                  if h.record.yield_percent is not None]
    if len(yield_hits) < MIN_TRAINING_POINTS:
        return None

    weights = [max(h.similarity, 0.1) for h, _ in yield_hits]
    yields = [y for _, y in yield_hits]
    sources = sorted({h.record.source.value for h, _ in yield_hits})

    total_w = sum(weights)
    mean = sum(w * y for w, y in zip(weights, yields, strict=False)) / total_w

    if len(yields) >= 2:
        variance = sum(
            w * (y - mean) ** 2 for w, y in zip(weights, yields, strict=False)
        ) / total_w
        std = variance ** 0.5
    else:                                # pragma: no cover - covered by MIN_TRAINING_POINTS
        std = 15.0

    # Two-sigma is roughly the 95% band assuming normality. Clamp to
    # plausible yield range.
    lower = max(0.0, mean - 2.0 * std)
    upper = min(105.0, mean + 2.0 * std)

    return YieldEstimate(
        point=round(mean, 1),
        lower_95=round(lower, 1),
        upper_95=round(upper, 1),
        n_support=len(yield_hits),
        sources=sources,
    )


def conservative_yield_score(estimate: YieldEstimate | None) -> float:
    """Translate an estimate (or None) into a 0..1 ranker score.

    Uses the lower 95% bound deliberately — see module docstring.
    ``None`` collapses to 0.5 (neutral) so unknown-yield candidates
    aren't suppressed; they're just not boosted.
    """
    if estimate is None:
        return 0.5
    return max(0.0, min(estimate.lower_95 / 100.0, 1.0))


# A median dataset value for ``statistics.median`` consumers — kept
# here as part of the public surface so the proposal layer can
# substitute a different aggregator without changing imports.
def median_yield(estimates: list[YieldEstimate]) -> float | None:
    values = [e.point for e in estimates]
    if not values:
        return None
    return statistics.median(values)


__all__ = [
    "MIN_TRAINING_POINTS",
    "YieldEstimate",
    "estimate_yield",
    "conservative_yield_score",
    "median_yield",
]
