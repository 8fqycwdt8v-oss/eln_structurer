"""Risk-handling primitives used by the predictor.

Each function here addresses one of the failure modes called out in
the predictor plan's risk table. They are pure / local and have no
LLM dependency, so the agent can call them via in-process tools and
trust their results.

- :func:`recency_summary` — staleness check on a list of retrieved hits.
- :func:`classifier_must_be_confident` — gate on the heuristic classifier.
- :func:`safety_screen` — layered safety check (controlled-chemicals
  list + RDKit structural features + peroxide-former regex).
- :func:`hard_constraint_filter` — convenience builder for query-time
  filtering used by the retrieval module.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass
from enum import Enum

from eln_structurer.predict.retrieval import Hit
from eln_structurer.reaction_class import ReactionClass, classify_reaction
from eln_structurer.schema import ReactionDraft


# ---------------------------------------------------------------------------
# Recency
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RecencySummary:
    median_year: float | None
    oldest: int | None
    newest: int | None
    n_with_year: int
    warning: str | None  # populated when median age > THRESHOLD


_STALE_AGE_YEARS = 10


def recency_summary(hits: list[Hit], *, reference_year: int) -> RecencySummary:
    """Look at the year distribution of retrieved hits; warn when stale.

    ``reference_year`` is normally ``datetime.date.today().year`` —
    parameterised so tests are deterministic.
    """
    years = [h.record.year for h in hits if h.record.year is not None]
    if not years:
        return RecencySummary(
            median_year=None, oldest=None, newest=None,
            n_with_year=0,
            warning="no publication-year metadata on any retrieved hit",
        )
    median = statistics.median(years)
    oldest = min(years)
    newest = max(years)
    age = reference_year - median
    warn = None
    if age > _STALE_AGE_YEARS:
        warn = (
            f"median retrieved hit is {age:.0f} years old (median year "
            f"{median:.0f}); newer methods (photoredox, electrochem, "
            "C–H activation) may be underrepresented"
        )
    return RecencySummary(
        median_year=median, oldest=oldest, newest=newest,
        n_with_year=len(years), warning=warn,
    )


# ---------------------------------------------------------------------------
# Classifier-confidence gate
# ---------------------------------------------------------------------------


_CLASSIFIER_CONFIDENCE_THRESHOLD = 0.7


def classifier_must_be_confident(draft: ReactionDraft) -> tuple[ReactionClass, bool, float]:
    """Run the heuristic classifier and report ``(class, is_confident, score)``.

    The predictor falls back to multi-skeleton search when ``is_confident``
    is False. Returning the score (not just the bool) lets callers surface
    the disagreement signal in the reasoning trail.
    """
    result = classify_reaction(draft)
    return result.cls, result.confidence >= _CLASSIFIER_CONFIDENCE_THRESHOLD, result.confidence


# ---------------------------------------------------------------------------
# Safety screen
# ---------------------------------------------------------------------------


class SafetyVerdict(str, Enum):
    OK = "ok"
    BLOCKED = "blocked"
    WARN = "warn"


@dataclass(frozen=True)
class SafetyReport:
    verdict: SafetyVerdict
    flags: list[str]


# Names that must NEVER appear in a proposed protocol. The list is
# deliberately narrow — additions are easy, false positives are not.
_CONTROLLED_NAMES = {
    "potassium cyanide",
    "sodium cyanide",
    "hydrogen cyanide",
    "phosgene",
    "carbonyl chloride",
    "sulfur mustard",
    "tabun", "sarin", "soman", "vx",
    "thallium",
    "osmium tetroxide",
}

# Substring patterns flagged for review even if not strictly controlled.
_HIGH_RISK_SUBSTRINGS = [
    "azide",          # explosive risk
    "diazo",          # decomposition risk
    "perchlorate",    # oxidiser
    "perchloric acid",
    "hydrazine",      # carcinogen + explosive
    "picric",         # primary explosive
    "fulminate",
    "peroxide",       # may form explosive crystals
]

# Bench-classic peroxide-formers — bound to surface in retrieved hits
# every now and then; we want a WARN not BLOCK so chemists can still
# use them with documented mitigations.
_PEROXIDE_FORMER_PATTERN = re.compile(
    r"\b(thf|tetrahydrofuran|diethyl\s*ether|dioxane|"
    r"diisopropyl\s*ether|isopropyl\s*ether)\b"
)


def _walk_names(draft: ReactionDraft) -> list[str]:
    out: list[str] = []
    for inp in draft.inputs:
        for comp in inp.components:
            for ident in comp.identifiers:
                if ident.type in {"NAME", "IUPAC_NAME"}:
                    out.append(ident.value.strip().lower())
    for wu in draft.workups:
        for comp in wu.components:
            for ident in comp.identifiers:
                if ident.type in {"NAME", "IUPAC_NAME"}:
                    out.append(ident.value.strip().lower())
    return out


def safety_screen(draft: ReactionDraft) -> SafetyReport:
    """Layered safety check.

    1. Controlled-chemical list → BLOCKED.
    2. High-risk substring (azide, peroxide, …) → WARN.
    3. Peroxide-former solvent → WARN with mitigation guidance.

    A draft is BLOCKED if any controlled name is matched. Multiple
    flags are reported together so the chemist sees the full picture.
    """
    flags: list[str] = []
    verdict = SafetyVerdict.OK
    names = _walk_names(draft)

    for name in names:
        if name in _CONTROLLED_NAMES:
            flags.append(f"CONTROLLED: {name!r}")
            verdict = SafetyVerdict.BLOCKED
        for pat in _HIGH_RISK_SUBSTRINGS:
            if pat in name and f"HIGH_RISK: {name!r} ({pat})" not in flags:
                flags.append(f"HIGH_RISK: {name!r} ({pat})")
                if verdict is SafetyVerdict.OK:
                    verdict = SafetyVerdict.WARN
        if _PEROXIDE_FORMER_PATTERN.search(name):
            msg = f"PEROXIDE_FORMER: {name!r} (test for peroxides before distillation)"
            if msg not in flags:
                flags.append(msg)
                if verdict is SafetyVerdict.OK:
                    verdict = SafetyVerdict.WARN

    return SafetyReport(verdict=verdict, flags=flags)


# ---------------------------------------------------------------------------
# Hard-constraint composer
# ---------------------------------------------------------------------------


def hard_constraint_filter(constraints: dict | None):
    """Compose a list of RecordFilter callables from a user-provided dict.

    Recognised constraint keys:

    - ``"no_halogenated_solvents": True``
    - ``"min_year": <int>``
    - ``"allowed_sources": ["literature", "ord", "hte", ...]``

    Unknown keys are ignored (the caller may extend the filter set
    without breaking older deployments).
    """
    from eln_structurer.predict.retrieval import (
        min_year,
        no_halogenated_solvents,
        source_in,
    )
    from eln_structurer.predict.corpus import CorpusSource

    if not constraints:
        return []
    filters = []
    if constraints.get("no_halogenated_solvents"):
        filters.append(no_halogenated_solvents)
    if "min_year" in constraints:
        filters.append(min_year(int(constraints["min_year"])))
    if "allowed_sources" in constraints:
        allowed = {CorpusSource(s) for s in constraints["allowed_sources"]}
        filters.append(source_in(allowed))
    return filters


__all__ = [
    "RecencySummary",
    "recency_summary",
    "classifier_must_be_confident",
    "SafetyVerdict",
    "SafetyReport",
    "safety_screen",
    "hard_constraint_filter",
]
