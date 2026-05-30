"""K-NN retrieval over a :class:`LocalCorpus`.

Tier 1 implementation is a brute-force Tanimoto sweep — fine up to a
few thousand records, which covers a hand-curated ORD + HTE seed
corpus. Tier 2 will swap in Faiss for scale; the API on this module
won't change.

Hard constraints (e.g. ``no_halogenated_solvents``) are applied at
QUERY time, not RANK time. The rationale is in the plan's risk table:
filtering at rank time means an entire top-K could be wasted slots if
every candidate violates a hard constraint. Filtering at query time
keeps top-K useful.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from eln_structurer.predict.corpus import LocalCorpus, ReactionRecord
from eln_structurer.predict.fingerprint import reaction_fingerprint, tanimoto


# Solvent → halogen-class lookup used by the built-in
# ``no_halogenated_solvents`` filter. Kept narrow and well-known; the
# filter is conservative — if a solvent isn't in this list we DON'T
# guess, we let the record through.
_HALOGENATED_SOLVENTS = {
    "dichloromethane", "dcm", "ch2cl2",
    "chloroform", "chcl3",
    "carbon tetrachloride", "ccl4",
    "1,2-dichloroethane", "dce",
    "trichloroethylene",
    "bromobenzene",
    "fluorobenzene",
}


@dataclass(frozen=True)
class Hit:
    """One retrieval result with its similarity score."""
    record: ReactionRecord
    similarity: float


# A filter is a callable that returns True to KEEP the record.
RecordFilter = Callable[[ReactionRecord], bool]


# --- built-in filters --------------------------------------------------------


def no_halogenated_solvents(record: ReactionRecord) -> bool:
    """Filter out records whose solvent is a halogenated one."""
    solvents = record.conditions.get("solvents") or []
    for s in solvents:
        if str(s).strip().lower() in _HALOGENATED_SOLVENTS:
            return False
    return True


def min_year(year_threshold: int) -> RecordFilter:
    """Filter out records published before ``year_threshold``."""
    def _f(r: ReactionRecord) -> bool:
        return r.year is None or r.year >= year_threshold
    return _f


def source_in(allowed: set) -> RecordFilter:
    """Allow only records whose source enum is in ``allowed``."""
    def _f(r: ReactionRecord) -> bool:
        return r.source in allowed
    return _f


# --- retrieval entry points --------------------------------------------------


def retrieve_exact(corpus: LocalCorpus, reaction_smiles: str) -> list[Hit]:
    """Channel C — same-reaction exact match by canonical SMILES hash.

    Returns every matching record with similarity=1.0. The retrieval is
    not silent when multiple records exist for the same SMILES; the
    caller (composition layer) is expected to vote among them.
    """
    records = corpus.retrieve_exact(reaction_smiles)
    return [Hit(record=r, similarity=1.0) for r in records]


def retrieve_knn(
    corpus: LocalCorpus,
    reaction_smiles: str,
    *,
    k: int = 5,
    filters: list[RecordFilter] | None = None,
    min_similarity: float = 0.0,
) -> list[Hit]:
    """Channel D — K nearest neighbours by reaction fingerprint Tanimoto.

    Hard filters apply BEFORE scoring so the top-K is guaranteed
    constraint-compliant. ``min_similarity`` is a final floor (default 0
    — no floor); useful when a caller wants to gate on "no match unless
    Tanimoto ≥ 0.4".
    """
    target_fp = reaction_fingerprint(reaction_smiles)
    if not target_fp:
        return []

    scored: list[Hit] = []
    for record in corpus:
        if filters and not all(f(record) for f in filters):
            continue
        sim = tanimoto(target_fp, reaction_fingerprint(record.reaction_smiles))
        if sim < min_similarity:
            continue
        scored.append(Hit(record=record, similarity=sim))

    scored.sort(key=lambda h: h.similarity, reverse=True)
    return scored[:k]
