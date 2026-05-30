"""Reaction-corpus types and a small in-memory implementation.

A ``ReactionRecord`` is the unit of retrieval — it carries the reaction
SMILES (canonical, atom-mapping-free for retrieval), the structured
conditions extracted from the source, the reported yield if any, the
source identifier (e.g. ``"ord:reaction_id_..."``, ``"hte:doyle_2018"``),
and the publication year for staleness checks.

``LocalCorpus`` is an in-memory store with O(1) exact lookup by canonical
SMILES and a precomputed list of fingerprints for KNN retrieval. The
search isn't optimised yet — Tier 1 ships with a brute-force Tanimoto
sweep that scales to a few thousand records; we'll add Faiss in Tier 2
when the corpus grows.

Stays free of LLM / network access; everything here is local Python.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class CorpusSource(str, Enum):
    """Provenance bucket for a ReactionRecord.

    Distinct sources travel through retrieval independently so the
    ranker can see "the only signal supporting THIS slot is the HTE
    bucket; literature disagrees". Important for the multi-source vote.
    """
    LITERATURE = "literature"   # USPTO patents, journals, Reaxys-mined
    ORD = "ord"                 # Open Reaction Database curated
    HTE = "hte"                 # High-throughput experimentation datasets
    INDUSTRIAL = "industrial"   # ELN extracts (AstraZeneca-750, etc.)
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ReactionRecord:
    """One record in the reaction corpus.

    Frozen — records are read-only after ingestion. Hashing is by
    canonical SMILES so duplicate ingestion is a no-op.
    """
    reaction_smiles: str
    source: CorpusSource
    source_id: str               # e.g. "ord:reaction_abc123", "hte:doyle_2018:reaction_4203"
    year: int | None = None      # publication / experiment year for staleness
    conditions: dict[str, Any] = field(default_factory=dict)   # solvent / catalyst / temp / etc.
    procedure_text: str | None = None
    yield_percent: float | None = None
    notes: str | None = None

    def __hash__(self) -> int:
        return hash((self.reaction_smiles, self.source_id))

    @property
    def signature(self) -> str:
        """SHA-256 of the canonical reaction SMILES, used for exact match."""
        return hashlib.sha256(self.reaction_smiles.encode("utf-8")).hexdigest()


class LocalCorpus:
    """In-memory reaction store.

    Records are keyed by ``(reaction_smiles, source_id)`` so the same
    SMILES published by multiple labs all sit in the corpus and are
    retrieved together — the multi-source vote in the predictor expects
    this. Use :meth:`add` to ingest, :meth:`retrieve_exact` for the
    same-reaction channel, and :meth:`__iter__` to drive K-NN retrieval
    in the sibling ``retrieval`` module.
    """

    def __init__(self) -> None:
        self._records: dict[tuple[str, str], ReactionRecord] = {}
        self._by_signature: dict[str, list[ReactionRecord]] = {}

    # --- ingestion ---------------------------------------------------------

    def add(self, record: ReactionRecord) -> None:
        key = (record.reaction_smiles, record.source_id)
        if key in self._records:
            return  # idempotent: re-adding the same record is silent
        self._records[key] = record
        self._by_signature.setdefault(record.signature, []).append(record)

    def add_many(self, records: list[ReactionRecord]) -> None:
        for r in records:
            self.add(r)

    # --- exact-match channel -----------------------------------------------

    def retrieve_exact(self, reaction_smiles: str) -> list[ReactionRecord]:
        """Return every record whose canonical SMILES matches exactly.

        This is the "same reaction known" channel. Multiple records are
        legitimate — different labs running the same coupling under
        different conditions — and the ranker picks among them.
        """
        sig = hashlib.sha256(reaction_smiles.encode("utf-8")).hexdigest()
        return list(self._by_signature.get(sig, []))

    # --- traversal for K-NN -------------------------------------------------

    def __iter__(self) -> Iterator[ReactionRecord]:
        return iter(self._records.values())

    def __len__(self) -> int:
        return len(self._records)

    # --- introspection ------------------------------------------------------

    def by_source(self, source: CorpusSource) -> list[ReactionRecord]:
        return [r for r in self._records.values() if r.source is source]

    def recency_stats(self) -> dict[str, Any]:
        """Year distribution across the corpus — used by the staleness
        check in ``risks.recency_warning``."""
        years = [r.year for r in self._records.values() if r.year is not None]
        if not years:
            return {"count": 0, "with_year": 0, "min": None, "max": None, "median": None}
        years_sorted = sorted(years)
        n = len(years_sorted)
        median = years_sorted[n // 2] if n % 2 else (years_sorted[n // 2 - 1] + years_sorted[n // 2]) / 2
        return {
            "count": len(self._records),
            "with_year": n,
            "min": years_sorted[0],
            "max": years_sorted[-1],
            "median": median,
        }
