"""Synthetic HTE-flavoured corpus used in tests and as a default seed.

A real production deployment would replace this with an actual
ingestion of the Doyle (Princeton) Buchwald HTE dataset, the Cernak
photoredox HTE corpus, or an internal industrial ELN extract. The
synthetic records here exist so the predictor's other modules have
something to retrieve against in unit tests AND so a fresh checkout
of the project ships with a non-empty corpus by default.

Every synthetic record carries ``source=CorpusSource.HTE`` so the
multi-source vote correctly treats them as a distinct source from
literature.
"""

from __future__ import annotations

from eln_structurer.predict.corpus import CorpusSource, LocalCorpus, ReactionRecord


def _suzuki_records() -> list[ReactionRecord]:
    """A handful of Suzuki couplings under different conditions."""
    base_smiles = "BrC1=CC=CC=C1.OB(O)c1ccccc1>>C1=CC=CC=C1C1=CC=CC=C1"
    return [
        ReactionRecord(
            reaction_smiles=base_smiles,
            source=CorpusSource.HTE, source_id="hte:suzuki:1",
            year=2018,
            conditions={
                "catalysts": ["Pd(PPh3)4"],
                "reagents": ["K2CO3"],
                "solvents": ["1,4-dioxane", "water"],
                "temperature_celsius": 90.0,
                "duration_minutes": 960.0,
                "atmosphere": "nitrogen",
            },
            yield_percent=89.0,
        ),
        ReactionRecord(
            reaction_smiles=base_smiles,
            source=CorpusSource.LITERATURE, source_id="lit:suzuki:tol",
            year=2016,
            conditions={
                "catalysts": ["Pd(PPh3)4"],
                "reagents": ["K2CO3"],
                "solvents": ["toluene"],
                "temperature_celsius": 100.0,
                "duration_minutes": 720.0,
                "atmosphere": "nitrogen",
            },
            yield_percent=82.0,
        ),
        ReactionRecord(
            reaction_smiles=base_smiles,
            source=CorpusSource.ORD, source_id="ord:suzuki:1",
            year=2021,
            conditions={
                "catalysts": ["Pd(OAc)2"],
                "reagents": ["XPhos", "Cs2CO3"],
                "solvents": ["1,4-dioxane"],
                "temperature_celsius": 100.0,
                "duration_minutes": 1080.0,
                "atmosphere": "argon",
            },
            yield_percent=92.0,
        ),
    ]


def _amide_records() -> list[ReactionRecord]:
    smi = "O=C(O)c1ccccc1.Nc1ccccc1>>O=C(Nc1ccccc1)c1ccccc1"
    return [
        ReactionRecord(
            reaction_smiles=smi,
            source=CorpusSource.LITERATURE, source_id="lit:amide:hatu",
            year=2019,
            conditions={
                "reagents": ["HATU", "DIPEA"],
                "solvents": ["DMF"],
                "temperature_celsius": 25.0,
                "duration_minutes": 240.0,
            },
            yield_percent=92.0,
        ),
        ReactionRecord(
            reaction_smiles=smi,
            source=CorpusSource.INDUSTRIAL, source_id="industrial:amide:1",
            year=2022,
            conditions={
                "reagents": ["EDC", "HOBt", "DIPEA"],
                "solvents": ["DCM"],
                "temperature_celsius": 25.0,
                "duration_minutes": 720.0,
            },
            yield_percent=85.0,
        ),
    ]


def default_seed_corpus() -> LocalCorpus:
    """Return a small fully-populated LocalCorpus for tests / smoke runs."""
    corpus = LocalCorpus()
    corpus.add_many(_suzuki_records())
    corpus.add_many(_amide_records())
    return corpus


__all__ = ["default_seed_corpus"]
