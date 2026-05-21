"""Benchmark harness for eln_structurer.

The benchmark framework scores any chemistry-text-to-structure tool on a
canonical projection of an ORD reaction (a small set of fields shared by every
tool we want to compare: reactants, reagents, solvents, products, yield,
temperature, duration, workup verb sequence).

Layout:
  canonical.py — CanonicalReaction dataclass + projector from ORD JSON
  scoring.py   — precision/recall/F1 per field
  adapters/    — one Adapter subclass per tool
  runner.py    — for each fixture, run every available adapter, score it
"""
