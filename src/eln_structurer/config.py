"""Centralised tunables for the extractor + benchmark.

Every magic number that used to live as a module-level constant (sometimes
under different names in different files) now lives here. Existing
callsites read the defaults; tests and CLI can override individual fields
via dataclasses.replace().
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExtractorConfig:
    """Knobs for ``agent.extract`` and the repair loop."""
    # Hard cap on paragraph length — beyond this we short-circuit before
    # spending any LLM tokens. ~4k tokens at the conventional 4-chars/token.
    max_paragraph_chars: int = 16_000
    # Repair-iteration soft budget surfaced to the agent via the
    # validate_reaction tool result. The SDK enforces ``max_turns =
    # max_iters * 3`` separately; this is the per-iteration human-readable cap.
    default_iter_budget: int = 5
    # Number of consecutive failures of the same rule_id that triggers
    # the divergence escalation message.
    divergence_threshold: int = 3


@dataclass(frozen=True)
class CriticConfig:
    """Knobs for the critic-reviser subagent."""
    enabled_by_default: bool = True


@dataclass(frozen=True)
class BenchmarkConfig:
    """Knobs for ``benchmarks/scoring.py`` and ``benchmarks/runner.py``."""
    yield_tolerance: float = 0.05
    temperature_tolerance: float = 0.05
    duration_tolerance: float = 0.10
    # Bound on the in-memory CaseRun accumulator before run_benchmark
    # refuses to add more (avoids OOM on absurd fixture sets).
    max_runs_accumulator: int = 10_000


# Module-level defaults — read directly throughout the codebase.
DEFAULT_EXTRACTOR_CONFIG = ExtractorConfig()
DEFAULT_CRITIC_CONFIG = CriticConfig()
DEFAULT_BENCHMARK_CONFIG = BenchmarkConfig()
