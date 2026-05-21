"""Adapter abstract base class for the benchmark harness."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass

from eln_structurer.benchmarks.canonical import CanonicalReaction


def anthropic_key_available() -> bool:
    """True iff a non-empty ANTHROPIC_API_KEY is exported."""
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


class AdapterError(RuntimeError):
    """Raised when an adapter could run but the prediction failed."""


class AdapterUnavailable(RuntimeError):
    """Raised when the underlying tool is not installed in this environment."""


@dataclass
class AdapterResult:
    adapter_name: str
    success: bool
    prediction: CanonicalReaction | None
    error: str | None
    elapsed_seconds: float
    # Optional observability fields populated by adapters that drive a
    # tool-using agent loop (eln_structurer). Defaults make
    # naive_llm-style single-shot adapters work unchanged.
    iterations: int = 0
    critic_ran: bool = False
    revision_triggered: bool = False
    rule_history: dict[str, int] | None = None
    cost_usd: float = 0.0
    api_duration_ms: int = 0


class Adapter(ABC):
    """Common interface every benchmarked tool implements."""

    name: str

    @abstractmethod
    async def is_available(self) -> bool:
        """Cheap, side-effect-free check that the tool can be invoked."""

    @abstractmethod
    async def extract(self, paragraph: str) -> CanonicalReaction:
        """Convert a paragraph to a CanonicalReaction; may raise AdapterError."""
