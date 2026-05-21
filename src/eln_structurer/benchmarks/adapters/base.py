"""Adapter abstract base class for the benchmark harness."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from eln_structurer.benchmarks.canonical import CanonicalReaction


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


class Adapter(ABC):
    """Common interface every benchmarked tool implements."""

    name: str

    @abstractmethod
    async def is_available(self) -> bool:
        """Cheap, side-effect-free check that the tool can be invoked."""

    @abstractmethod
    async def extract(self, paragraph: str) -> CanonicalReaction:
        """Convert a paragraph to a CanonicalReaction; may raise AdapterError."""
