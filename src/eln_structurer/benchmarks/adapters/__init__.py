"""Adapter registry.

The default registry ships with the four built-in adapters
(``eln_structurer``, ``naive_llm``, ``paragraph2actions``, ``openchemie``).
Third parties register additional adapters at import time via
``REGISTRY.register(name, factory)``. The benchmark CLI and runner go
through the registry; they never reach into the dict directly.
"""

from __future__ import annotations

from typing import Callable

from eln_structurer.benchmarks.adapters.base import (
    Adapter,
    AdapterError,
    AdapterResult,
    AdapterUnavailable,
)
from eln_structurer.benchmarks.adapters.eln import ElnStructurerAdapter
from eln_structurer.benchmarks.adapters.experimental.openchemie import OpenChemIEAdapter
from eln_structurer.benchmarks.adapters.experimental.paragraph2actions import (
    Paragraph2ActionsAdapter,
)
from eln_structurer.benchmarks.adapters.naive_llm import NaiveLlmAdapter


AdapterFactory = Callable[[], Adapter]


class AdapterRegistry:
    """Name → factory mapping with explicit register / get semantics.

    Replaces a bare ``dict`` so third parties can plug in their own adapters
    without monkey-patching. Duplicate registrations raise unless
    ``overwrite=True`` is passed.
    """

    def __init__(self) -> None:
        self._adapters: dict[str, AdapterFactory] = {}

    def register(
        self, name: str, factory: AdapterFactory, *, overwrite: bool = False
    ) -> None:
        if not name:
            raise ValueError("adapter name must be non-empty")
        if name in self._adapters and not overwrite:
            raise ValueError(
                f"adapter {name!r} already registered; pass overwrite=True to replace"
            )
        self._adapters[name] = factory

    def get(self, name: str) -> AdapterFactory | None:
        return self._adapters.get(name)

    def names(self) -> list[str]:
        return sorted(self._adapters.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._adapters

    def __iter__(self):
        return iter(self._adapters.keys())

    def __len__(self) -> int:
        return len(self._adapters)


REGISTRY = AdapterRegistry()
REGISTRY.register("eln_structurer", ElnStructurerAdapter)
REGISTRY.register("naive_llm", NaiveLlmAdapter)
REGISTRY.register("paragraph2actions", Paragraph2ActionsAdapter)
REGISTRY.register("openchemie", OpenChemIEAdapter)


# Back-compat dict view of the registry — read-only by convention.
# Existing code (and the bench CLI) used to look up adapters via the
# ``ADAPTERS`` dict; keep the alias so callers don't need to change.
ADAPTERS: dict[str, AdapterFactory] = {n: REGISTRY.get(n) for n in REGISTRY.names()}  # type: ignore[misc]


__all__ = [
    "ADAPTERS",
    "REGISTRY",
    "AdapterRegistry",
    "Adapter",
    "AdapterError",
    "AdapterFactory",
    "AdapterResult",
    "AdapterUnavailable",
    "ElnStructurerAdapter",
    "NaiveLlmAdapter",
    "Paragraph2ActionsAdapter",
    "OpenChemIEAdapter",
]
