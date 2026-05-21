"""Adapter registry. Each entry produces a fresh Adapter instance on demand."""

from __future__ import annotations

from typing import Callable

from eln_structurer.benchmarks.adapters.base import (
    Adapter,
    AdapterError,
    AdapterResult,
    AdapterUnavailable,
)
from eln_structurer.benchmarks.adapters.eln import ElnStructurerAdapter
from eln_structurer.benchmarks.adapters.naive_llm import NaiveLlmAdapter
from eln_structurer.benchmarks.adapters.openchemie import OpenChemIEAdapter
from eln_structurer.benchmarks.adapters.paragraph2actions import (
    Paragraph2ActionsAdapter,
)


ADAPTERS: dict[str, Callable[[], Adapter]] = {
    "eln_structurer": ElnStructurerAdapter,
    "naive_llm": NaiveLlmAdapter,
    "paragraph2actions": Paragraph2ActionsAdapter,
    "openchemie": OpenChemIEAdapter,
}


__all__ = [
    "ADAPTERS",
    "Adapter",
    "AdapterError",
    "AdapterResult",
    "AdapterUnavailable",
    "ElnStructurerAdapter",
    "NaiveLlmAdapter",
    "Paragraph2ActionsAdapter",
    "OpenChemIEAdapter",
]
