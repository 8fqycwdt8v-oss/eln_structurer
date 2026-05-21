"""Adapter for the eln_structurer agent itself."""

from __future__ import annotations

from eln_structurer.agent import DEFAULT_MODEL, extract
from eln_structurer.benchmarks.adapters.base import Adapter, AdapterError
from eln_structurer.benchmarks.canonical import (
    CanonicalReaction,
    canonicalize_ord_json,
)


class ElnStructurerAdapter(Adapter):
    name = "eln_structurer"

    def __init__(self, *, model: str = DEFAULT_MODEL, max_iters: int = 5) -> None:
        self.model = model
        self.max_iters = max_iters

    async def is_available(self) -> bool:
        # Available iff Anthropic credentials are set (the agent checks this).
        import os
        return bool(os.environ.get("ANTHROPIC_API_KEY"))

    async def extract(self, paragraph: str) -> CanonicalReaction:
        result = await extract(
            paragraph, model=self.model, max_iters=self.max_iters, debug=False
        )
        if not result.success:
            raise AdapterError(
                f"eln_structurer failed to converge: {result.validation_summary}"
            )
        return canonicalize_ord_json(result.json_text)
