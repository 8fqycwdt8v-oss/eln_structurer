"""Adapter for the eln_structurer agent itself."""

from __future__ import annotations

from eln_structurer.agent import DEFAULT_MODEL, ExtractResult, extract
from eln_structurer.benchmarks.adapters.base import (
    Adapter,
    AdapterError,
    anthropic_key_available,
)
from eln_structurer.benchmarks.canonical import (
    CanonicalReaction,
    canonicalize_ord_json,
)


class ElnStructurerAdapter(Adapter):
    name = "eln_structurer"

    def __init__(self, *, model: str = DEFAULT_MODEL, max_iters: int = 5) -> None:
        self.model = model
        self.max_iters = max_iters
        # The full ExtractResult from the last extract() call. The
        # benchmark runner harvests iterations / cost / critic_ran etc.
        # from this single attribute instead of pulling them out of
        # six separate _last_* sidechannel fields the previous design
        # used.
        self._last_result: ExtractResult | None = None

    async def is_available(self) -> bool:
        return anthropic_key_available()

    async def extract(self, paragraph: str) -> CanonicalReaction:
        result = await extract(
            paragraph, model=self.model, max_iters=self.max_iters, debug=False
        )
        self._last_result = result
        if not result.success:
            raise AdapterError(
                f"eln_structurer failed to converge: {result.failure_summary}"
            )
        return canonicalize_ord_json(result.json_text)
