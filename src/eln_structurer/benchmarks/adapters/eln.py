"""Adapter for the eln_structurer agent itself."""

from __future__ import annotations

from eln_structurer.agent import DEFAULT_MODEL, extract
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
        # Last-call telemetry, exposed for the benchmark runner to harvest.
        self._last_iterations: int = 0
        self._last_critic_ran: bool = False
        self._last_revision_triggered: bool = False
        self._last_rule_history: dict[str, int] = {}

    async def is_available(self) -> bool:
        return anthropic_key_available()

    async def extract(self, paragraph: str) -> CanonicalReaction:
        result = await extract(
            paragraph, model=self.model, max_iters=self.max_iters, debug=False
        )
        if not result.success:
            raise AdapterError(
                f"eln_structurer failed to converge: {result.failure_summary}"
            )
        # Stash observability fields where run_adapter can pick them up.
        self._last_iterations = result.iterations
        self._last_critic_ran = result.critic_ran
        self._last_revision_triggered = result.revision_triggered
        self._last_rule_history = dict(result.rule_history)
        return canonicalize_ord_json(result.json_text)
