"""Critic subagent — independent review of a finalized draft against
the source paragraph. Public surface is just ``run_critic`` and the
``CriticReport`` / ``CriticFinding`` dataclasses."""

from __future__ import annotations

from eln_structurer.critic.models import CriticFinding, CriticReport
from eln_structurer.critic.runner import (
    _parse_findings,  # exposed for tests
    critic_available,
    run_critic,
)

__all__ = [
    "CriticReport",
    "CriticFinding",
    "critic_available",
    "run_critic",
    "_parse_findings",
]
