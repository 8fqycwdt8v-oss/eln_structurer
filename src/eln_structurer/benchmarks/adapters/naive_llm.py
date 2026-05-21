"""Naive single-shot LLM baseline: prompt-only, no rule harness, no repair loop.

This is the most informative comparator for the eln_structurer harness — it
isolates the contribution of the validator-driven self-repair from the raw
LLM extraction quality. Uses ``claude_agent_sdk.query`` for the one-shot call
to avoid adding the bare anthropic SDK as a second dependency.
"""

from __future__ import annotations

import json
import re

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    TextBlock,
    query,
)

from eln_structurer.benchmarks.adapters.base import (
    Adapter,
    AdapterError,
    anthropic_key_available,
)
from eln_structurer.benchmarks.canonical import CanonicalReaction


_SCHEMA_HINT = """\
{
  "reactant_names": [string, ...],
  "reactant_smiles": [string, ...],
  "reagent_names": [string, ...],
  "solvent_names": [string, ...],
  "catalyst_names": [string, ...],
  "product_names": [string, ...],
  "product_smiles": [string, ...],
  "yield_percent": float or null,
  "temperature_celsius": float or null,
  "duration_minutes": float or null,
  "workup_verbs": [string, ...]
}
"""


_SYSTEM = f"""\
You extract structured information from a chemical synthesis paragraph.

Output ONLY a single JSON object matching this exact shape (no markdown, no
prose, no code fences):

{_SCHEMA_HINT}

Field semantics:
- reactant_names: the main starting materials.
- reagent_names: reagents that participate but aren't the main substrate (bases, oxidants).
- solvent_names: solvents.
- catalyst_names: catalysts (Pd complexes, acids used catalytically).
- product_names / product_smiles: the isolated product(s).
- yield_percent: numeric % yield if stated.
- temperature_celsius: the main reaction temperature (not workup); use 25 for rt.
- duration_minutes: total reaction time in minutes.
- workup_verbs: ordered, uppercase ORD workup types from this list:
  ADDITION, WASH, DRY_WITH_MATERIAL, EXTRACTION, FILTRATION, CONCENTRATION,
  PH_ADJUST, DISSOLUTION, TEMPERATURE, STIRRING, WAIT, DISTILLATION,
  FLASH_CHROMATOGRAPHY, CUSTOM.

If a field is not stated in the paragraph, use null (scalars) or an empty list.
Do not invent SMILES; if uncertain, leave reactant_smiles / product_smiles empty.
Do NOT call any tools. Emit the JSON object directly.
"""


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```\s*$", re.MULTILINE)


class NaiveLlmAdapter(Adapter):
    name = "naive_llm"

    def __init__(self, *, model: str = "claude-opus-4-7") -> None:
        self.model = model

    async def is_available(self) -> bool:
        return anthropic_key_available()

    async def extract(self, paragraph: str) -> CanonicalReaction:
        options = ClaudeAgentOptions(
            model=self.model,
            system_prompt=_SYSTEM,
            allowed_tools=[],  # no tools — single-shot extraction
            max_turns=1,
        )
        text_parts: list[str] = []
        async for message in query(prompt=paragraph, options=options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        text_parts.append(block.text)
        text = _FENCE_RE.sub("", "".join(text_parts)).strip()
        if not text:
            raise AdapterError("naive_llm returned empty output")
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise AdapterError(
                f"naive_llm produced non-JSON: {exc}; raw={text[:200]!r}"
            )
        return CanonicalReaction.from_dict(payload)
