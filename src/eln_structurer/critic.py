"""Critic subagent.

After the primary agent finalizes a draft, a fresh critic LLM call reads
the original paragraph and the finalized JSON and answers a single
question: "does this draft faithfully represent the paragraph?"

The critic is intentionally fresh — no view of the primary agent's
transcript or tool calls — but it IS given the same schema and the same
chemistry heuristics the primary worked from. That parity lets it flag
chemistry omissions (Grignard without atmosphere, Suzuki without base,
yield outside [0, 105]) that the primary missed, while NOT inventing
false positives on optional fields it doesn't recognize.

Findings are parsed via a Pydantic model so a malformed critic response
becomes a clean parse error instead of a silent semantic drift.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Literal

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    TextBlock,
    query,
)
from pydantic import BaseModel, ValidationError

from eln_structurer.prompts import (
    EDGE_CASE_HEURISTICS,
    compressed_reaction_draft_schema,
)


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```\s*$", re.MULTILINE)


def _critic_system_prompt() -> str:
    return CRITIC_SYSTEM_PROMPT_TEMPLATE.format(
        heuristics=EDGE_CASE_HEURISTICS,
        schema_json=compressed_reaction_draft_schema(),
    )


# Pydantic-validated finding shape — the critic's output MUST match this.
class _CriticFindingModel(BaseModel):
    path: str
    severity: Literal["ERROR", "WARNING"]
    message: str


class _CriticResponseModel(BaseModel):
    findings: list[_CriticFindingModel]


CRITIC_SYSTEM_PROMPT_TEMPLATE = """\
You are a chemistry critic. Your only job is to compare a structured ORD
JSON draft against the original synthesis paragraph and report any
inconsistency or omission.

You DO NOT rewrite the draft. You DO NOT call tools. You DO NOT propose
new fields. You produce ONE JSON object of the exact form:

{{
  "findings": [
    {{
      "path": "<JSONPath-like string into the draft>",
      "severity": "ERROR" | "WARNING",
      "message": "<one-sentence description of what is wrong or missing>"
    }}
  ]
}}

Rules for findings:
- Only flag concrete mismatches between the paragraph and the draft.
- Findings must be actionable — the primary agent must be able to fix
  them by editing a specific field.
- If the draft is faithful, return {{"findings": []}}. An empty list IS
  the success signal; do not invent findings just to seem useful.
- The paragraph is authoritative. If the draft contradicts the paragraph,
  the draft is wrong.
- Focus areas, by failure-mode frequency:
    1. Numbers — yield%, mass, equivalents, temperature, duration. These
       are the most common transcription errors.
    2. Compound roles — REACTANT vs. REAGENT vs. SOLVENT vs. CATALYST.
       The agent often miscategorizes coupling partners or solvent-reagents.
    3. is_limiting — the smallest-moles REACTANT must carry the flag.
    4. atmosphere — Grignards, organolithiums, and Pd-catalysed couplings
       under inert atmospheres (nitrogen / argon) must say so.
    5. Workup completeness — extraction → wash → dry → concentrate.

DO NOT flag missing OPTIONAL fields. Below is the ReactionDraft JSON
schema; consult it before claiming a field is required.

# Chemistry heuristics the primary agent works from

{heuristics}

# ReactionDraft JSON Schema (compressed)

```json
{schema_json}
```

Output ONLY the JSON object with the findings list. No markdown, no
prose, no code fences. An empty findings list ({{"findings": []}}) is
the most common correct answer.
"""


@dataclass(frozen=True)
class CriticFinding:
    path: str
    severity: str
    message: str


@dataclass
class CriticReport:
    findings: list[CriticFinding] = field(default_factory=list)
    raw_text: str = ""
    parse_error: str | None = None

    @property
    def is_clean(self) -> bool:
        return self.parse_error is None and not any(
            f.severity == "ERROR" for f in self.findings
        )

    def as_repair_prompt(self) -> str:
        if not self.findings:
            return "CRITIC FOUND NO ISSUES."
        lines = [f"CRITIC FOUND {len(self.findings)} FINDING(S):"]
        for f in self.findings:
            lines.append(f"[{f.severity} {f.path}] {f.message}")
        lines.append(
            "\nFix every finding above, then call validate_reaction and "
            "finalize_reaction again. Do not introduce new fields the "
            "paragraph does not support."
        )
        return "\n".join(lines)


def _parse_findings(text: str) -> CriticReport:
    cleaned = _FENCE_RE.sub("", text).strip()
    if not cleaned:
        return CriticReport(raw_text=text, parse_error="critic returned empty output")
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        return CriticReport(raw_text=text, parse_error=f"non-JSON critic output: {exc}")
    # Strict Pydantic validation of the response shape — a malformed critic
    # response is treated as no-findings (parse_error set) instead of
    # silently dropping fields.
    try:
        parsed = _CriticResponseModel.model_validate(payload)
    except ValidationError as exc:
        return CriticReport(raw_text=text, parse_error=f"critic schema error: {exc}")
    findings = [
        CriticFinding(path=f.path, severity=f.severity, message=f.message)
        for f in parsed.findings
    ]
    return CriticReport(findings=findings, raw_text=cleaned)


def critic_available() -> bool:
    """The critic needs an Anthropic key. Skip it gracefully without one."""
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


async def run_critic(
    *,
    paragraph: str,
    draft_json: str,
    model: str,
) -> CriticReport:
    """Ask a fresh LLM whether the draft faithfully represents the paragraph."""
    user_prompt = (
        "Original paragraph:\n"
        "```\n"
        f"{paragraph}\n"
        "```\n\n"
        "Structured ORD draft (JSON):\n"
        "```json\n"
        f"{draft_json}\n"
        "```\n\n"
        "Now report findings as the JSON object specified in the system prompt."
    )
    options = ClaudeAgentOptions(
        model=model,
        system_prompt=_critic_system_prompt(),
        allowed_tools=[],
        max_turns=1,
    )
    text_parts: list[str] = []
    async for message in query(prompt=user_prompt, options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    text_parts.append(block.text)
    return _parse_findings("".join(text_parts))
