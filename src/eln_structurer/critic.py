"""Critic subagent.

After the primary agent finalizes a draft, a fresh critic LLM call reads
the original paragraph and the finalized JSON and answers a single
question: "does this draft faithfully represent the paragraph?"

The critic is intentionally *fresh* — it has no view of the primary
agent's transcript, no tools, no schema dump. Its only output is a JSON
list of findings, each with a path into the draft and an explanation.
That isolation lets it catch systematic mistakes the primary made because
its earlier reasoning was framed wrong.

The critic loop is bounded: at most one critic pass per ``extract()``
call. If the critic surfaces findings the harness can act on, they get
fed back into the agent for a single revision round. Past round 1 the
critic is skipped to avoid pathological loops.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    TextBlock,
    query,
)


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```\s*$", re.MULTILINE)


CRITIC_SYSTEM_PROMPT = """\
You are a chemistry critic. Your only job is to compare a structured ORD
JSON draft against the original synthesis paragraph and report any
inconsistency or omission.

You DO NOT rewrite the draft. You DO NOT call tools. You DO NOT propose
new fields. You produce ONE JSON object of the exact form:

{
  "findings": [
    {
      "path": "<JSONPath-like string into the draft>",
      "severity": "ERROR" | "WARNING",
      "message": "<one-sentence description of what is wrong or missing>"
    },
    ...
  ]
}

Rules for findings:
- Only flag concrete mismatches between the paragraph and the draft.
- A "finding" should be actionable — the primary agent must be able to fix
  it by editing a specific field. Avoid vague critiques like "could be
  more complete".
- If the draft is faithful, return {"findings": []}. An empty list is
  the success signal.
- The paragraph is authoritative. If the draft contradicts the paragraph,
  the draft is wrong.
- Pay particular attention to numbers (yield %, masses, equivalents,
  temperatures, durations) — transcription errors are the most common
  failure mode.
- Pay attention to compound roles: the substrate vs. reagent vs. solvent
  distinction is often wrong.

Output ONLY the JSON object. No markdown, no prose, no code fences.
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
    if not isinstance(payload, dict) or "findings" not in payload:
        return CriticReport(
            raw_text=text,
            parse_error="critic output missing top-level 'findings' key",
        )
    findings_raw = payload["findings"] or []
    if not isinstance(findings_raw, list):
        return CriticReport(
            raw_text=text, parse_error="'findings' is not a list"
        )
    findings: list[CriticFinding] = []
    for f in findings_raw:
        if not isinstance(f, dict):
            continue
        findings.append(
            CriticFinding(
                path=str(f.get("path", "")),
                severity=str(f.get("severity", "WARNING")).upper(),
                message=str(f.get("message", "")),
            )
        )
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
        system_prompt=CRITIC_SYSTEM_PROMPT,
        allowed_tools=[],  # critic gets NO tools
        max_turns=1,
    )
    text_parts: list[str] = []
    async for message in query(prompt=user_prompt, options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    text_parts.append(block.text)
    return _parse_findings("".join(text_parts))
