"""verify_quote — pre-flight check that a string appears in the source paragraph.

The agent uses this BEFORE setting Amount.source_quote (or
ProductMeasurement.source_quote) to confirm the substring it's about to
commit actually exists in the source text. Saves a wasted validate→fix
cycle when a quote is off by a character.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from claude_agent_sdk import tool


_WHITESPACE_RE = re.compile(r"\s+")


def _normalise(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text.lower()).strip()


@dataclass(frozen=True)
class QuoteCheck:
    ok: bool
    quote: str
    nearest_match: str | None
    message: str


def verify_quote_against(quote: str, paragraph: str) -> QuoteCheck:
    q = _normalise(quote)
    h = _normalise(paragraph)
    if not q:
        return QuoteCheck(False, quote, None, "empty quote")
    if q in h:
        return QuoteCheck(True, quote, None, "exact match found")
    # Try a soft nearest-match diagnosis: find the longest left-anchored
    # prefix of the quote that DOES occur in the paragraph. Helps the
    # agent realise it overshot by a few characters.
    longest_ok = ""
    for end in range(1, len(q) + 1):
        if q[:end] in h:
            longest_ok = q[:end]
        else:
            break
    if longest_ok and len(longest_ok) >= 6:
        return QuoteCheck(
            False, quote, longest_ok,
            f"quote not in paragraph; longest matching prefix: {longest_ok!r}",
        )
    return QuoteCheck(False, quote, None, "quote not in paragraph")


@tool(
    "verify_quote",
    (
        "Check that a candidate source_quote string appears in the source "
        "paragraph after normalisation (lowercase + collapsed whitespace). "
        "Returns ok=True/False and, when False, the longest matching prefix "
        "of the quote that DOES occur in the paragraph (helps diagnose "
        "overshoots). Call BEFORE setting Amount.source_quote / "
        "ProductMeasurement.source_quote so you don't waste a validate cycle."
    ),
    {"quote": str, "paragraph": str},
)
async def verify_quote(args: dict[str, Any]) -> dict[str, Any]:
    quote = args.get("quote", "")
    paragraph = args.get("paragraph", "")
    if not isinstance(quote, str) or not isinstance(paragraph, str):
        return {
            "content": [
                {"type": "text", "text": "ERROR: quote and paragraph must be strings."}
            ],
            "isError": True,
        }
    result = verify_quote_against(quote, paragraph)
    if result.ok:
        return {
            "content": [{"type": "text", "text": "VALID: quote found in paragraph."}]
        }
    return {
        "content": [{"type": "text", "text": f"INVALID: {result.message}"}],
        "isError": True,
    }
