"""Pre-extraction paragraph normalization.

Cleans up common Unicode and formatting quirks before the agent sees the
text, so the LLM doesn't waste tokens reasoning about typography. Returns
both the normalized text (fed to the agent) and the original (preserved
for ``source_paragraph``).

Intentionally light — anything that risks changing chemistry meaning is
left alone. We only normalize encoding-level noise.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


# Unicode → ASCII substitutions that never change chemistry meaning.
# Note: we deliberately keep °C as ° + C because RDKit / downstream code
# may grep for "°C"; the underlying character is preserved in NFC form.
_CHAR_FIXUPS = {
    "µ": "u",      # micro sign (µ) → u, so "µL" reads as "uL"
    "μ": "u",      # Greek mu (μ) → u
    "×": "x",      # multiplication sign × → x
    "·": ".",      # middle dot · → .
    "–": "-",      # en dash – → -
    "—": "-",      # em dash — → -
    "−": "-",      # minus sign − → -
    "‘": "'",      # left single quote
    "’": "'",      # right single quote
    "“": '"',      # left double quote
    "”": '"',      # right double quote
    " ": " ",      # non-breaking space → regular space
    " ": " ",      # narrow non-breaking space
    " ": " ",      # thin space
}


# Common chemistry abbreviations the LLM understands but better-when-spelled.
# Single-word replacements only; we don't try to expand phrases or context.
# Substitution happens at *word boundaries* via regex.
_ABBREVIATION_EXPANSIONS = {
    "rt": "room temperature",
    "r.t.": "room temperature",
    "rt.": "room temperature",
    "o.n.": "overnight",
    "o/n": "overnight",
    "aq.": "aqueous",
    "sat.": "saturated",
    "satd.": "saturated",
    "conc.": "concentrated",
    "anhyd.": "anhydrous",
    "anh.": "anhydrous",
    "abs.": "absolute",
    "dr": "diastereomeric ratio",
    "ee": "enantiomeric excess",
    "de": "diastereomeric excess",
    "rxn": "reaction",
    "vac": "vacuum",
}


@dataclass
class NormalizationResult:
    original: str
    normalized: str
    changed: bool


def normalize_paragraph(text: str) -> NormalizationResult:
    """Return both the original and normalized paragraph.

    The agent receives ``normalized`` (cleaner, easier to extract from); the
    ``source_paragraph`` field in the emitted draft should preserve the
    ``original`` so downstream tooling can audit against the actual input.
    """
    original = text
    # 1. NFC normalize so combining characters collapse cleanly.
    normalized = unicodedata.normalize("NFC", text)
    # 2. Substitute the troublesome single chars.
    for src, dst in _CHAR_FIXUPS.items():
        normalized = normalized.replace(src, dst)
    # 3. Expand abbreviations at word boundaries. Case-insensitive match
    #    but preserve the surrounding casing.
    for short, full in _ABBREVIATION_EXPANSIONS.items():
        pattern = rf"(?<!\w){re.escape(short)}(?!\w)"
        normalized = re.sub(pattern, full, normalized, flags=re.IGNORECASE)
    # 4. Collapse runs of whitespace (newlines preserved as single spaces;
    #    chemistry paragraphs come as a single block anyway).
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"\n{2,}", "\n", normalized)
    normalized = normalized.strip()
    return NormalizationResult(
        original=original,
        normalized=normalized,
        changed=(normalized != original),
    )
