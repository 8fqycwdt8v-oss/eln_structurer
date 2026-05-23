"""Shared text-normalisation helpers.

Replaces the three separate normalisation implementations that previously
lived in benchmarks/canonical.py, preprocess.py, and rules/numeric_grounding.py.
Pure functions — no chemistry, no network.
"""

from __future__ import annotations

import re


_WHITESPACE_RE = re.compile(r"\s+")
# Punctuation characters dropped when ``drop_punctuation=True``. Hyphens are
# deliberately kept since they're meaningful in chemical names (4-bromoanisole).
_DROP_CHARS = str.maketrans({c: " " for c in ".,;:()[]{}/\\'\"!?"})


def collapse_whitespace(text: str) -> str:
    """Collapse runs of whitespace (including newlines) into single spaces."""
    return _WHITESPACE_RE.sub(" ", text).strip()


def normalize_text(text: str, *, drop_punctuation: bool = False) -> str:
    """Lowercase + whitespace-collapse. Optionally drop punctuation."""
    t = (text or "").lower()
    if drop_punctuation:
        t = t.translate(_DROP_CHARS)
    return collapse_whitespace(t)


def normalize_compound_name(name: str) -> str:
    """Aggressive normalisation suitable for compound-name comparison."""
    return normalize_text(name, drop_punctuation=True)


def normalize_for_substring_search(text: str) -> str:
    """Light normalisation suitable for substring grounding checks (NUM-001).

    Identical to ``normalize_text`` without punctuation stripping — we
    want to preserve units like 'g' and special chars like '%' in quotes.
    """
    return normalize_text(text)
