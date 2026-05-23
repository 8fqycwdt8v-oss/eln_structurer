"""Adapter for MIT/Coley's OpenChemIE.

⚠️  UNVERIFIED & SCOPE-MISMATCH. The real OpenChemIE public API is
PDF-centric — its text-extraction methods are
``extract_reactions_from_text_in_pdf(pdf_path)`` and
``extract_reactions_from_text_in_pdf_combined(pdf_path)``, both of which
take a PDF file path rather than a string. There is no public
``extract_reactions_from_text(list[str])`` entry point.

For our single-paragraph benchmark we need to either (a) wrap the paragraph
into a minimal one-page PDF on the fly, or (b) drop this adapter and
acknowledge that OpenChemIE isn't designed for raw-text-string input.
The current implementation calls a non-existent method and will raise
``AttributeError`` the moment the underlying package is installed; treat
it as a stub until someone wires in the PDF wrapping path.
"""

from __future__ import annotations

import importlib

from eln_structurer.benchmarks.adapters.base import (
    Adapter,
    AdapterError,
    AdapterUnavailable,
)
from eln_structurer.benchmarks.canonical import CanonicalReaction


class OpenChemIEAdapter(Adapter):
    name = "openchemie"

    def __init__(self, *, device: str = "cpu") -> None:
        self._model = None
        self.device = device

    async def is_available(self) -> bool:
        try:
            importlib.import_module("openchemie")
            return True
        except ImportError:
            return False

    def _load(self):
        if self._model is not None:
            return self._model
        try:
            oce = importlib.import_module("openchemie")
        except ImportError as exc:
            raise AdapterUnavailable(
                "openchemie is not installed in this environment. "
                "See scripts/setup_bench_envs.sh."
            ) from exc
        try:
            cls = oce.OpenChemIE  # type: ignore[attr-defined]
        except AttributeError as exc:
            raise AdapterError(f"OpenChemIE class not found: {exc}") from exc
        self._model = cls(device=self.device)
        return self._model

    async def extract(self, paragraph: str) -> CanonicalReaction:
        # OpenChemIE's text path expects a PDF, not a paragraph string. Until
        # someone wires up an on-the-fly PDF wrapper, this method raises and
        # the runner records the adapter as failed for that fixture.
        self._load()
        raise AdapterError(
            "OpenChemIE's text-extraction API expects a PDF path, not a "
            "paragraph string. Wrap the paragraph into a single-page PDF "
            "(reportlab / fpdf) and call extract_reactions_from_text_in_pdf "
            "before this adapter can produce a CanonicalReaction. "
            "Adapter intentionally left unimplemented to avoid silent wrong "
            "results — see the module docstring."
        )
