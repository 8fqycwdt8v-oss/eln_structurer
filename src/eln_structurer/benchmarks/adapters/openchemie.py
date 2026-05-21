"""Adapter for MIT/Coley's OpenChemIE.

OpenChemIE is a multimodal pipeline targeted at full chemistry papers (text +
tables + reaction schemes). For our single-paragraph benchmark we use only its
text-extraction subcomponent. The package is heavyweight (PyTorch + several
model checkpoints) and is not installed in the base venv; see
scripts/setup_bench_envs.sh.
"""

from __future__ import annotations

import importlib

from eln_structurer.benchmarks.adapters.base import (
    Adapter,
    AdapterError,
    AdapterUnavailable,
)
from eln_structurer.benchmarks.canonical import CanonicalReaction, normalize_name


class OpenChemIEAdapter(Adapter):
    name = "openchemie"

    def __init__(self) -> None:
        self._model = None

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
        self._model = cls()
        return self._model

    async def extract(self, paragraph: str) -> CanonicalReaction:
        model = self._load()
        try:
            reactions = model.extract_reactions_from_text([paragraph])
        except Exception as exc:  # pragma: no cover — depends on installed model
            raise AdapterError(f"openchemie runtime error: {exc}") from exc

        canon = CanonicalReaction()
        for rxn_list in reactions:
            for rxn in rxn_list:
                for r in (rxn.get("reactants") or []):
                    name = r.get("text") or r.get("name")
                    if name:
                        canon.reactant_names.add(normalize_name(name))
                for p in (rxn.get("products") or []):
                    name = p.get("text") or p.get("name")
                    if name:
                        canon.product_names.add(normalize_name(name))
                for cond in (rxn.get("conditions") or []):
                    name = cond.get("text")
                    if name:
                        canon.reagent_names.add(normalize_name(name))
        return canon
