"""Adapter for IBM's paragraph2actions (rxn4chemistry).

paragraph2actions emits an action sequence (STIR, FILTER, ADD, DRYSOLUTION,
NOACTION, plus a handful of others). We normalize that vocabulary to ORD
workup types and extract reactant/product names from ADD-style actions.

paragraph2actions pins an old torch (<1.5) that conflicts with our base venv
on Python 3.11. Install it in a separate environment per
scripts/setup_bench_envs.sh and run the benchmark with that env active.
"""

from __future__ import annotations

import importlib

from eln_structurer.benchmarks.adapters.base import (
    Adapter,
    AdapterError,
    AdapterUnavailable,
)
from eln_structurer.benchmarks.canonical import CanonicalReaction, normalize_name


_VERB_MAP = {
    "STIR": "STIRRING",
    "WAIT": "WAIT",
    "FILTER": "FILTRATION",
    "DRYSOLUTION": "DRY_WITH_MATERIAL",
    "DRYSOLID": "DRY_WITH_MATERIAL",
    "WASH": "WASH",
    "CONCENTRATE": "CONCENTRATION",
    "EXTRACT": "EXTRACTION",
    "PURIFY": "FLASH_CHROMATOGRAPHY",
    "PH": "PH_ADJUST",
    "QUENCH": "ADDITION",
    "MAKESOLUTION": "DISSOLUTION",
    "ADD": "ADDITION",
    "PARTITION": "EXTRACTION",
    "DISTILL": "DISTILLATION",
    "RECRYSTALLIZE": "CUSTOM",
    "REMOVE": "CONCENTRATION",
    "COLLECTLAYER": "EXTRACTION",
    "PHASESEP": "EXTRACTION",
    "NOACTION": None,
}


class Paragraph2ActionsAdapter(Adapter):
    name = "paragraph2actions"

    def __init__(self) -> None:
        self._extractor = None

    async def is_available(self) -> bool:
        try:
            importlib.import_module("paragraph2actions")
            return True
        except ImportError:
            return False

    def _load(self):
        if self._extractor is not None:
            return self._extractor
        try:
            mod = importlib.import_module("paragraph2actions.action_extractor")
        except ImportError as exc:
            raise AdapterUnavailable(
                "paragraph2actions is not installed in this environment. "
                "See scripts/setup_bench_envs.sh."
            ) from exc
        try:
            cls = mod.ActionExtractor  # type: ignore[attr-defined]
        except AttributeError as exc:
            raise AdapterError(
                f"paragraph2actions ActionExtractor class not found: {exc}"
            ) from exc
        self._extractor = cls()
        return self._extractor

    async def extract(self, paragraph: str) -> CanonicalReaction:
        extractor = self._load()
        try:
            actions = extractor.extract_actions(paragraph)
        except Exception as exc:  # pragma: no cover — depends on installed model
            raise AdapterError(f"paragraph2actions runtime error: {exc}") from exc

        canon = CanonicalReaction()
        for act in actions:
            verb = getattr(act, "action_name", str(act)).upper()
            mapped = _VERB_MAP.get(verb)
            if mapped:
                canon.workup_verbs.append(mapped)

            # Best-effort name extraction from common Action structures.
            for attr in ("material", "compound", "reagent"):
                val = getattr(act, attr, None)
                if val:
                    name = getattr(val, "name", str(val))
                    if name and verb == "ADD":
                        canon.reactant_names.add(normalize_name(name))
        return canon
