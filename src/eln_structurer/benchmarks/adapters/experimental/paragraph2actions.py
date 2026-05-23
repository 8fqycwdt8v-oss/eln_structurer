"""Adapter for IBM's paragraph2actions (rxn4chemistry).

paragraph2actions emits an action sequence built from a fixed action
vocabulary (Add, Stir, Wait, Filter, Wash, Extract, Concentrate, DrySolid,
DrySolution, Reflux, Degas, MakeSolution, Microwave, CollectLayer, Yield,
plus the "fake" actions NoAction / InvalidAction / OtherLanguage /
FollowOtherProcedure). We normalize that vocabulary to ORD workup types and
extract reactant names from Add-style actions.

⚠️  UNVERIFIED. The adapter was authored without ever running the real
package — paragraph2actions pins an old torch (<1.5) that conflicts with
our base venv on Python 3.11, and was never installed during development.
The class / module paths below match the GitHub README's documented API
but have not been exercised end-to-end. When you actually install the
package via scripts/setup_bench_envs.sh, expect to debug:
- the import path may be `paragraph2actions.readable_converter` rather
  than `action_extractor` (the README is unclear)
- the model may require an explicit checkpoint path argument
- the action class attribute names (.action_name, .material) are educated
  guesses from the upstream source
Treat this adapter as a starting scaffold, not a working integration.
"""

from __future__ import annotations

import importlib

from eln_structurer.benchmarks.adapters.base import (
    Adapter,
    AdapterError,
    AdapterUnavailable,
)
from eln_structurer.benchmarks.canonical import CanonicalReaction, normalize_name


# Map of real paragraph2actions action class names (uppercased) to the
# closest ORD workup type. NoAction / InvalidAction are dropped via None.
_VERB_MAP = {
    "ADD": "ADDITION",
    "STIR": "STIRRING",
    "WAIT": "WAIT",
    "FILTER": "FILTRATION",
    "WASH": "WASH",
    "EXTRACT": "EXTRACTION",
    "CONCENTRATE": "CONCENTRATION",
    "DRYSOLID": "DRY_WITH_MATERIAL",
    "DRYSOLUTION": "DRY_WITH_MATERIAL",
    "REFLUX": "TEMPERATURE",
    "DEGAS": "CUSTOM",
    "MAKESOLUTION": "DISSOLUTION",
    "MICROWAVE": "TEMPERATURE",
    "COLLECTLAYER": "EXTRACTION",
    "YIELD": None,         # paragraph2actions Yield carries product data, not a workup
    "NOACTION": None,
    "INVALIDACTION": None,
    "OTHERLANGUAGE": None,
    "FOLLOWOTHERPROCEDURE": None,
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
