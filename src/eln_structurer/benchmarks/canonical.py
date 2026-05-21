"""Canonical reaction projection used by the benchmark scorer.

Every adapter normalizes its native output to a ``CanonicalReaction`` so that
heterogeneous tools (full-ORD extractors, action-sequence models, multimodal
parsers) can be compared on a common axis.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from eln_structurer.schema import ReactionDraft


_NORMALIZE_RE = re.compile(r"\s+")
_DROP_CHARS = str.maketrans({c: " " for c in ".,;:()[]{}/\\'\"!?"})


def normalize_name(text: str) -> str:
    """Lowercase, strip punctuation and collapse whitespace."""
    t = (text or "").lower().translate(_DROP_CHARS)
    t = _NORMALIZE_RE.sub(" ", t).strip()
    # Drop very short stop-words but keep chemistry tokens.
    return t


def normalize_smiles(smiles: str) -> str:
    """Canonicalize a SMILES via RDKit; returns input on parse failure."""
    if not smiles:
        return ""
    from eln_structurer.chemistry import canonical_smiles

    return canonical_smiles(smiles) or smiles


@dataclass
class CanonicalReaction:
    reactant_names: set[str] = field(default_factory=set)
    reactant_smiles: set[str] = field(default_factory=set)
    reagent_names: set[str] = field(default_factory=set)
    solvent_names: set[str] = field(default_factory=set)
    catalyst_names: set[str] = field(default_factory=set)
    product_names: set[str] = field(default_factory=set)
    product_smiles: set[str] = field(default_factory=set)
    yield_percent: float | None = None
    temperature_celsius: float | None = None
    duration_minutes: float | None = None
    workup_verbs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "reactant_names": sorted(self.reactant_names),
            "reactant_smiles": sorted(self.reactant_smiles),
            "reagent_names": sorted(self.reagent_names),
            "solvent_names": sorted(self.solvent_names),
            "catalyst_names": sorted(self.catalyst_names),
            "product_names": sorted(self.product_names),
            "product_smiles": sorted(self.product_smiles),
            "yield_percent": self.yield_percent,
            "temperature_celsius": self.temperature_celsius,
            "duration_minutes": self.duration_minutes,
            "workup_verbs": list(self.workup_verbs),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CanonicalReaction":
        def _set(key: str) -> set[str]:
            return {normalize_name(v) for v in payload.get(key, []) if v}

        def _smiles_set(key: str) -> set[str]:
            return {normalize_smiles(v) for v in payload.get(key, []) if v}

        return cls(
            reactant_names=_set("reactant_names"),
            reactant_smiles=_smiles_set("reactant_smiles"),
            reagent_names=_set("reagent_names"),
            solvent_names=_set("solvent_names"),
            catalyst_names=_set("catalyst_names"),
            product_names=_set("product_names"),
            product_smiles=_smiles_set("product_smiles"),
            yield_percent=payload.get("yield_percent"),
            temperature_celsius=payload.get("temperature_celsius"),
            duration_minutes=payload.get("duration_minutes"),
            workup_verbs=[v.upper() for v in payload.get("workup_verbs", []) if v],
        )


def canonicalize_draft(draft: ReactionDraft) -> CanonicalReaction:
    """Project a ReactionDraft (our internal type) into CanonicalReaction."""
    canon = CanonicalReaction()
    for inp in draft.inputs:
        for comp in inp.components:
            name = next(
                (i.value for i in comp.identifiers if i.type in {"NAME", "IUPAC_NAME"}),
                None,
            )
            smi = next(
                (i.value for i in comp.identifiers if i.type == "SMILES"),
                None,
            )
            norm_name = normalize_name(name) if name else ""
            norm_smi = normalize_smiles(smi) if smi else ""
            role = comp.reaction_role
            if role == "REACTANT":
                if norm_name:
                    canon.reactant_names.add(norm_name)
                if norm_smi:
                    canon.reactant_smiles.add(norm_smi)
            elif role == "SOLVENT" and norm_name:
                canon.solvent_names.add(norm_name)
            elif role == "CATALYST" and norm_name:
                canon.catalyst_names.add(norm_name)
            elif role == "REAGENT" and norm_name:
                canon.reagent_names.add(norm_name)

    for outcome in draft.outcomes:
        for prod in outcome.products:
            name = next(
                (i.value for i in prod.compound.identifiers if i.type in {"NAME", "IUPAC_NAME"}),
                None,
            )
            smi = next(
                (i.value for i in prod.compound.identifiers if i.type == "SMILES"),
                None,
            )
            if name:
                canon.product_names.add(normalize_name(name))
            if smi:
                canon.product_smiles.add(normalize_smiles(smi))
            for m in prod.measurements:
                if m.type == "YIELD" and canon.yield_percent is None:
                    canon.yield_percent = float(m.value)

    if draft.conditions.temperature and draft.conditions.temperature.setpoint_celsius is not None:
        canon.temperature_celsius = float(draft.conditions.temperature.setpoint_celsius)
    if draft.conditions.duration_minutes is not None:
        canon.duration_minutes = float(draft.conditions.duration_minutes)
    else:
        for outcome in draft.outcomes:
            if outcome.reaction_time_minutes is not None:
                canon.duration_minutes = float(outcome.reaction_time_minutes)
                break

    for wu in sorted(draft.workups, key=lambda w: w.order):
        canon.workup_verbs.append(wu.type.upper())

    return canon


def canonicalize_ord_json(json_text: str) -> CanonicalReaction:
    """Project an ord-schema JSON-formatted Reaction message into canonical form.

    Used by the eln_structurer adapter, which emits canonical ORD JSON. We parse
    the JSON ourselves instead of relying on protobuf round-tripping so this
    function stays useful for ad-hoc inputs.
    """
    data = json.loads(json_text)
    canon = CanonicalReaction()

    inputs = data.get("inputs", {})
    for _key, inp in inputs.items():
        for comp in inp.get("components", []):
            name_ident = next(
                (i for i in comp.get("identifiers", []) if i.get("type") in {"NAME", "IUPAC_NAME"}),
                None,
            )
            smi_ident = next(
                (i for i in comp.get("identifiers", []) if i.get("type") == "SMILES"),
                None,
            )
            norm_name = normalize_name(name_ident["value"]) if name_ident else ""
            norm_smi = normalize_smiles(smi_ident["value"]) if smi_ident else ""
            role = comp.get("reactionRole") or comp.get("reaction_role") or "REACTANT"
            if role == "REACTANT":
                if norm_name:
                    canon.reactant_names.add(norm_name)
                if norm_smi:
                    canon.reactant_smiles.add(norm_smi)
            elif role == "SOLVENT" and norm_name:
                canon.solvent_names.add(norm_name)
            elif role == "CATALYST" and norm_name:
                canon.catalyst_names.add(norm_name)
            elif role == "REAGENT" and norm_name:
                canon.reagent_names.add(norm_name)

    for outcome in data.get("outcomes", []):
        for prod in outcome.get("products", []):
            for ident in prod.get("identifiers", []):
                t = ident.get("type")
                v = ident.get("value", "")
                if t in {"NAME", "IUPAC_NAME"}:
                    canon.product_names.add(normalize_name(v))
                elif t == "SMILES":
                    canon.product_smiles.add(normalize_smiles(v))
            for m in prod.get("measurements", []):
                if m.get("type") == "YIELD":
                    pct = (m.get("percentage") or {}).get("value")
                    if pct is not None and canon.yield_percent is None:
                        canon.yield_percent = float(pct)
        rt = (outcome.get("reactionTime") or outcome.get("reaction_time") or {}).get("value")
        if rt is not None and canon.duration_minutes is None:
            canon.duration_minutes = float(rt)

    conditions = data.get("conditions", {})
    setpoint = (
        ((conditions.get("temperature") or {}).get("setpoint")) or {}
    ).get("value")
    if setpoint is not None:
        canon.temperature_celsius = float(setpoint)

    for wu in data.get("workups", []):
        verb = wu.get("type", "")
        if verb:
            canon.workup_verbs.append(verb.upper())

    return canon


def load_gold(path: Path) -> CanonicalReaction:
    return CanonicalReaction.from_dict(json.loads(path.read_text()))
