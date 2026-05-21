"""Convert the Pydantic ``ReactionDraft`` to ord_schema's protobuf ``Reaction``.

The ord-schema ``Amount`` is a protobuf oneof over ``mass`` / ``moles`` /
``volume`` / ``unmeasured``; equivalents is not a first-class field on
Compound, so we record it under ``Compound.features['equivalents']``.
"""

from __future__ import annotations

from collections import Counter

from ord_schema.message_helpers import json_format, text_format
from ord_schema.proto import reaction_pb2

from eln_structurer.schema import (
    AmountModel,
    CompoundModel,
    ConditionsModel,
    OutcomeModel,
    ProductMeasurementModel,
    ReactionDraft,
    StirringModel,
    TemperatureModel,
    WorkupModel,
)


_MASS_UNIT_PROTO = {
    "g": reaction_pb2.Mass.GRAM,
    "mg": reaction_pb2.Mass.MILLIGRAM,
    "kg": reaction_pb2.Mass.KILOGRAM,
}

_MOLES_UNIT_PROTO = {
    "mol": reaction_pb2.Moles.MOLE,
    "mmol": reaction_pb2.Moles.MILLIMOLE,
    "umol": reaction_pb2.Moles.MICROMOLE,
}

_VOLUME_UNIT_PROTO = {
    "L": reaction_pb2.Volume.LITER,
    "mL": reaction_pb2.Volume.MILLILITER,
    "uL": reaction_pb2.Volume.MICROLITER,
}

_IDENTIFIER_TYPE_PROTO = {
    "NAME": reaction_pb2.CompoundIdentifier.NAME,
    "SMILES": reaction_pb2.CompoundIdentifier.SMILES,
    "INCHI": reaction_pb2.CompoundIdentifier.INCHI,
    "CAS_NUMBER": reaction_pb2.CompoundIdentifier.CAS_NUMBER,
    "IUPAC_NAME": reaction_pb2.CompoundIdentifier.IUPAC_NAME,
}

_ROLE_PROTO = {
    "REACTANT": reaction_pb2.ReactionRole.REACTANT,
    "REAGENT": reaction_pb2.ReactionRole.REAGENT,
    "SOLVENT": reaction_pb2.ReactionRole.SOLVENT,
    "CATALYST": reaction_pb2.ReactionRole.CATALYST,
    "INTERNAL_STANDARD": reaction_pb2.ReactionRole.INTERNAL_STANDARD,
    "WORKUP": reaction_pb2.ReactionRole.WORKUP,
    "PRODUCT": reaction_pb2.ReactionRole.PRODUCT,
}

_STIRRING_PROTO = {
    "MAGNETIC": reaction_pb2.StirringConditions.STIR_BAR,
    "OVERHEAD": reaction_pb2.StirringConditions.OVERHEAD_MIXER,
    "SHAKER": reaction_pb2.StirringConditions.AGITATION,
    "NONE": reaction_pb2.StirringConditions.NONE,
    "UNSPECIFIED": reaction_pb2.StirringConditions.UNSPECIFIED,
}

# Our Literal enum -> ORD TemperatureControl enum. REFLUX has no direct
# counterpart in ORD; map to OIL_BATH. HEATER maps to OIL_BATH for the same
# reason (no dedicated heating-mantle enum value).
_TEMP_CONTROL_PROTO = {
    "AMBIENT": reaction_pb2.TemperatureConditions.TemperatureControl.AMBIENT,
    "ICE_BATH": reaction_pb2.TemperatureConditions.TemperatureControl.ICE_BATH,
    "DRY_ICE": reaction_pb2.TemperatureConditions.TemperatureControl.DRY_ICE_BATH,
    "OIL_BATH": reaction_pb2.TemperatureConditions.TemperatureControl.OIL_BATH,
    "HEATER": reaction_pb2.TemperatureConditions.TemperatureControl.OIL_BATH,
    "REFLUX": reaction_pb2.TemperatureConditions.TemperatureControl.OIL_BATH,
    "WATER_BATH": reaction_pb2.TemperatureConditions.TemperatureControl.WATER_BATH,
    "LIQUID_NITROGEN": reaction_pb2.TemperatureConditions.TemperatureControl.LIQUID_NITROGEN,
    "UNSPECIFIED": reaction_pb2.TemperatureConditions.TemperatureControl.UNSPECIFIED,
}

# Our Literal enum -> ORD ReactionWorkup enum. RECRYSTALLIZATION has no direct
# counterpart; map to CUSTOM with details. DRY_WITH_MATERIAL covers drying.
_WORKUP_TYPE_PROTO = {
    "ADDITION": reaction_pb2.ReactionWorkup.ADDITION,
    "WASH": reaction_pb2.ReactionWorkup.WASH,
    "DRY_WITH_MATERIAL": reaction_pb2.ReactionWorkup.DRY_WITH_MATERIAL,
    "EXTRACTION": reaction_pb2.ReactionWorkup.EXTRACTION,
    "FILTRATION": reaction_pb2.ReactionWorkup.FILTRATION,
    "CONCENTRATION": reaction_pb2.ReactionWorkup.CONCENTRATION,
    "PH_ADJUST": reaction_pb2.ReactionWorkup.PH_ADJUST,
    "DISSOLUTION": reaction_pb2.ReactionWorkup.DISSOLUTION,
    "TEMPERATURE": reaction_pb2.ReactionWorkup.TEMPERATURE,
    "STIRRING": reaction_pb2.ReactionWorkup.STIRRING,
    "WAIT": reaction_pb2.ReactionWorkup.WAIT,
    "DISTILLATION": reaction_pb2.ReactionWorkup.DISTILLATION,
    "FLASH_CHROMATOGRAPHY": reaction_pb2.ReactionWorkup.FLASH_CHROMATOGRAPHY,
    "RECRYSTALLIZATION": reaction_pb2.ReactionWorkup.CUSTOM,
    "CUSTOM": reaction_pb2.ReactionWorkup.CUSTOM,
}

_MEASUREMENT_TYPE_PROTO = {
    "YIELD": reaction_pb2.ProductMeasurement.YIELD,
    "AMOUNT": reaction_pb2.ProductMeasurement.AMOUNT,
    "SELECTIVITY": reaction_pb2.ProductMeasurement.SELECTIVITY,
    "PURITY": reaction_pb2.ProductMeasurement.PURITY,
    "AREA": reaction_pb2.ProductMeasurement.AREA,
    "COUNTS": reaction_pb2.ProductMeasurement.COUNTS,
    "CUSTOM": reaction_pb2.ProductMeasurement.CUSTOM,
}


class ProtoBridgeError(ValueError):
    """Raised when the draft cannot be coerced into a valid Reaction proto."""


def _apply_amount(compound_pb: reaction_pb2.Compound, amount: AmountModel) -> None:
    """Populate a Compound's amount/equivalents according to ``AmountModel.units``."""
    if amount.units in _MASS_UNIT_PROTO:
        compound_pb.amount.mass.value = amount.value
        compound_pb.amount.mass.units = _MASS_UNIT_PROTO[amount.units]
    elif amount.units in _MOLES_UNIT_PROTO:
        compound_pb.amount.moles.value = amount.value
        compound_pb.amount.moles.units = _MOLES_UNIT_PROTO[amount.units]
    elif amount.units in _VOLUME_UNIT_PROTO:
        compound_pb.amount.volume.value = amount.value
        compound_pb.amount.volume.units = _VOLUME_UNIT_PROTO[amount.units]
    elif amount.units == "equiv":
        # Equivalents are not first-class on Compound; record via features map
        # and mark the amount as UNSPECIFIED so ord-schema knows the kind oneof
        # was intentionally left empty for a non-mass/moles/volume quantity.
        compound_pb.amount.unmeasured.type = reaction_pb2.UnmeasuredAmount.CUSTOM
        compound_pb.amount.unmeasured.details = f"{amount.value} equivalents"
        compound_pb.features["equivalents"].float_value = amount.value
    elif amount.units in {"mass_pct", "mol_pct", "vol_pct"}:
        compound_pb.amount.unmeasured.type = reaction_pb2.UnmeasuredAmount.CUSTOM
        compound_pb.amount.unmeasured.details = f"{amount.value} {amount.units}"
    else:  # pragma: no cover — Pydantic Literal already guards
        raise ProtoBridgeError(f"Unknown amount units: {amount.units!r}")


def _build_compound(comp: CompoundModel) -> reaction_pb2.Compound:
    compound_pb = reaction_pb2.Compound()
    for ident in comp.identifiers:
        ident_pb = compound_pb.identifiers.add()
        ident_pb.type = _IDENTIFIER_TYPE_PROTO[ident.type]
        ident_pb.value = ident.value
    if comp.amount is not None:
        _apply_amount(compound_pb, comp.amount)
    else:
        # ord-schema requires every input component to have an amount.
        # When the paragraph doesn't quantify a catalyst/reagent, encode it
        # as an UnmeasuredAmount so the schema stays valid.
        if comp.reaction_role == "CATALYST":
            compound_pb.amount.unmeasured.type = reaction_pb2.UnmeasuredAmount.CATALYTIC
        else:
            compound_pb.amount.unmeasured.type = reaction_pb2.UnmeasuredAmount.CUSTOM
            compound_pb.amount.unmeasured.details = "amount not specified in paragraph"
    compound_pb.reaction_role = _ROLE_PROTO[comp.reaction_role]
    compound_pb.is_limiting = comp.is_limiting
    return compound_pb


def _unique_input_key(used: Counter, raw_name: str) -> str:
    key = raw_name.strip() or "input"
    used[key] += 1
    if used[key] == 1:
        return key
    return f"{key}_{used[key]}"


def _apply_temperature(
    reaction_pb: reaction_pb2.Reaction, temp: TemperatureModel
) -> None:
    tc = reaction_pb.conditions.temperature
    tc.control.type = _TEMP_CONTROL_PROTO.get(
        temp.control_type,
        reaction_pb2.TemperatureConditions.TemperatureControl.UNSPECIFIED,
    )
    if temp.setpoint_celsius is not None:
        tc.setpoint.value = temp.setpoint_celsius
        tc.setpoint.units = reaction_pb2.Temperature.CELSIUS


def _apply_stirring(reaction_pb: reaction_pb2.Reaction, stir: StirringModel) -> None:
    sc = reaction_pb.conditions.stirring
    sc.type = _STIRRING_PROTO.get(stir.type, reaction_pb2.StirringConditions.UNSPECIFIED)
    if stir.rate_rpm is not None:
        sc.rate.rpm = stir.rate_rpm


def _apply_conditions(
    reaction_pb: reaction_pb2.Reaction, conditions: ConditionsModel
) -> None:
    if conditions.temperature is not None:
        _apply_temperature(reaction_pb, conditions.temperature)
    if conditions.stirring is not None:
        _apply_stirring(reaction_pb, conditions.stirring)
    if conditions.pressure_atm is not None:
        pc = reaction_pb.conditions.pressure
        pc.setpoint.value = conditions.pressure_atm
        pc.setpoint.units = reaction_pb2.Pressure.ATMOSPHERE
    if conditions.atmosphere:
        existing = reaction_pb.conditions.details or ""
        reaction_pb.conditions.details = (
            f"{existing} atmosphere: {conditions.atmosphere}".strip()
        )


def _build_workup(wu: WorkupModel) -> reaction_pb2.ReactionWorkup:
    wu_pb = reaction_pb2.ReactionWorkup()
    wu_pb.type = _WORKUP_TYPE_PROTO[wu.type]
    wu_pb.details = wu.description
    if wu.duration_minutes is not None:
        wu_pb.duration.value = wu.duration_minutes
        wu_pb.duration.units = reaction_pb2.Time.MINUTE
    if wu.temperature_celsius is not None:
        wu_pb.temperature.setpoint.value = wu.temperature_celsius
        wu_pb.temperature.setpoint.units = reaction_pb2.Temperature.CELSIUS
    for comp in wu.components:
        wu_pb.input.components.add().CopyFrom(_build_compound(comp))
    return wu_pb


def _build_outcome(outcome: OutcomeModel) -> reaction_pb2.ReactionOutcome:
    oc_pb = reaction_pb2.ReactionOutcome()
    if outcome.reaction_time_minutes is not None:
        oc_pb.reaction_time.value = outcome.reaction_time_minutes
        oc_pb.reaction_time.units = reaction_pb2.Time.MINUTE
    for prod in outcome.products:
        prod_pb = oc_pb.products.add()
        compound_pb = _build_compound(prod.compound)
        for ident in compound_pb.identifiers:
            prod_pb.identifiers.add().CopyFrom(ident)
        prod_pb.is_desired_product = True
        prod_pb.reaction_role = reaction_pb2.ReactionRole.PRODUCT
        for m in prod.measurements:
            _apply_measurement(prod_pb, m)
    return oc_pb


def _apply_measurement(
    prod_pb: reaction_pb2.ProductCompound, m: ProductMeasurementModel
) -> None:
    mm = prod_pb.measurements.add()
    mm.type = _MEASUREMENT_TYPE_PROTO[m.type]
    if m.type in {"YIELD", "PURITY"}:
        mm.percentage.value = m.value
    elif m.type == "SELECTIVITY":
        mm.float_value.value = m.value
    elif m.type == "AMOUNT":
        # ord-schema's ProductMeasurement.amount is an Amount oneof; encode
        # a generic float instead unless we know the units.
        mm.float_value.value = m.value
    else:
        mm.float_value.value = m.value
    if m.units:
        mm.details = m.units
    mm.is_normalized = m.is_normalized


def _build_identifiers(draft: ReactionDraft) -> list[reaction_pb2.ReactionIdentifier]:
    out: list[reaction_pb2.ReactionIdentifier] = []
    for ident in draft.identifiers:
        ri = reaction_pb2.ReactionIdentifier()
        if ident.type == "SMILES":
            ri.type = reaction_pb2.ReactionIdentifier.REACTION_SMILES
        else:
            ri.type = reaction_pb2.ReactionIdentifier.CUSTOM
        ri.value = ident.value
        out.append(ri)
    return out


def draft_to_proto(draft: ReactionDraft) -> reaction_pb2.Reaction:
    """Convert a ``ReactionDraft`` to ord_schema's ``reaction_pb2.Reaction``."""
    reaction_pb = reaction_pb2.Reaction()

    for ri in _build_identifiers(draft):
        reaction_pb.identifiers.add().CopyFrom(ri)

    used_keys: Counter = Counter()
    for inp in draft.inputs:
        key = _unique_input_key(used_keys, inp.name)
        ri_pb = reaction_pb.inputs[key]
        for comp in inp.components:
            ri_pb.components.add().CopyFrom(_build_compound(comp))
        if inp.addition_order is not None:
            ri_pb.addition_order = inp.addition_order
        if inp.addition_time_minutes is not None:
            ri_pb.addition_time.value = inp.addition_time_minutes
            ri_pb.addition_time.units = reaction_pb2.Time.MINUTE
        if inp.addition_temperature_celsius is not None:
            ri_pb.addition_temperature.value = inp.addition_temperature_celsius
            ri_pb.addition_temperature.units = reaction_pb2.Temperature.CELSIUS

    _apply_conditions(reaction_pb, draft.conditions)

    for wu in draft.workups:
        reaction_pb.workups.add().CopyFrom(_build_workup(wu))

    for outcome in draft.outcomes:
        reaction_pb.outcomes.add().CopyFrom(_build_outcome(outcome))

    if draft.notes:
        reaction_pb.notes.procedure_details = draft.notes

    # Provenance: ord-schema requires record_created with a Person that has
    # an email. We fill a synthetic, schema-valid stub — this is a machine-
    # generated record, not a real lab notebook entry.
    reaction_pb.provenance.record_created.time.value = "1970-01-01T00:00:00Z"
    reaction_pb.provenance.record_created.person.name = "eln_structurer"
    reaction_pb.provenance.record_created.person.email = "eln-structurer@invalid.local"

    return reaction_pb


def serialize_reaction(
    reaction_pb: reaction_pb2.Reaction, fmt: str = "json"
) -> str:
    """Serialize to ``json`` or ``pbtxt``."""
    if fmt == "json":
        return json_format.MessageToJson(reaction_pb, indent=2)
    if fmt == "pbtxt":
        return text_format.MessageToString(reaction_pb)
    raise ValueError(f"Unknown serialization format: {fmt!r}")
