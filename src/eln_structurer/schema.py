"""Pydantic mirror of the ORD subset we use for single-reaction paragraphs.

The LLM emits a ``ReactionDraft`` (JSON), the rule harness operates on it, and
``proto_bridge.draft_to_proto`` converts it to ``ord_schema.proto.reaction_pb2.Reaction``.

We deliberately keep this flatter and more LLM-friendly than the raw ORD protobuf:
- ``inputs`` is a list (not a map keyed by string)
- ``Amount`` is a single (value, units) pair instead of a protobuf oneof
- enums are restricted to a small, well-known set
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

# ---------------------------------------------------------------------------
# Enums (kept as Literal types so they appear verbatim in JSON Schema output)
# ---------------------------------------------------------------------------

AmountUnit = Literal[
    "g", "mg", "kg",
    "mol", "mmol", "umol",
    "L", "mL", "uL",
    "equiv",
    "mass_pct", "mol_pct", "vol_pct",
]

IdentifierType = Literal["NAME", "SMILES", "INCHI", "CAS_NUMBER", "IUPAC_NAME"]

ReactionRole = Literal[
    "REACTANT",
    "REAGENT",
    "SOLVENT",
    "CATALYST",
    "INTERNAL_STANDARD",
    "WORKUP",
    "PRODUCT",
]

TempControlType = Literal[
    "AMBIENT", "ICE_BATH", "DRY_ICE", "OIL_BATH", "HEATER",
    "REFLUX", "WATER_BATH", "LIQUID_NITROGEN", "UNSPECIFIED",
]

StirringType = Literal["MAGNETIC", "OVERHEAD", "SHAKER", "NONE", "UNSPECIFIED"]

WorkupType = Literal[
    "ADDITION", "WASH", "DRY_WITH_MATERIAL", "EXTRACTION", "FILTRATION",
    "CONCENTRATION", "PH_ADJUST", "DISSOLUTION", "TEMPERATURE",
    "STIRRING", "WAIT", "DISTILLATION", "FLASH_CHROMATOGRAPHY",
    "RECRYSTALLIZATION", "CUSTOM",
]

ProductMeasurementType = Literal[
    "YIELD", "AMOUNT", "SELECTIVITY", "PURITY", "AREA", "COUNTS", "CUSTOM"
]


# ---------------------------------------------------------------------------
# Leaf models
# ---------------------------------------------------------------------------


class AmountModel(BaseModel):
    """A single quantity such as '1.25 g' or '5 mmol'."""

    value: float = Field(..., description="Numeric value, e.g. 1.25")
    units: AmountUnit = Field(..., description="One of the allowed unit strings")
    source_quote: str | None = Field(
        default=None,
        description=(
            "Exact substring from the source paragraph the value was lifted "
            "from (e.g. '1.38 g, 10.0 mmol'). When set, NUM-001 verifies it "
            "appears verbatim in ReactionDraft.source_paragraph. Use this "
            "for every value transcribed from the paragraph; leave it null "
            "(and set inferred=True) for values you derived."
        ),
    )
    inferred: bool = Field(
        default=False,
        description=(
            "True if the value was derived (e.g. equivalents computed from "
            "masses, hours-to-minutes conversion, °F-to-°C conversion). "
            "Inferred values do NOT need a source_quote."
        ),
    )

    @model_validator(mode="after")
    def _positive_value(self) -> "AmountModel":
        if self.value <= 0:
            raise ValueError(
                f"Amount value must be > 0 (got {self.value} {self.units})"
            )
        return self


class CompoundIdentifierModel(BaseModel):
    """One way of naming a compound (NAME, SMILES, CAS, ...)."""

    type: IdentifierType
    value: str


class CompoundModel(BaseModel):
    """A single chemical entity with role, amount, and identifiers."""

    identifiers: list[CompoundIdentifierModel] = Field(
        default_factory=list,
        description="At least one of NAME or SMILES must be present.",
    )
    amount: AmountModel | None = None
    reaction_role: ReactionRole = "REACTANT"
    is_limiting: bool = False


class ReactionInputModel(BaseModel):
    """A logical 'addition step' — one or more compounds added together."""

    name: str = Field(..., description="Short label, e.g. 'limiting_reactant' or 'base'.")
    components: list[CompoundModel]
    addition_order: int | None = Field(
        default=None,
        description="1-indexed ordering of additions; lower numbers added first.",
    )
    addition_time_minutes: float | None = None
    addition_temperature_celsius: float | None = None


class TemperatureModel(BaseModel):
    setpoint_celsius: float | None = None
    control_type: TempControlType = "UNSPECIFIED"


class StirringModel(BaseModel):
    type: StirringType = "UNSPECIFIED"
    rate_rpm: int | None = None


class ConditionsModel(BaseModel):
    temperature: TemperatureModel | None = None
    stirring: StirringModel | None = None
    pressure_atm: float | None = None
    duration_minutes: float | None = None
    atmosphere: str | None = Field(
        default=None,
        description="Free text such as 'nitrogen', 'argon', 'air'.",
    )


class WorkupModel(BaseModel):
    """A single post-reaction operation (extract, wash, dry, ...)."""

    type: WorkupType
    description: str = Field(..., description="Free-text description of the step.")
    components: list[CompoundModel] = Field(
        default_factory=list,
        description="Materials added during the workup (e.g. wash solvent).",
    )
    temperature_celsius: float | None = None
    duration_minutes: float | None = None
    order: int = Field(..., description="1-indexed step order within the workup.")


class ProductMeasurementModel(BaseModel):
    type: ProductMeasurementType
    value: float
    units: str | None = None
    is_normalized: bool = False
    source_quote: str | None = Field(
        default=None,
        description=(
            "Exact substring from the source paragraph the measurement was "
            "lifted from (e.g. '92%' or '181 mg'). Verified by NUM-001."
        ),
    )
    inferred: bool = Field(
        default=False,
        description="True if the measurement was derived rather than quoted.",
    )


class ProductModel(BaseModel):
    compound: CompoundModel
    measurements: list[ProductMeasurementModel] = Field(default_factory=list)


class OutcomeModel(BaseModel):
    products: list[ProductModel]
    reaction_time_minutes: float | None = None


class ReactionDraft(BaseModel):
    """Top-level LLM-emitted reaction. Bridged to ord_schema.proto.Reaction."""

    identifiers: list[CompoundIdentifierModel] = Field(
        default_factory=list,
        description="Reaction-level identifiers, e.g. a reaction SMILES.",
    )
    inputs: list[ReactionInputModel]
    conditions: ConditionsModel
    workups: list[WorkupModel] = Field(default_factory=list)
    outcomes: list[OutcomeModel] = Field(default_factory=list)
    notes: str | None = None
    source_paragraph: str = Field(
        ...,
        description="The original unstructured paragraph this draft was derived from.",
    )
    unspecified_fields: list[str] = Field(
        default_factory=list,
        description=(
            "JSONPath-like strings naming fields the paragraph did NOT "
            "specify. Use this instead of silently omitting fields. "
            "Examples: 'conditions.duration_minutes', "
            "'conditions.atmosphere', 'outcomes[0].products[0].measurements:YIELD'. "
            "Verified by NUM-003."
        ),
    )

    @model_validator(mode="after")
    def _at_least_one_input(self) -> "ReactionDraft":
        if not self.inputs:
            raise ValueError("ReactionDraft.inputs must contain at least one input")
        return self


def reaction_draft_json_schema() -> dict:
    """Return the JSON Schema for the draft, suitable for embedding in a prompt."""
    return ReactionDraft.model_json_schema()
