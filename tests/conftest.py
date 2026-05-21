"""Shared fixtures for the eln_structurer test suite."""

from __future__ import annotations

import pytest

from eln_structurer.schema import (
    AmountModel,
    CompoundIdentifierModel,
    CompoundModel,
    ConditionsModel,
    OutcomeModel,
    ProductMeasurementModel,
    ProductModel,
    ReactionDraft,
    ReactionInputModel,
    StirringModel,
    TemperatureModel,
    WorkupModel,
)


def _build_aspirin_draft() -> ReactionDraft:
    """A minimal, validation-clean draft for the aspirin synthesis."""
    salicylic_acid = CompoundModel(
        identifiers=[
            CompoundIdentifierModel(type="NAME", value="salicylic acid"),
            CompoundIdentifierModel(type="SMILES", value="OC(=O)c1ccccc1O"),
        ],
        amount=AmountModel(value=1.38, units="g"),
        reaction_role="REACTANT",
        is_limiting=True,
    )
    acetic_anhydride = CompoundModel(
        identifiers=[
            CompoundIdentifierModel(type="NAME", value="acetic anhydride"),
            CompoundIdentifierModel(type="SMILES", value="CC(=O)OC(C)=O"),
        ],
        amount=AmountModel(value=5.0, units="mL"),
        reaction_role="SOLVENT",
    )
    h2so4 = CompoundModel(
        identifiers=[CompoundIdentifierModel(type="NAME", value="sulfuric acid")],
        amount=None,
        reaction_role="CATALYST",
    )
    product = CompoundModel(
        identifiers=[
            CompoundIdentifierModel(type="NAME", value="acetylsalicylic acid"),
            CompoundIdentifierModel(type="SMILES", value="CC(=O)Oc1ccccc1C(=O)O"),
        ],
        reaction_role="PRODUCT",
    )

    return ReactionDraft(
        identifiers=[],
        inputs=[
            ReactionInputModel(
                name="limiting_reactant",
                components=[salicylic_acid],
                addition_order=1,
            ),
            ReactionInputModel(
                name="solvent",
                components=[acetic_anhydride],
                addition_order=1,
            ),
            ReactionInputModel(
                name="catalyst",
                components=[h2so4],
                addition_order=2,
            ),
        ],
        conditions=ConditionsModel(
            temperature=TemperatureModel(setpoint_celsius=85.0, control_type="OIL_BATH"),
            stirring=StirringModel(type="MAGNETIC"),
            duration_minutes=30.0,
        ),
        workups=[
            WorkupModel(
                type="ADDITION",
                description="Quenched with cold water (20 mL).",
                components=[
                    CompoundModel(
                        identifiers=[CompoundIdentifierModel(type="NAME", value="water")],
                        amount=AmountModel(value=20.0, units="mL"),
                        reaction_role="WORKUP",
                    )
                ],
                order=1,
            ),
            WorkupModel(
                type="FILTRATION",
                description="Precipitate collected by filtration.",
                order=2,
            ),
            WorkupModel(
                type="WASH",
                description="Washed with cold water (3 x 10 mL).",
                order=3,
            ),
        ],
        outcomes=[
            OutcomeModel(
                products=[
                    ProductModel(
                        compound=product,
                        measurements=[
                            ProductMeasurementModel(type="YIELD", value=90.0, units="%"),
                            ProductMeasurementModel(type="AMOUNT", value=1.62, units="g"),
                        ],
                    )
                ],
                reaction_time_minutes=30.0,
            )
        ],
        notes="Classical aspirin synthesis test fixture.",
        source_paragraph="(redacted for fixture)",
    )


@pytest.fixture
def aspirin_draft() -> ReactionDraft:
    return _build_aspirin_draft()
