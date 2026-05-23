"""Shared fixtures for the eln_structurer test suite."""

from __future__ import annotations

from collections.abc import Iterable

import pytest

from eln_structurer.rules.base import RuleViolation
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


def rule_ids(violations: Iterable[RuleViolation]) -> set[str]:
    """Set of rule IDs from a violations iterable; reused by every rules test."""
    return {v.rule_id for v in violations}


@pytest.fixture(autouse=True)
def _clear_module_caches():
    """Clear lru_caches between tests so global state never leaks.

    The codebase memoises three hot paths: chemistry.parse_mol (SMILES →
    Mol), prompts.build_system_prompt, and
    prompts.schema.compressed_reaction_draft_schema. Without this fixture,
    a test that mutates module-level state (e.g. patches the schema)
    would taint every subsequent test that consults the cache.
    """
    from eln_structurer.chemistry import parse_mol
    from eln_structurer.prompts import build_system_prompt
    from eln_structurer.prompts.schema import (
        compressed_reaction_draft_schema,
        reaction_draft_json_schema,
    )

    parse_mol.cache_clear()
    build_system_prompt.cache_clear()
    compressed_reaction_draft_schema.cache_clear()
    reaction_draft_json_schema.cache_clear()
    yield


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
    # Acetic anhydride is both the medium and the acyl donor in this
    # reaction — chemically the substance is a REAGENT (it participates
    # in the bond-forming step), not a SOLVENT. Marking it SOLVENT would
    # make STR-003 fire because the donor heavy-atom budget would be
    # insufficient to construct the aspirin product.
    acetic_anhydride = CompoundModel(
        identifiers=[
            CompoundIdentifierModel(type="NAME", value="acetic anhydride"),
            CompoundIdentifierModel(type="SMILES", value="CC(=O)OC(C)=O"),
        ],
        amount=AmountModel(value=5.0, units="mL"),
        reaction_role="REAGENT",
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
        # The original paragraph just says "warmed to 85 °C" without
        # specifying a bath/heater apparatus; use UNSPECIFIED control so
        # ORD-002 (solvent-required-when-heating) does not fire for what is
        # in fact a neat-in-excess-reagent procedure.
        conditions=ConditionsModel(
            temperature=TemperatureModel(setpoint_celsius=85.0, control_type="UNSPECIFIED"),
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
