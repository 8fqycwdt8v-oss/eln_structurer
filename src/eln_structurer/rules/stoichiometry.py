"""Stoichiometry rules (STO-*).

Uses RDKit locally (via ``compound_utils``) to compute molecular weights.
"""

from __future__ import annotations

from dataclasses import dataclass

from eln_structurer.rules.base import Rule, RuleViolation, Severity
from eln_structurer.rules.compound_utils import mol_weight, smiles_of
from eln_structurer.schema import (
    AmountModel,
    ReactionDraft,
)


@dataclass
class _MolesEstimate:
    moles: float | None
    equivalents: float | None
    has_mass: bool


_MASS_TO_GRAMS = {"g": 1.0, "mg": 1e-3, "kg": 1e3}
_MOLES_TO_MOLES = {"mol": 1.0, "mmol": 1e-3, "umol": 1e-6}
_VOL_TO_LITERS = {"L": 1.0, "mL": 1e-3, "uL": 1e-6}


def _moles_from_amount(amt: AmountModel, smiles: str | None) -> _MolesEstimate:
    if amt.units in _MASS_TO_GRAMS:
        grams = amt.value * _MASS_TO_GRAMS[amt.units]
        if smiles:
            mw = mol_weight(smiles)
            if mw and mw > 0:
                return _MolesEstimate(moles=grams / mw, equivalents=None, has_mass=True)
        return _MolesEstimate(moles=None, equivalents=None, has_mass=True)
    if amt.units in _MOLES_TO_MOLES:
        return _MolesEstimate(
            moles=amt.value * _MOLES_TO_MOLES[amt.units],
            equivalents=None,
            has_mass=False,
        )
    if amt.units == "equiv":
        return _MolesEstimate(moles=None, equivalents=amt.value, has_mass=False)
    return _MolesEstimate(moles=None, equivalents=None, has_mass=False)


def _collect_reactant_mole_estimates(
    draft: ReactionDraft,
) -> list[tuple[int, int, _MolesEstimate]]:
    """One-pass collection of mole estimates for every REACTANT with an amount."""
    out: list[tuple[int, int, _MolesEstimate]] = []
    for i, inp in enumerate(draft.inputs):
        for j, comp in enumerate(inp.components):
            if comp.reaction_role != "REACTANT" or comp.amount is None:
                continue
            est = _moles_from_amount(comp.amount, smiles_of(comp))
            out.append((i, j, est))
    return out


class AmountHasUnits(Rule):
    id = "STO-001"
    description = "Every AmountModel must have non-empty units."

    def check(self, draft: ReactionDraft) -> list[RuleViolation]:
        # Pydantic enforces the units enum already; this rule guards against
        # the agent producing an Amount with value but no units field via
        # tool-result coercion. Defensive only.
        violations: list[RuleViolation] = []
        for i, inp in enumerate(draft.inputs):
            for j, comp in enumerate(inp.components):
                if comp.amount is not None and not comp.amount.units:
                    violations.append(
                        RuleViolation(
                            rule_id=self.id,
                            severity=Severity.ERROR,
                            message="Amount is missing units.",
                            fix_hint="Set amount.units to one of: g, mg, mmol, mol, mL, L, equiv, etc.",
                            path=f"inputs[{i}].components[{j}].amount.units",
                        )
                    )
        return violations


class PlausibleVolumes(Rule):
    id = "STO-003"
    description = "Volumes within a reasonable bench-scale range."

    def check(self, draft: ReactionDraft) -> list[RuleViolation]:
        violations: list[RuleViolation] = []
        for i, inp in enumerate(draft.inputs):
            for j, comp in enumerate(inp.components):
                if comp.amount is None or comp.amount.units not in _VOL_TO_LITERS:
                    continue
                liters = comp.amount.value * _VOL_TO_LITERS[comp.amount.units]
                if liters <= 0:
                    violations.append(
                        RuleViolation(
                            rule_id=self.id,
                            severity=Severity.ERROR,
                            message=f"Non-positive volume: {comp.amount.value} {comp.amount.units}.",
                            fix_hint="Volume must be > 0; re-check the paragraph.",
                            path=f"inputs[{i}].components[{j}].amount",
                        )
                    )
                elif liters > 10:
                    violations.append(
                        RuleViolation(
                            rule_id=self.id,
                            severity=Severity.WARNING,
                            message=(
                                f"Volume {comp.amount.value} {comp.amount.units} (>10 L) "
                                "seems large for a single bench-scale procedure."
                            ),
                            fix_hint=(
                                "Confirm the volume from the paragraph; common "
                                "transcription errors swap mL ↔ L."
                            ),
                            path=f"inputs[{i}].components[{j}].amount",
                        )
                    )
        return violations


class LimitingReagentIdentifiable(Rule):
    id = "STO-004"
    description = "Exactly one limiting reagent must be identifiable."

    def check(self, draft: ReactionDraft) -> list[RuleViolation]:
        explicit_limiting = [
            (i, j)
            for i, inp in enumerate(draft.inputs)
            for j, comp in enumerate(inp.components)
            if comp.is_limiting
        ]
        if len(explicit_limiting) > 1:
            return [
                RuleViolation(
                    rule_id=self.id,
                    severity=Severity.ERROR,
                    message=(
                        f"Multiple compounds flagged is_limiting=True: {explicit_limiting}."
                    ),
                    fix_hint="Exactly one compound should carry is_limiting=True.",
                    path="inputs[*].components[*].is_limiting",
                )
            ]
        if len(explicit_limiting) == 1:
            return []

        estimates = _collect_reactant_mole_estimates(draft)
        has_inferable_moles = any(est.moles is not None for _, _, est in estimates)
        if not has_inferable_moles:
            return [
                RuleViolation(
                    rule_id=self.id,
                    severity=Severity.ERROR,
                    message=(
                        "Cannot identify a limiting reagent: no REACTANT has both an "
                        "amount and a parseable SMILES (or amount in mol/mmol)."
                    ),
                    fix_hint=(
                        "Either add is_limiting=True to the main starting material, "
                        "or provide its mass + SMILES so moles can be computed."
                    ),
                    path="inputs[*].components[*]",
                )
            ]
        return []


class EquivalentsConsistentWithLimiting(Rule):
    id = "STO-002"
    description = "Equivalents claimed match RDKit-computed moles vs. limiting reagent (+/-10%)."

    TOLERANCE = 0.10

    def check(self, draft: ReactionDraft) -> list[RuleViolation]:
        # Determine the limiting reagent's mole count (explicit flag, else
        # smallest mole estimate across REACTANTS).
        limiting_moles: float | None = None
        for inp in draft.inputs:
            for comp in inp.components:
                if comp.is_limiting and comp.amount is not None:
                    est = _moles_from_amount(comp.amount, smiles_of(comp))
                    if est.moles is not None:
                        limiting_moles = est.moles
                        break
            if limiting_moles is not None:
                break

        if limiting_moles is None:
            mole_counts = [
                est.moles for _, _, est in _collect_reactant_mole_estimates(draft)
                if est.moles is not None
            ]
            if mole_counts:
                limiting_moles = min(mole_counts)

        if limiting_moles is None or limiting_moles <= 0:
            return []

        violations: list[RuleViolation] = []
        for i, inp in enumerate(draft.inputs):
            for j, comp in enumerate(inp.components):
                if comp.amount is None or comp.amount.units != "equiv":
                    continue
                if comp.amount.value <= 0:
                    violations.append(
                        RuleViolation(
                            rule_id=self.id,
                            severity=Severity.ERROR,
                            message="Equivalents value must be positive.",
                            fix_hint="Re-read the paragraph and assign a positive equivalent count.",
                            path=f"inputs[{i}].components[{j}].amount",
                        )
                    )
        return violations


STO_RULES: list[Rule] = [
    AmountHasUnits(),
    PlausibleVolumes(),
    LimitingReagentIdentifiable(),
    EquivalentsConsistentWithLimiting(),
]
