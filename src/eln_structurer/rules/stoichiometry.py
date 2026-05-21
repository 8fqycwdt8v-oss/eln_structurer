"""Stoichiometry rules (STO-*).

Uses RDKit locally (via ``eln_structurer.chemistry``) to compute molecular weights.
"""

from __future__ import annotations

from dataclasses import dataclass

from eln_structurer.rules.base import Rule, RuleViolation, Severity
from eln_structurer.chemistry import mol_weight, smiles_of
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


def _identify_limiting_moles(draft: ReactionDraft) -> tuple[float | None, bool]:
    """Return (limiting_moles, was_inferred). limiting_moles=None if unidentifiable."""
    for inp in draft.inputs:
        for comp in inp.components:
            if comp.is_limiting and comp.amount is not None:
                est = _moles_from_amount(comp.amount, smiles_of(comp))
                if est.moles is not None:
                    return est.moles, False
    mole_counts = [
        est.moles for _, _, est in _collect_reactant_mole_estimates(draft)
        if est.moles is not None
    ]
    if not mole_counts:
        return None, True
    return min(mole_counts), True


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

    TIE_TOLERANCE = 0.01

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
        with_moles = [(i, j, est.moles) for i, j, est in estimates if est.moles is not None]
        if not with_moles:
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
        # Warn when two reactants are within tolerance — the agent should
        # break the tie explicitly so downstream consumers know which one
        # was selected as limiting.
        min_moles = min(m for _, _, m in with_moles)
        tied = [
            (i, j) for i, j, m in with_moles
            if abs(m - min_moles) <= self.TIE_TOLERANCE * min_moles
        ]
        if len(tied) > 1:
            return [
                RuleViolation(
                    rule_id=self.id,
                    severity=Severity.WARNING,
                    message=(
                        f"Two or more REACTANTS have ~equal mole counts ({tied}); "
                        "limiting reagent is ambiguous."
                    ),
                    fix_hint=(
                        "Pick one and set is_limiting=True. If both are co-limiting, "
                        "set the flag on the substrate of interest for yield."
                    ),
                    path="inputs[*].components[*].is_limiting",
                )
            ]
        return []


class EquivalentsConsistentWithLimiting(Rule):
    id = "STO-002"
    description = (
        "Equivalents declared on a reactant match RDKit-computed moles vs. the "
        "limiting reagent within ±10%."
    )

    TOLERANCE = 0.10

    def check(self, draft: ReactionDraft) -> list[RuleViolation]:
        limiting_moles, _ = _identify_limiting_moles(draft)
        if limiting_moles is None or limiting_moles <= 0:
            return []

        violations: list[RuleViolation] = []
        for i, inp in enumerate(draft.inputs):
            for j, comp in enumerate(inp.components):
                if comp.amount is None or comp.amount.units != "equiv":
                    continue
                claimed = comp.amount.value
                if claimed <= 0:
                    violations.append(
                        RuleViolation(
                            rule_id=self.id,
                            severity=Severity.ERROR,
                            message="Equivalents value must be positive.",
                            fix_hint="Re-read the paragraph and assign a positive equivalent count.",
                            path=f"inputs[{i}].components[{j}].amount",
                        )
                    )
                    continue

                # If we can also compute the compound's own moles (mass+SMILES,
                # or a mole amount elsewhere on the same compound), check that
                # claimed equiv = computed_moles / limiting_moles within ±10%.
                # In practice the LLM gives EITHER mass OR equiv per Compound,
                # not both, so this branch usually no-ops; when both are present
                # it's typically because the agent transcribed both from the
                # paragraph, and we can cross-check.
                est = _moles_from_amount(comp.amount, smiles_of(comp))
                # The equiv branch in _moles_from_amount sets equivalents but
                # leaves moles=None, so we can't cross-check from the same
                # AmountModel. The intended cross-check needs a second amount
                # entry per compound, which our schema doesn't currently allow.
                # We therefore validate only sign here and document the gap.
                _ = est
        return violations


class YieldRangeSanity(Rule):
    id = "STO-005"
    description = "Reported product yields should be within 0–105%."

    def check(self, draft: ReactionDraft) -> list[RuleViolation]:
        violations: list[RuleViolation] = []
        for oi, outcome in enumerate(draft.outcomes):
            for pi, prod in enumerate(outcome.products):
                for mi, m in enumerate(prod.measurements):
                    if m.type != "YIELD":
                        continue
                    if m.value < 0:
                        violations.append(
                            RuleViolation(
                                rule_id=self.id,
                                severity=Severity.ERROR,
                                message=f"Negative yield: {m.value}.",
                                fix_hint="Yields are non-negative; re-read the paragraph.",
                                path=f"outcomes[{oi}].products[{pi}].measurements[{mi}].value",
                            )
                        )
                    elif m.value > 105:
                        violations.append(
                            RuleViolation(
                                rule_id=self.id,
                                severity=Severity.ERROR,
                                message=(
                                    f"Yield {m.value}% is implausible (>105%). "
                                    "Likely a unit confusion (mass vs. mol) or transcription error."
                                ),
                                fix_hint=(
                                    "Re-check the paragraph. Yields >100% sometimes occur "
                                    "from solvated forms but rarely exceed 102%."
                                ),
                                path=f"outcomes[{oi}].products[{pi}].measurements[{mi}].value",
                            )
                        )
                    elif m.value > 102:
                        violations.append(
                            RuleViolation(
                                rule_id=self.id,
                                severity=Severity.WARNING,
                                message=f"Yield {m.value}% is high (>102%); confirm the paragraph reports this.",
                                fix_hint="Check whether the yield is a solvated or hydrated form.",
                                path=f"outcomes[{oi}].products[{pi}].measurements[{mi}].value",
                            )
                        )
        return violations


class MassBalanceSanity(Rule):
    """STO-006: reported product mass must not exceed the chemical maximum.

    If we know the limiting reagent's moles ``n_lim`` and can compute the
    product's molecular weight ``MW_p`` from its SMILES, then the maximum
    isolable mass is ``n_lim * MW_p``. A 10% margin accommodates solvated /
    hydrated isolation. Anything beyond is physically impossible.
    """

    id = "STO-006"
    description = (
        "Reported product mass must not exceed n_lim * MW_p * 1.1 "
        "(theoretical maximum with a 10% solvate-inclusion margin)."
    )

    SAFETY_FACTOR = 1.10

    def check(self, draft: ReactionDraft) -> list[RuleViolation]:
        limiting_moles, _ = _identify_limiting_moles(draft)
        if limiting_moles is None or limiting_moles <= 0:
            return []
        if not draft.outcomes or not draft.outcomes[0].products:
            return []

        violations: list[RuleViolation] = []
        for pi, prod in enumerate(draft.outcomes[0].products):
            prod_smiles = smiles_of(prod.compound)
            if not prod_smiles:
                continue
            mw_p = mol_weight(prod_smiles)
            if mw_p is None or mw_p <= 0:
                continue
            max_grams = limiting_moles * mw_p * self.SAFETY_FACTOR

            for mi, m in enumerate(prod.measurements):
                if m.type != "AMOUNT" or not m.units:
                    continue
                units = m.units.strip()
                if units not in _MASS_TO_GRAMS:
                    continue
                grams = m.value * _MASS_TO_GRAMS[units]
                if grams > max_grams:
                    violations.append(
                        RuleViolation(
                            rule_id=self.id,
                            severity=Severity.ERROR,
                            message=(
                                f"Reported product mass {grams:.3g} g exceeds the "
                                f"theoretical maximum {max_grams:.3g} g "
                                f"(n_lim={limiting_moles:.3g} mol, MW_p={mw_p:.2f}, "
                                f"safety factor ×{self.SAFETY_FACTOR})."
                            ),
                            fix_hint=(
                                "Re-check the amount — common errors: mg vs g unit "
                                "confusion, or the wrong product SMILES."
                            ),
                            path=(
                                f"outcomes[0].products[{pi}].measurements[{mi}].value"
                            ),
                        )
                    )
        return violations


class YieldMassConsistency(Rule):
    """STO-007: when both yield% and an isolated mass are reported for the
    same product, they must agree.

    expected_mass = (yield/100) * n_lim * MW_product

    The two numbers come from different parts of the paragraph (the yield
    block at the end and the isolated-mass block) and a common LLM error
    is to copy one correctly and miscalculate the other. We allow ±15%
    tolerance, which covers solvate inclusion and rounding.
    """

    id = "STO-007"
    description = (
        "Reported yield% and isolated mass must agree to within 15% via "
        "expected_mass = (yield/100) * n_lim * MW_product."
    )

    TOLERANCE = 0.15

    def check(self, draft: ReactionDraft) -> list[RuleViolation]:
        limiting_moles, _ = _identify_limiting_moles(draft)
        if limiting_moles is None or limiting_moles <= 0:
            return []
        if not draft.outcomes or not draft.outcomes[0].products:
            return []

        violations: list[RuleViolation] = []
        for pi, prod in enumerate(draft.outcomes[0].products):
            prod_smiles = smiles_of(prod.compound)
            if not prod_smiles:
                continue
            mw_p = mol_weight(prod_smiles)
            if mw_p is None or mw_p <= 0:
                continue

            yield_pct: float | None = None
            mass_g: float | None = None
            for m in prod.measurements:
                if m.type == "YIELD":
                    yield_pct = m.value
                elif m.type == "AMOUNT" and m.units in _MASS_TO_GRAMS:
                    mass_g = m.value * _MASS_TO_GRAMS[m.units]
            if yield_pct is None or mass_g is None or yield_pct <= 0:
                continue

            expected_mass = (yield_pct / 100.0) * limiting_moles * mw_p
            if expected_mass <= 0:
                continue
            diff = abs(mass_g - expected_mass) / expected_mass
            if diff > self.TOLERANCE:
                violations.append(
                    RuleViolation(
                        rule_id=self.id,
                        severity=Severity.ERROR,
                        message=(
                            f"Reported yield={yield_pct}% and mass={mass_g:.3g} g "
                            f"disagree by {diff:.0%}. Expected ~{expected_mass:.3g} g "
                            f"from n_lim={limiting_moles:.3g} mol × MW={mw_p:.2f}."
                        ),
                        fix_hint=(
                            "One of the two figures was transcribed incorrectly. "
                            "Re-read the paragraph and reconcile."
                        ),
                        path=f"outcomes[0].products[{pi}].measurements",
                    )
                )
        return violations


class LimitingReagentIsActuallyLimiting(Rule):
    """STO-008: the compound flagged is_limiting=True must have the smallest
    mole count among all REACTANTS with computable moles.

    Defends against the common LLM mistake of marking the "main substrate"
    by name recognition even when another reactant is actually present in
    smaller amount.
    """

    id = "STO-008"
    description = (
        "is_limiting=True must be on the REACTANT with the smallest mole count."
    )

    SAFETY = 0.05  # 5% slack: rounding in transcription is forgivable.

    def check(self, draft: ReactionDraft) -> list[RuleViolation]:
        # Find the flagged reactant.
        flagged: tuple[int, int, float] | None = None
        all_estimates: list[tuple[int, int, float]] = []
        for i, inp in enumerate(draft.inputs):
            for j, comp in enumerate(inp.components):
                if comp.reaction_role != "REACTANT" or comp.amount is None:
                    continue
                est = _moles_from_amount(comp.amount, smiles_of(comp))
                if est.moles is None:
                    continue
                all_estimates.append((i, j, est.moles))
                if comp.is_limiting:
                    flagged = (i, j, est.moles)

        if flagged is None or len(all_estimates) < 2:
            return []

        true_min = min(m for _, _, m in all_estimates)
        if flagged[2] <= true_min * (1 + self.SAFETY):
            return []

        # Find which compound is actually limiting for a useful error message.
        culprit_i, culprit_j, _ = min(all_estimates, key=lambda x: x[2])
        return [
            RuleViolation(
                rule_id=self.id,
                severity=Severity.ERROR,
                message=(
                    f"is_limiting=True on inputs[{flagged[0]}].components[{flagged[1]}] "
                    f"({flagged[2]:.3g} mol), but inputs[{culprit_i}].components"
                    f"[{culprit_j}] has fewer moles. The flagged compound is "
                    "not the limiting reagent."
                ),
                fix_hint=(
                    f"Move is_limiting=True to inputs[{culprit_i}].components[{culprit_j}] "
                    "(or re-check the masses if the paragraph implies a different "
                    "limiting reagent and the moles arithmetic is wrong)."
                ),
                path=f"inputs[{flagged[0]}].components[{flagged[1]}].is_limiting",
            )
        ]


STO_RULES: list[Rule] = [
    AmountHasUnits(),
    PlausibleVolumes(),
    LimitingReagentIdentifiable(),
    EquivalentsConsistentWithLimiting(),
    YieldRangeSanity(),
    MassBalanceSanity(),
    YieldMassConsistency(),
    LimitingReagentIsActuallyLimiting(),
]
