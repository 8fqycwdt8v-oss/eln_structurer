"""Structure rules (STR-*) — checks on SMILES validity and atom-count plausibility.

All checks here are local: RDKit only, no network calls.
"""

from __future__ import annotations

from rdkit import Chem
from rdkit import RDLogger

from eln_structurer.rules.base import Rule, RuleViolation, Severity
from eln_structurer.schema import CompoundModel, ReactionDraft

# Silence RDKit's verbose parse-error logging; we surface our own messages.
RDLogger.DisableLog("rdApp.*")


def _smiles_of(comp: CompoundModel) -> str | None:
    for ident in comp.identifiers:
        if ident.type == "SMILES":
            return ident.value
    return None


def _name_or_smiles(comp: CompoundModel) -> bool:
    return any(i.type in {"NAME", "SMILES", "IUPAC_NAME"} for i in comp.identifiers)


def _heavy_atoms(smiles: str) -> int | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return mol.GetNumHeavyAtoms()


class SmilesParses(Rule):
    id = "STR-001"
    description = "Every SMILES identifier must parse via RDKit."

    def check(self, draft: ReactionDraft) -> list[RuleViolation]:
        violations: list[RuleViolation] = []
        for i, inp in enumerate(draft.inputs):
            for j, comp in enumerate(inp.components):
                for k, ident in enumerate(comp.identifiers):
                    if ident.type == "SMILES" and Chem.MolFromSmiles(ident.value) is None:
                        violations.append(
                            RuleViolation(
                                rule_id=self.id,
                                severity=Severity.ERROR,
                                message=(
                                    f"SMILES {ident.value!r} fails RDKit parse."
                                ),
                                fix_hint=(
                                    "Re-derive the SMILES, or remove the SMILES "
                                    "identifier and keep only NAME."
                                ),
                                path=f"inputs[{i}].components[{j}].identifiers[{k}]",
                            )
                        )
        for oi, outcome in enumerate(draft.outcomes):
            for pi, prod in enumerate(outcome.products):
                for k, ident in enumerate(prod.compound.identifiers):
                    if ident.type == "SMILES" and Chem.MolFromSmiles(ident.value) is None:
                        violations.append(
                            RuleViolation(
                                rule_id=self.id,
                                severity=Severity.ERROR,
                                message=f"Product SMILES {ident.value!r} fails RDKit parse.",
                                fix_hint=(
                                    "Re-derive the product SMILES, or keep only the NAME."
                                ),
                                path=(
                                    f"outcomes[{oi}].products[{pi}]"
                                    f".compound.identifiers[{k}]"
                                ),
                            )
                        )
        return violations


class NameOrSmilesPresent(Rule):
    id = "STR-002"
    description = "Every compound must have at least one NAME or SMILES identifier."

    def check(self, draft: ReactionDraft) -> list[RuleViolation]:
        violations: list[RuleViolation] = []
        for i, inp in enumerate(draft.inputs):
            for j, comp in enumerate(inp.components):
                if not _name_or_smiles(comp):
                    violations.append(
                        RuleViolation(
                            rule_id=self.id,
                            severity=Severity.ERROR,
                            message="Compound has no NAME or SMILES identifier.",
                            fix_hint=(
                                "Add at least one identifier with type='NAME' or "
                                "type='SMILES'."
                            ),
                            path=f"inputs[{i}].components[{j}].identifiers",
                        )
                    )
        return violations


class AtomBalanceSanity(Rule):
    id = "STR-003"
    description = (
        "Heavy-atom count of reactants should be >= heavy-atom count of first product."
    )

    def check(self, draft: ReactionDraft) -> list[RuleViolation]:
        reactant_heavy = 0
        any_reactant_parsed = False
        for inp in draft.inputs:
            for comp in inp.components:
                if comp.reaction_role != "REACTANT":
                    continue
                smi = _smiles_of(comp)
                if not smi:
                    continue
                ha = _heavy_atoms(smi)
                if ha is not None:
                    reactant_heavy += ha
                    any_reactant_parsed = True

        if not any_reactant_parsed or not draft.outcomes or not draft.outcomes[0].products:
            return []

        first_product = draft.outcomes[0].products[0]
        prod_smi = _smiles_of(first_product.compound)
        if not prod_smi:
            return []
        prod_heavy = _heavy_atoms(prod_smi)
        if prod_heavy is None:
            return []

        if reactant_heavy < prod_heavy:
            return [
                RuleViolation(
                    rule_id=self.id,
                    severity=Severity.WARNING,
                    message=(
                        f"Sum of REACTANT heavy atoms ({reactant_heavy}) is less than "
                        f"heavy atoms in the first product ({prod_heavy}). "
                        "The product is larger than its inputs."
                    ),
                    fix_hint=(
                        "Check whether you marked a coupling partner as REAGENT "
                        "instead of REACTANT, or whether the product SMILES is wrong."
                    ),
                    path="outcomes[0].products[0]",
                )
            ]
        return []


STR_RULES: list[Rule] = [SmilesParses(), NameOrSmilesPresent(), AtomBalanceSanity()]
