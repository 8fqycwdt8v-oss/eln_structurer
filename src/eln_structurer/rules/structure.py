"""Structure rules (STR-*) — checks on SMILES validity and atom-count plausibility.

All checks here are local: RDKit only, no network calls.
"""

from __future__ import annotations

from eln_structurer.rules.base import Rule, RuleViolation, Severity
from eln_structurer.rules.compound_utils import (
    has_name_or_smiles,
    heavy_atoms,
    parse_mol,
    smiles_of,
)
from eln_structurer.schema import ReactionDraft


class SmilesParses(Rule):
    id = "STR-001"
    description = "Every SMILES identifier must parse via RDKit."

    def check(self, draft: ReactionDraft) -> list[RuleViolation]:
        violations: list[RuleViolation] = []
        for i, inp in enumerate(draft.inputs):
            for j, comp in enumerate(inp.components):
                for k, ident in enumerate(comp.identifiers):
                    if ident.type == "SMILES" and parse_mol(ident.value) is None:
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
                    if ident.type == "SMILES" and parse_mol(ident.value) is None:
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
                if not has_name_or_smiles(comp):
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
        "First product heavy-atom count must not exceed the sum of REACTANT "
        "heavy atoms (matter is conserved; products can be smaller than the "
        "sum of inputs from elimination/fragmentation, but never larger)."
    )

    # REACTANTs are the main substrates; REAGENTs frequently donate atoms
    # too (methylating agents, electrophiles, etc.). SOLVENTs typically
    # don't contribute heavy atoms to the product — hydrolysis is a
    # special case the LLM is expected to model by promoting water to
    # REACTANT role.
    _ATOM_DONOR_ROLES = {"REACTANT", "REAGENT"}

    def check(self, draft: ReactionDraft) -> list[RuleViolation]:
        donor_heavy = 0
        any_parsed = False
        for inp in draft.inputs:
            for comp in inp.components:
                if comp.reaction_role not in self._ATOM_DONOR_ROLES:
                    continue
                smi = smiles_of(comp)
                if not smi:
                    continue
                ha = heavy_atoms(smi)
                if ha is not None:
                    donor_heavy += ha
                    any_parsed = True

        if not any_parsed or not draft.outcomes or not draft.outcomes[0].products:
            return []

        first_product = draft.outcomes[0].products[0]
        prod_smi = smiles_of(first_product.compound)
        if not prod_smi:
            return []
        prod_heavy = heavy_atoms(prod_smi)
        if prod_heavy is None:
            return []

        # Chemically: product cannot have more heavy atoms than the sum of
        # everything that supplied them. Fragmentation/elimination products
        # are smaller and pass silently.
        if prod_heavy > donor_heavy:
            return [
                RuleViolation(
                    rule_id=self.id,
                    severity=Severity.ERROR,
                    message=(
                        f"First product has {prod_heavy} heavy atoms but "
                        f"REACTANT+REAGENT inputs only supply {donor_heavy}. "
                        "Mass cannot be created."
                    ),
                    fix_hint=(
                        "Either an atom-donating compound was misclassified as "
                        "SOLVENT/CATALYST, or the product SMILES is wrong. "
                        "Hydrolysis-type reactions where water contributes atoms "
                        "should mark water with reaction_role='REACTANT'."
                    ),
                    path="outcomes[0].products[0]",
                )
            ]
        return []


STR_RULES: list[Rule] = [SmilesParses(), NameOrSmilesPresent(), AtomBalanceSanity()]
