"""Static reference blocks embedded in the system prompt.

Lives outside the template module so the strings can be re-imported by
the critic and the naive-LLM adapter without depending on prompt
construction order.
"""

from __future__ import annotations


WORKUP_VERB_REFERENCE = """\
The ordered workup-type vocabulary you may emit:
- ADDITION              — adding a reagent/quench to the reaction mixture
- WASH                  — washing an organic or aqueous layer
- DRY_WITH_MATERIAL     — drying with Na2SO4 / MgSO4 / Celite
- EXTRACTION            — separating phases (often with EtOAc, DCM, ether)
- FILTRATION            — filtering through paper / sinter / Celite
- CONCENTRATION         — evaporating solvent (rotovap, vacuum)
- PH_ADJUST             — adding acid or base to bring pH to a target
- DISSOLUTION           — redissolving a residue
- TEMPERATURE           — cooling or heating step within the workup
- STIRRING              — explicit stirring action within the workup
- WAIT                  — let stand for a period
- DISTILLATION          — purification by distillation
- FLASH_CHROMATOGRAPHY  — silica column chromatography
- CUSTOM                — anything else; describe in the `description` field
"""


EDGE_CASE_HEURISTICS = """\
Edge-case heuristics:
- "equivalents" without a parent mass: emit the equiv amount as-is
  (`units="equiv"`). The limiting reagent's mass + SMILES anchors the
  conversion downstream.
- Heat source disambiguation: prefer `OIL_BATH` when the paragraph mentions
  an oil bath, `WATER_BATH` for water bath, `REFLUX` when refluxing in a
  named solvent, `HEATER`/`UNSPECIFIED` otherwise. For solvent-free / neat
  reactions intentionally above ambient, use `UNSPECIFIED` so the
  solvent-required rule does not fire spuriously.
- Quench is an ADDITION (you add the quench agent), not a WASH. Emit it as
  the first workup with type=ADDITION and the quench agent listed under
  `components`.
- "Diluted with X, washed with Y, dried over Z, concentrated" is FOUR
  workups in order: EXTRACTION (dilution effectively partitions the
  mixture), WASH (with Y), DRY_WITH_MATERIAL (with Z), CONCENTRATION.
- Air- / moisture-sensitive reactions (Grignards, organolithiums, NaH,
  LDA, …) MUST have `conditions.atmosphere` set to "nitrogen" or "argon".
- For ambient temperature ("rt", "room temperature"), set
  `conditions.temperature.control_type="AMBIENT"` and leave setpoint_celsius
  null. For reflux, use control_type="REFLUX".
"""


# Three diverse worked examples covering: a reduction, a Pd-catalysed Suzuki
# coupling, and a Grignard formation. Together ~700 tokens — well inside
# prompt-cache windows.
FEW_SHOT_EXAMPLE = """\
Example 1 — reduction:
    To a stirred solution of benzaldehyde (1.06 g, 10.0 mmol, 1.0 equiv) in
    methanol (10 mL) was added NaBH4 (0.45 g, 12.0 mmol, 1.2 equiv) at 0 °C.
    The mixture was warmed to rt and stirred for 1 h, then quenched with
    saturated NH4Cl (10 mL), extracted with EtOAc (3 × 15 mL), dried over
    Na2SO4, and concentrated to give benzyl alcohol (0.97 g, 90%).

Example draft (key fields shown):
    {
      "inputs": [
        {"name": "limiting_reactant", "components": [{
          "identifiers": [{"type": "NAME", "value": "benzaldehyde"},
                          {"type": "SMILES", "value": "O=Cc1ccccc1"}],
          "amount": {"value": 10.0, "units": "mmol"},
          "reaction_role": "REACTANT", "is_limiting": true}]},
        {"name": "reductant", "components": [{
          "identifiers": [{"type": "NAME", "value": "NaBH4"}],
          "amount": {"value": 1.2, "units": "equiv"},
          "reaction_role": "REAGENT"}]},
        {"name": "solvent", "components": [{
          "identifiers": [{"type": "NAME", "value": "methanol"},
                          {"type": "SMILES", "value": "CO"}],
          "amount": {"value": 10.0, "units": "mL"},
          "reaction_role": "SOLVENT"}]}
      ],
      "conditions": {
        "temperature": {"control_type": "AMBIENT"},
        "stirring": {"type": "MAGNETIC"},
        "duration_minutes": 60
      },
      "workups": [
        {"type": "ADDITION", "description": "Quenched with saturated NH4Cl (10 mL).",
         "components": [{"identifiers": [{"type": "NAME", "value": "NH4Cl"}],
                         "reaction_role": "WORKUP"}], "order": 1},
        {"type": "EXTRACTION", "description": "Extracted with EtOAc (3 × 15 mL).",
         "order": 2},
        {"type": "DRY_WITH_MATERIAL", "description": "Dried over Na2SO4.", "order": 3},
        {"type": "CONCENTRATION", "description": "Concentrated to dryness.", "order": 4}
      ],
      "outcomes": [{"products": [{"compound": {
        "identifiers": [{"type": "NAME", "value": "benzyl alcohol"},
                        {"type": "SMILES", "value": "OCc1ccccc1"}],
        "reaction_role": "PRODUCT"},
        "measurements": [{"type": "YIELD", "value": 90.0, "units": "%"},
                         {"type": "AMOUNT", "value": 0.97, "units": "g"}]}]}],
      "notes": "NaBH4 reduction; standard workup."
    }

Example 2 — Pd-catalysed coupling (truncated to the salient differences):
    "A flask was charged with 4-bromoanisole (1.0 equiv) and phenylboronic
    acid (1.5 equiv), Pd(PPh3)4 (5 mol%), and K2CO3 (3.0 equiv) in
    dioxane/water (4:1) under nitrogen; heated to 90 °C, 16 h; standard
    workup with EtOAc; flash chromatography (silica) gives 4-methoxybiphenyl
    (89%)."
Salient draft fields:
    - inputs: arylhalide is REACTANT+is_limiting; boronic acid is REACTANT
      (NOT REAGENT — it donates a C-C bond); Pd(PPh3)4 is CATALYST; K2CO3 is
      REAGENT; dioxane and water are both SOLVENT.
    - conditions: temperature.control_type=OIL_BATH, setpoint_celsius=90,
      stirring.type=MAGNETIC, atmosphere='nitrogen', duration_minutes=960.
    - workups end with type=FLASH_CHROMATOGRAPHY.
    - product yield 89%.

Example 3 — Grignard formation (illustrates inert-atmosphere requirement):
    "Mg turnings (2.0 equiv) and anhydrous THF were charged under argon;
    bromobenzene (2.0 equiv) in THF was added dropwise at reflux for 30 min;
    after 1 h at reflux, the solution was cooled to 0 °C, benzaldehyde
    (1.0 equiv) added dropwise; warmed to rt, 2 h; quenched with sat. NH4Cl
    at 0 °C; extracted, dried, concentrated, purified to give diphenyl-
    methanol (90%)."
Salient draft fields:
    - inputs: bromobenzene + magnesium + benzaldehyde all REACTANT; THF is
      SOLVENT; benzaldehyde carries is_limiting=True.
    - conditions: control_type=REFLUX (setpoint may be omitted — REFLUX
      implies bp), stirring.type=MAGNETIC, atmosphere='argon'.
    - first workup: ADDITION with NH4Cl listed in components, order=1.
"""
