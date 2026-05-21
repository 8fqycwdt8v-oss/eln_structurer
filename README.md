# eln_structurer

LLM-driven structuring of unstructured chemical synthesis protocols (free-text
"experimental procedure" paragraphs) into the
[Open Reaction Database](https://docs.open-reaction-database.org/) (ORD) format,
driven by a rule-based validation harness with agentic self-repair.

The tool takes a paragraph like:

> *To a stirred solution of salicylic acid (1.38 g, 10.0 mmol, 1.0 equiv) in
> acetic anhydride (5.0 mL) was added concentrated sulfuric acid (3 drops,
> catalytic) at room temperature. The mixture was warmed to 85 °C and stirred
> for 30 min …*

and emits a fully validated `ord_schema.proto.reaction_pb2.Reaction` (JSON or
pbtxt). Behind the scenes, a Claude agent (via the
[Anthropic Agent SDK](https://code.claude.com/docs/en/agent-sdk/overview))
drafts a structured representation, a pluggable rule pack audits it (ordering,
stoichiometry, structure, completeness), and the agent self-repairs until every
rule passes.

## Architecture

```
paragraph (str)
   │
   ▼
Claude agent  ◄────────────────────────────────────┐
   │  emits ReactionDraft (Pydantic JSON)          │
   ▼                                               │
validate_reaction (in-process MCP tool)            │
   │                                               │
   ├── Pydantic shape check                        │
   ├── proto_bridge: draft → reaction_pb2.Reaction │  errors fed back
   ├── ord_schema.validations.validate_message     │  to the agent
   └── rules.ALL_RULES  (CMP / STR / STO / ORD)    │
   │                                               │
   ▼                                               │
ValidationReport.as_repair_prompt() ────────────-──┘
   │  (when clean)
   ▼
finalize_reaction  →  reaction.pbtxt + reaction.json
```

The repair loop is implicit in the agent's tool-use cycle, bounded by
`max_iters` (default 5).

## Rule pack

| ID | Rule | Severity |
|---|---|---|
| CMP-001 | At least one input has role REACTANT | ERROR |
| CMP-002 | At least one product captured | ERROR |
| CMP-003 | conditions.temperature is set (even if AMBIENT) | ERROR |
| CMP-004 | A reaction duration is captured | WARNING |
| CMP-005 | `notes` captures provenance | WARNING |
| STR-001 | Every SMILES parses via RDKit | ERROR |
| STR-002 | Every compound has NAME or SMILES | ERROR |
| STR-003 | Reactant heavy atoms ≥ first product heavy atoms | WARNING |
| STO-001 | Every Amount has units | ERROR |
| STO-002 | Equivalents are positive | ERROR |
| STO-003 | Volumes within bench-scale range | WARNING |
| STO-004 | Exactly one limiting reagent identifiable | ERROR |
| ORD-001 | Compounds referenced in workups exist as inputs | ERROR |
| ORD-002 | Heated reactions have a SOLVENT input | ERROR |
| ORD-003 | Heated reactions are stirred | WARNING |
| ORD-004 | Quench workups come after the main reaction | ERROR |
| ORD-005 | Workup order is monotonically increasing | WARNING |

## Constraints

- **No external/public APIs except the Anthropic LLM API.** No PubChem, no
  CACTUS, no ChemSpider, no OPSIN web. Local Python (RDKit) and the
  `ord-schema` library only.
- Compound name → SMILES resolution is done entirely by the LLM. RDKit is used
  to *validate* what the LLM produced, not to look anything up.
- Scope is a single reaction paragraph in, a single ORD Reaction out. No PDF
  ingestion, no multi-reaction splitting.

## Quickstart

```bash
uv sync
export ANTHROPIC_API_KEY=...

# extract a paragraph from a file
uv run eln-structurer extract examples/aspirin.txt --format pbtxt --out aspirin.pbtxt

# extract from stdin
cat examples/suzuki_coupling.txt | uv run eln-structurer extract -

# debug mode prints the agent transcript
uv run eln-structurer extract examples/grignard.txt --debug
```

## Tests

```bash
# Unit tests (no API calls)
uv run pytest tests/ -v --ignore=tests/test_agent_e2e.py

# Live end-to-end (consumes API credits)
RUN_LIVE=1 uv run pytest tests/test_agent_e2e.py -v
```

Independent verification of an emitted pbtxt:

```bash
uv run python - <<'PY'
from ord_schema import validations
from google.protobuf import text_format
from ord_schema.proto import reaction_pb2

r = reaction_pb2.Reaction()
text_format.Parse(open("aspirin.pbtxt").read(), r)
print(validations.validate_message(r))
PY
```

## Layout

```
src/eln_structurer/
  agent.py            # Anthropic Agent SDK wiring + main loop
  cli.py              # `eln-structurer extract …`
  harness.py          # ValidationReport + run_harness
  prompts.py          # system prompt with embedded JSON schema
  proto_bridge.py     # ReactionDraft → reaction_pb2.Reaction
  schema.py           # Pydantic mirror of the ORD subset
  rules/
    base.py           # Rule ABC + RuleViolation
    completeness.py   # CMP-*
    ordering.py       # ORD-*
    stoichiometry.py  # STO-*
    structure.py      # STR-*
  tools/
    validate_reaction.py
    validate_smiles.py
    finalize_reaction.py
```

## Status

Phases 1–4 of the plan implemented (skeleton, proto bridge, full rule pack,
agent loop). Phase 5 (benchmark harness against paragraph2actions / OpenChemIE)
is deferred.
