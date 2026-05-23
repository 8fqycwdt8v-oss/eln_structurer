"""Snapshot test for ``ValidationReport.as_repair_prompt()``.

The repair prompt is the contract between the harness and the agent —
unintended formatting changes can confuse the agent loop in subtle ways
(e.g. by breaking the agent's regex for parsing rule_ids). This test
pins the format using a known-bad draft.

On legitimate format changes, regenerate the snapshot by running:
    UPDATE_SNAPSHOTS=1 uv run pytest tests/test_repair_prompt_snapshot.py
"""

from __future__ import annotations

import os
from pathlib import Path

from eln_structurer.harness import run_harness
from eln_structurer.schema import ReactionDraft

SNAPSHOT_DIR = Path(__file__).parent / "snapshots"


def _strip_volatile(text: str) -> str:
    """Drop the bits of the repair prompt that aren't structurally meaningful.

    ord-schema validation warnings can vary across versions; the rule
    list itself is what we pin. Strip lines starting with the ord-schema
    warning markers so the snapshot survives upstream cosmetic changes.
    """
    lines = []
    for line in text.splitlines():
        if line.startswith("[WARN  ORD-SCHEMA]"):
            continue
        if "ord_schema" in line.lower() and "warning" in line.lower():
            continue
        lines.append(line)
    return "\n".join(lines)


def test_repair_prompt_format_pinned(aspirin_draft: ReactionDraft) -> None:
    # Break CMP-001 (no REACTANT) — small, deterministic, known.
    for inp in aspirin_draft.inputs:
        for comp in inp.components:
            comp.reaction_role = "REAGENT"

    report = run_harness(aspirin_draft)
    actual = _strip_volatile(report.as_repair_prompt())

    snapshot_path = SNAPSHOT_DIR / "repair_prompt_aspirin_missing_reactant.txt"

    if os.environ.get("UPDATE_SNAPSHOTS") == "1" or not snapshot_path.read_text():
        snapshot_path.write_text(actual + "\n", encoding="utf-8")

    expected = snapshot_path.read_text(encoding="utf-8").rstrip("\n")
    assert actual == expected, (
        "Repair-prompt format changed. If this is intentional, regenerate "
        "with UPDATE_SNAPSHOTS=1 pytest tests/test_repair_prompt_snapshot.py"
    )
