"""Benchmark runner: for each (paragraph, gold) pair, run every adapter."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path

from eln_structurer.benchmarks.adapters import REGISTRY, AdapterResult
from eln_structurer.benchmarks.adapters.base import (
    Adapter,
    AdapterError,
    AdapterUnavailable,
)
from eln_structurer.benchmarks.canonical import CanonicalReaction, load_gold
from eln_structurer.benchmarks.scoring import FieldScore, macro_f1, score_against_gold

log = logging.getLogger(__name__)


@dataclass
class FixtureCase:
    name: str
    paragraph: str
    gold: CanonicalReaction


@dataclass
class CaseRun:
    fixture: str
    adapter: str
    success: bool
    error: str | None
    elapsed_seconds: float
    macro_f1: float | None
    field_scores: list[FieldScore]


def _failure_run(
    fixture: str, adapter: str, error: str, elapsed: float = 0.0
) -> CaseRun:
    return CaseRun(
        fixture=fixture,
        adapter=adapter,
        success=False,
        error=error,
        elapsed_seconds=elapsed,
        macro_f1=None,
        field_scores=[],
    )


def discover_fixtures(
    paragraphs_dir: Path, gold_dir: Path
) -> list[FixtureCase]:
    cases: list[FixtureCase] = []
    for p_path in sorted(paragraphs_dir.glob("*.txt")):
        gold_path = gold_dir / f"{p_path.stem}.gold.json"
        if not gold_path.exists():
            continue
        cases.append(
            FixtureCase(
                name=p_path.stem,
                paragraph=p_path.read_text(encoding="utf-8").strip(),
                gold=load_gold(gold_path),
            )
        )
    return cases


async def run_adapter(adapter: Adapter, paragraph: str) -> AdapterResult:
    start = time.monotonic()
    try:
        prediction = await adapter.extract(paragraph)
        return AdapterResult(
            adapter_name=adapter.name,
            success=True,
            prediction=prediction,
            error=None,
            elapsed_seconds=time.monotonic() - start,
        )
    except AdapterUnavailable as exc:
        return AdapterResult(
            adapter_name=adapter.name,
            success=False,
            prediction=None,
            error=f"UNAVAILABLE: {exc}",
            elapsed_seconds=time.monotonic() - start,
        )
    except AdapterError as exc:
        return AdapterResult(
            adapter_name=adapter.name,
            success=False,
            prediction=None,
            error=f"FAILED: {exc}",
            elapsed_seconds=time.monotonic() - start,
        )
    except Exception as exc:  # noqa: BLE001 — log and continue
        log.exception("Unexpected error in adapter %r", adapter.name)
        return AdapterResult(
            adapter_name=adapter.name,
            success=False,
            prediction=None,
            error=f"ERROR: {exc!r}",
            elapsed_seconds=time.monotonic() - start,
        )


async def run_benchmark(
    cases: list[FixtureCase],
    adapter_names: list[str],
) -> list[CaseRun]:
    """Run every (case, adapter) pair sequentially and return CaseRun records."""
    runs: list[CaseRun] = []
    for case in cases:
        for adapter_name in adapter_names:
            factory = REGISTRY.get(adapter_name)
            if factory is None:
                runs.append(
                    _failure_run(case.name, adapter_name, f"UNKNOWN adapter: {adapter_name}")
                )
                continue
            adapter = factory()
            if not await adapter.is_available():
                runs.append(
                    _failure_run(case.name, adapter_name, "UNAVAILABLE (precheck)")
                )
                continue
            result = await run_adapter(adapter, case.paragraph)
            if result.success and result.prediction is not None:
                field_scores = score_against_gold(result.prediction, case.gold)
                runs.append(
                    CaseRun(
                        fixture=case.name,
                        adapter=adapter_name,
                        success=True,
                        error=None,
                        elapsed_seconds=result.elapsed_seconds,
                        macro_f1=macro_f1(field_scores),
                        field_scores=field_scores,
                    )
                )
            else:
                runs.append(
                    _failure_run(
                        case.name, adapter_name, result.error or "unknown",
                        elapsed=result.elapsed_seconds,
                    )
                )
    return runs


def render_markdown_report(runs: list[CaseRun]) -> str:
    if not runs:
        return "_No benchmark runs to report._"
    fixtures = sorted({r.fixture for r in runs})
    adapters = sorted({r.adapter for r in runs})

    lines = ["# Benchmark report", "", "## Macro-F1 by adapter × fixture", ""]
    header = "| adapter | " + " | ".join(fixtures) + " | mean |"
    sep = "|" + "---|" * (len(fixtures) + 2)
    lines.append(header)
    lines.append(sep)
    for adapter in adapters:
        cells = []
        scores: list[float] = []
        for fixture in fixtures:
            run = next(
                (r for r in runs if r.adapter == adapter and r.fixture == fixture),
                None,
            )
            if run is None or run.macro_f1 is None:
                cells.append("—")
            else:
                cells.append(f"{run.macro_f1:.2f}")
                scores.append(run.macro_f1)
        mean = f"**{sum(scores) / len(scores):.2f}**" if scores else "—"
        lines.append(f"| {adapter} | " + " | ".join(cells) + f" | {mean} |")

    lines.extend(["", "## Per-field F1 (averaged across fixtures)", ""])
    field_names: list[str] = []
    seen: set[str] = set()
    for r in runs:
        for fs in r.field_scores:
            if fs.field_name not in seen:
                seen.add(fs.field_name)
                field_names.append(fs.field_name)

    lines.append("| adapter | " + " | ".join(field_names) + " |")
    lines.append("|" + "---|" * (len(field_names) + 1))
    for adapter in adapters:
        adapter_runs = [r for r in runs if r.adapter == adapter and r.success]
        if not adapter_runs:
            lines.append(f"| {adapter} | " + " | ".join("—" for _ in field_names) + " |")
            continue
        cells = []
        for fname in field_names:
            f1s = [
                next((s.f1 for s in r.field_scores if s.field_name == fname), None)
                for r in adapter_runs
            ]
            f1s = [x for x in f1s if x is not None]
            cells.append(f"{sum(f1s) / len(f1s):.2f}" if f1s else "—")
        lines.append(f"| {adapter} | " + " | ".join(cells) + " |")

    lines.extend(["", "## Failures", ""])
    failed = [r for r in runs if not r.success]
    if not failed:
        lines.append("_None._")
    else:
        for r in failed:
            lines.append(f"- **{r.adapter}** on `{r.fixture}`: {r.error}")
    return "\n".join(lines) + "\n"


def run_benchmark_sync(
    cases: list[FixtureCase], adapter_names: list[str]
) -> list[CaseRun]:
    return asyncio.run(run_benchmark(cases, adapter_names))
