"""Command-line entry point for eln_structurer."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel

from eln_structurer.agent import DEFAULT_MODEL, HIGH_QUALITY_MODEL, extract


console = Console()


def _format_failure_summary(summary: dict) -> str:
    """Render a structured failure summary as Rich-friendly markdown."""
    lines = [
        f"Iterations: {summary.get('iterations', 0)}",
    ]
    history = summary.get("rule_history") or {}
    if history:
        lines.append("Repeat-violation counts:")
        for rule_id, count in sorted(history.items(), key=lambda kv: -kv[1]):
            lines.append(f"  {rule_id:8s} fired {count}x")
    last = summary.get("last_validation_summary") or {}
    errors = (last.get("errors") or [])[:5]
    if errors:
        lines.append("Last validation errors:")
        for e in errors:
            lines.append(f"  [{e.get('rule_id')}] {e.get('message')}")
            if e.get("fix_hint"):
                lines.append(f"      → {e['fix_hint']}")
    ord_errors = (last.get("ord_validation_errors") or [])[:5]
    if ord_errors:
        lines.append("ord-schema errors:")
        for e in ord_errors:
            lines.append(f"  - {e}")
    explanation = summary.get("explanation")
    if explanation:
        lines.append("")
        lines.append(explanation)
    return "\n".join(lines)


def _read_input(source: str) -> str:
    if source == "-":
        return sys.stdin.read()
    path = Path(source)
    if not path.exists():
        raise click.ClickException(f"File not found: {source}")
    return path.read_text(encoding="utf-8")


@click.group()
def main() -> None:
    """eln_structurer — LLM-driven structuring of chemical synthesis protocols."""


@main.command("extract")
@click.argument("source", type=str)
@click.option(
    "--format", "fmt", type=click.Choice(["json", "pbtxt"]), default="json",
    show_default=True, help="Output format.",
)
@click.option(
    "--out", "out_path", type=click.Path(dir_okay=False, writable=True), default=None,
    help="Write to this file instead of stdout.",
)
@click.option(
    "--model", default=None,
    help=(
        f"Claude model identifier. Default: {DEFAULT_MODEL}. Use --quality "
        f"as a shortcut for {HIGH_QUALITY_MODEL}."
    ),
)
@click.option(
    "--quality", is_flag=True,
    help=f"Shortcut: --model={HIGH_QUALITY_MODEL} for maximum extraction quality.",
)
@click.option(
    "--max-iters", type=int, default=5, show_default=True,
    help="Maximum repair iterations.",
)
@click.option(
    "--no-critic",
    is_flag=True,
    help="Skip the critic-reviser subagent pass (faster, but less robust).",
)
@click.option("--debug", is_flag=True, help="Print agent transcript to stderr.")
def extract_cmd(
    source: str,
    fmt: str,
    out_path: str | None,
    model: str | None,
    quality: bool,
    max_iters: int,
    no_critic: bool,
    debug: bool,
) -> None:
    """Extract a single reaction paragraph from FILE (or '-' for stdin)."""
    paragraph = _read_input(source).strip()
    if not paragraph:
        raise click.ClickException("Input paragraph is empty.")

    if quality and model is not None and model != HIGH_QUALITY_MODEL:
        raise click.ClickException(
            f"--quality conflicts with --model={model}. Pick one."
        )
    chosen_model = HIGH_QUALITY_MODEL if quality else (model or DEFAULT_MODEL)

    result = asyncio.run(
        extract(
            paragraph,
            model=chosen_model,
            max_iters=max_iters,
            debug=debug,
            enable_critic=not no_critic,
        )
    )

    if debug and result.transcript:
        console.print(Panel("\n\n".join(result.transcript), title="agent transcript",
                            border_style="dim"), highlight=False)

    if not result.success:
        console.print(
            "[red]Extraction did not converge to a clean ORD reaction.[/red]",
            highlight=False,
        )
        if result.failure_summary:
            console.print(
                Panel(
                    _format_failure_summary(result.failure_summary),
                    title="failure summary",
                    border_style="red",
                ),
                highlight=False,
            )
        elif result.validation_summary:
            console.print(result.validation_summary)
        sys.exit(2)

    output = result.pbtxt if fmt == "pbtxt" else result.json_text
    if out_path:
        Path(out_path).write_text(output, encoding="utf-8")
        console.print(f"[green]Wrote {fmt} output to {out_path}[/green]")
    else:
        click.echo(output)


@main.command("bench")
@click.option(
    "--fixtures-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("tests/fixtures"),
    show_default=True,
    help="Directory containing paragraphs/*.txt and golden/*.gold.json",
)
@click.option(
    "--adapter",
    "adapter_names",
    multiple=True,
    default=("eln_structurer", "naive_llm", "paragraph2actions", "openchemie"),
    show_default=True,
    help="Adapter(s) to run. Repeat for multiple.",
)
@click.option(
    "--out",
    "out_path",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    default=None,
    help="Write the markdown report to this file (default: stdout).",
)
@click.option(
    "--snapshot",
    "snapshot_path",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    default=None,
    help="Also save raw CaseRun records as JSON for later `bench-compare`.",
)
def bench_cmd(
    fixtures_dir: Path,
    adapter_names: tuple[str, ...],
    out_path: Path | None,
    snapshot_path: Path | None,
) -> None:
    """Run the benchmark harness across all adapters on the golden fixture set."""
    from eln_structurer.benchmarks.runner import (
        discover_fixtures,
        render_markdown_report,
        run_benchmark_sync,
        runs_to_json,
    )

    paragraphs_dir = fixtures_dir / "paragraphs"
    gold_dir = fixtures_dir / "golden"
    if not paragraphs_dir.is_dir() or not gold_dir.is_dir():
        raise click.ClickException(
            f"Expected {paragraphs_dir} and {gold_dir} to exist with fixture files."
        )
    cases = discover_fixtures(paragraphs_dir, gold_dir)
    if not cases:
        raise click.ClickException(
            f"No fixtures found in {paragraphs_dir} (need matching *.gold.json in {gold_dir})."
        )
    console.print(
        f"[cyan]Running {len(cases)} fixtures × {len(adapter_names)} adapters[/cyan]"
    )
    runs = run_benchmark_sync(cases, list(adapter_names))
    report = render_markdown_report(runs)
    if out_path:
        out_path.write_text(report, encoding="utf-8")
        console.print(f"[green]Wrote report to {out_path}[/green]")
    else:
        click.echo(report)
    if snapshot_path:
        snapshot_path.write_text(runs_to_json(runs), encoding="utf-8")
        console.print(f"[green]Wrote snapshot JSON to {snapshot_path}[/green]")


@main.command("bench-compare")
@click.argument(
    "baseline",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.argument(
    "current",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--out",
    "out_path",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    default=None,
    help="Write the comparison report here (default: stdout).",
)
def bench_compare_cmd(baseline: Path, current: Path, out_path: Path | None) -> None:
    """Compare two snapshot JSONs produced by `bench --snapshot`."""
    from eln_structurer.benchmarks.compare import (
        load_run_json,
        paired_comparison,
        render_comparison_report,
    )

    baseline_runs = load_run_json(baseline)
    current_runs = load_run_json(current)
    deltas, summary = paired_comparison(baseline_runs, current_runs)
    report = render_comparison_report(
        deltas, summary,
        baseline_label=baseline.stem,
        current_label=current.stem,
    )
    if out_path:
        out_path.write_text(report, encoding="utf-8")
        console.print(f"[green]Wrote comparison to {out_path}[/green]")
    else:
        click.echo(report)


if __name__ == "__main__":  # pragma: no cover
    main()
