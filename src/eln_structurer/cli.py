"""Command-line entry point for eln_structurer."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel

from eln_structurer.agent import DEFAULT_MODEL, extract


console = Console()


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
    "--model", default=DEFAULT_MODEL, show_default=True,
    help="Claude model identifier.",
)
@click.option(
    "--max-iters", type=int, default=5, show_default=True,
    help="Maximum repair iterations.",
)
@click.option("--debug", is_flag=True, help="Print agent transcript to stderr.")
def extract_cmd(
    source: str, fmt: str, out_path: str | None, model: str, max_iters: int, debug: bool
) -> None:
    """Extract a single reaction paragraph from FILE (or '-' for stdin)."""
    paragraph = _read_input(source).strip()
    if not paragraph:
        raise click.ClickException("Input paragraph is empty.")

    result = asyncio.run(
        extract(paragraph, model=model, max_iters=max_iters, debug=debug)
    )

    if debug and result.transcript:
        console.print(Panel("\n\n".join(result.transcript), title="agent transcript",
                            border_style="dim"), highlight=False)

    if not result.success:
        console.print(
            "[red]Extraction did not converge to a clean ORD reaction.[/red]",
            highlight=False,
        )
        if result.validation_summary:
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
def bench_cmd(fixtures_dir: Path, adapter_names: tuple[str, ...], out_path: Path | None) -> None:
    """Run the benchmark harness across all adapters on the golden fixture set."""
    from eln_structurer.benchmarks.runner import (
        discover_fixtures,
        render_markdown_report,
        run_benchmark_sync,
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


if __name__ == "__main__":  # pragma: no cover
    main()
