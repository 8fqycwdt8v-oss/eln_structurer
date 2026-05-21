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


if __name__ == "__main__":  # pragma: no cover
    main()
