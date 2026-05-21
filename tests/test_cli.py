"""CLI smoke tests.

These tests cover the file-IO and error-handling boundaries of the Click
commands. They never spin up the LLM agent — that path is gated behind the
live E2E test.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from eln_structurer.cli import main


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def test_extract_help(runner: CliRunner) -> None:
    result = runner.invoke(main, ["extract", "--help"])
    assert result.exit_code == 0
    assert "--format" in result.output
    assert "--max-iters" in result.output


def test_bench_help(runner: CliRunner) -> None:
    result = runner.invoke(main, ["bench", "--help"])
    assert result.exit_code == 0
    assert "--fixtures-dir" in result.output
    assert "--adapter" in result.output


def test_extract_missing_file(runner: CliRunner) -> None:
    result = runner.invoke(main, ["extract", "/definitely/not/a/real/path.txt"])
    assert result.exit_code != 0
    assert "not found" in result.output.lower()


def test_extract_empty_file_errors(runner: CliRunner, tmp_path: Path) -> None:
    empty = tmp_path / "empty.txt"
    empty.write_text("")
    result = runner.invoke(main, ["extract", str(empty)])
    assert result.exit_code != 0
    assert "empty" in result.output.lower()


def test_bench_unknown_fixtures_dir(runner: CliRunner) -> None:
    result = runner.invoke(main, ["bench", "--fixtures-dir", "/no/such/dir"])
    assert result.exit_code != 0


def test_bench_with_real_fixtures_degrades_gracefully(
    runner: CliRunner, tmp_path: Path, monkeypatch
) -> None:
    """Without an ANTHROPIC_API_KEY, every adapter precheck fails, but the
    runner still produces a valid markdown report."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    out = tmp_path / "report.md"
    result = runner.invoke(
        main,
        [
            "bench",
            "--fixtures-dir",
            "tests/fixtures",
            "--adapter",
            "eln_structurer",
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0
    text = out.read_text()
    assert "Benchmark report" in text
    assert "UNAVAILABLE" in text
