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


def test_bench_compare_with_synthetic_snapshots(runner: CliRunner, tmp_path: Path) -> None:
    """bench-compare reads two snapshot JSONs and renders a delta report."""
    import json

    baseline = tmp_path / "a.json"
    current = tmp_path / "b.json"
    baseline.write_text(json.dumps([
        {
            "fixture": "aspirin",
            "adapter": "x",
            "success": True,
            "error": None,
            "elapsed_seconds": 1.0,
            "macro_f1": 0.50,
            "field_scores": [
                {"field_name": "a", "precision": 1.0, "recall": 0.5, "f1": 0.5, "support": 1},
            ],
        }
    ]))
    current.write_text(json.dumps([
        {
            "fixture": "aspirin",
            "adapter": "x",
            "success": True,
            "error": None,
            "elapsed_seconds": 1.0,
            "macro_f1": 0.70,
            "field_scores": [
                {"field_name": "a", "precision": 1.0, "recall": 0.7, "f1": 0.7, "support": 1},
            ],
        }
    ]))
    result = runner.invoke(main, ["bench-compare", str(baseline), str(current)])
    assert result.exit_code == 0
    # The CLI must surface the +0.2 delta.
    assert "Δ F1" in result.output or "delta" in result.output.lower()
    assert "0.20" in result.output or "+0.2" in result.output


def test_extract_no_critic_flag_passes_through(
    runner: CliRunner, tmp_path: Path, monkeypatch
) -> None:
    """--no-critic should be recognised by Click without invoking the LLM.

    We capture the flag's effect by patching agent.extract to inspect the
    enable_critic kwarg, then exit cleanly.
    """
    p = tmp_path / "p.txt"
    p.write_text("To 1 g X was added Y at rt.")

    captured: dict[str, object] = {}

    async def fake_extract(paragraph, *, model, max_iters, debug, enable_critic):
        captured["enable_critic"] = enable_critic
        # Return a failure-shaped result so the CLI exits with code 2.
        from eln_structurer.agent import ExtractResult
        return ExtractResult(
            success=False, pbtxt="", json_text="", validation_summary={},
            transcript=[], failure_summary={"reason": "stub"},
        )

    import eln_structurer.cli as cli_mod
    monkeypatch.setattr(cli_mod, "extract", fake_extract)

    result = runner.invoke(main, ["extract", str(p), "--no-critic"])
    assert captured.get("enable_critic") is False
    # exit code 2 from the stubbed failure
    assert result.exit_code == 2


def test_extract_default_enables_critic(
    runner: CliRunner, tmp_path: Path, monkeypatch
) -> None:
    """Without --no-critic, enable_critic defaults to True."""
    p = tmp_path / "p.txt"
    p.write_text("To 1 g X was added Y at rt.")
    captured: dict[str, object] = {}

    async def fake_extract(paragraph, *, model, max_iters, debug, enable_critic):
        captured["enable_critic"] = enable_critic
        from eln_structurer.agent import ExtractResult
        return ExtractResult(
            success=False, pbtxt="", json_text="", validation_summary={},
            transcript=[], failure_summary={"reason": "stub"},
        )

    import eln_structurer.cli as cli_mod
    monkeypatch.setattr(cli_mod, "extract", fake_extract)
    runner.invoke(main, ["extract", str(p)])
    assert captured.get("enable_critic") is True
