"""CLI surface tests.

We don't run real migrations, real network calls, or real graph builds
here — those are exercised by the storage / sources / pipeline / graph
slices' own tests. This file guards the CLI plumbing: command parsing,
help text, option wiring, and that each command's async coroutine is
called with the parsed arguments.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from typer.testing import CliRunner

from midas import __version__
from midas.cli import app
from midas.pipeline import IngestStats


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


# ---------- version & help ----------


def test_version_prints_installed_version(runner: CliRunner) -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_top_level_help_lists_subcommands(runner: CliRunner) -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for sub in ("ingest", "graph", "init", "version"):
        assert sub in result.stdout


def test_ingest_help_lists_sec(runner: CliRunner) -> None:
    result = runner.invoke(app, ["ingest", "--help"])
    assert result.exit_code == 0
    assert "sec" in result.stdout


def test_graph_help_lists_render(runner: CliRunner) -> None:
    result = runner.invoke(app, ["graph", "--help"])
    assert result.exit_code == 0
    assert "render" in result.stdout


# ---------- init ----------


def test_init_runs_migrations_then_seeds(
    runner: CliRunner, monkeypatch: Any, tmp_path: Path
) -> None:
    """`midas init` calls alembic upgrade then seeds the registry."""
    upgrade_called: list[Path] = []
    seed_called: list[bool] = []

    monkeypatch.setattr(
        "midas.cli._run_alembic_upgrade",
        lambda ini: upgrade_called.append(ini),
    )

    async def fake_seed() -> None:
        seed_called.append(True)

    monkeypatch.setattr("midas.cli._seed_registry", fake_seed)

    fake_ini = tmp_path / "alembic.ini"
    fake_ini.write_text("[alembic]\n")
    result = runner.invoke(app, ["init", "--alembic-ini", str(fake_ini)])

    assert result.exit_code == 0, result.stdout
    assert upgrade_called == [fake_ini]
    assert seed_called == [True]


def test_init_seed_only_skips_migrations(runner: CliRunner, monkeypatch: Any) -> None:
    upgrade_called: list[Path] = []
    seed_called: list[bool] = []

    monkeypatch.setattr(
        "midas.cli._run_alembic_upgrade",
        lambda ini: upgrade_called.append(ini),
    )

    async def fake_seed() -> None:
        seed_called.append(True)

    monkeypatch.setattr("midas.cli._seed_registry", fake_seed)

    result = runner.invoke(app, ["init", "--seed-only"])
    assert result.exit_code == 0, result.stdout
    assert upgrade_called == []
    assert seed_called == [True]


def test_init_missing_alembic_ini_fails(runner: CliRunner, tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist.ini"
    result = runner.invoke(app, ["init", "--alembic-ini", str(missing)])
    assert result.exit_code != 0
    assert "not found" in (result.stdout + result.stderr).lower()


# ---------- ingest sec ----------


def test_ingest_sec_parses_args_and_invokes_pipeline(runner: CliRunner, monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}

    async def fake_ingest(
        ticker: str, since: date | None, forms: list[str], extractor: Any
    ) -> None:
        captured["ticker"] = ticker
        captured["since"] = since
        captured["forms"] = forms
        captured["extractor_kind"] = type(extractor).__name__

    monkeypatch.setattr("midas.cli._ingest_sec", fake_ingest)

    result = runner.invoke(
        app,
        [
            "ingest",
            "sec",
            "--ticker",
            "MSFT",
            "--since",
            "2024-01-01",
            "--form",
            "10-K",
            "--form",
            "8-K",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert captured["ticker"] == "MSFT"
    assert captured["since"] == date(2024, 1, 1)
    assert captured["forms"] == ["10-K", "8-K"]
    assert captured["extractor_kind"] == "RegexExtractor"  # default


def test_ingest_sec_with_claude_extractor(runner: CliRunner, monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}

    async def fake_ingest(
        ticker: str, since: date | None, forms: list[str], extractor: Any
    ) -> None:
        captured["extractor_kind"] = type(extractor).__name__

    monkeypatch.setattr("midas.cli._ingest_sec", fake_ingest)

    result = runner.invoke(
        app,
        ["ingest", "sec", "--ticker", "NVDA", "--extractor", "claude"],
    )
    assert result.exit_code == 0, result.stdout
    assert captured["extractor_kind"] == "ClaudeExtractor"


def test_ingest_sec_unknown_extractor_rejected(runner: CliRunner) -> None:
    result = runner.invoke(
        app,
        ["ingest", "sec", "--ticker", "MSFT", "--extractor", "magic"],
    )
    assert result.exit_code != 0


def test_print_stats_renders_counters(monkeypatch: Any, capsys: Any) -> None:
    """Smoke-check the human-friendly stats line."""
    from midas.cli import _print_stats

    stats = IngestStats(
        documents_seen=2,
        sources_added=1,
        sources_skipped_duplicate=1,
        deals_added=3,
        deals_skipped_unknown_party=1,
        evidence_spans_added=3,
        errors=["boom"],
    )
    _print_stats("test", stats)
    out = capsys.readouterr().out + capsys.readouterr().err
    assert "documents=2" in out
    assert "deals(+/merged/skip)=3/0/1" in out


# ---------- graph render ----------


def test_graph_render_parses_args(runner: CliRunner, monkeypatch: Any, tmp_path: Path) -> None:
    captured: dict[str, Any] = {}

    async def fake_render(sector: str | None, as_of: date | None, output: Path, title: str) -> None:
        captured["sector"] = sector
        captured["as_of"] = as_of
        captured["output"] = output
        captured["title"] = title

    monkeypatch.setattr("midas.cli._render_graph", fake_render)

    output = tmp_path / "g.html"
    result = runner.invoke(
        app,
        [
            "graph",
            "render",
            "--sector",
            "ai",
            "--as-of",
            "2025-09-01",
            "--output",
            str(output),
            "--title",
            "AI cash flow",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert captured["sector"] == "ai"
    assert captured["as_of"] == date(2025, 9, 1)
    assert captured["output"] == output
    assert captured["title"] == "AI cash flow"


# ---------- reconcile ----------


def test_reconcile_dry_run_reports_plan(runner: CliRunner, monkeypatch: Any) -> None:
    """`midas reconcile --dry-run` prints the plan and writes nothing."""
    captured: dict[str, Any] = {}

    async def fake_reconcile(*, dry_run: bool, verbose: bool) -> None:
        captured["dry_run"] = dry_run
        captured["verbose"] = verbose

    monkeypatch.setattr("midas.cli._reconcile", fake_reconcile)

    result = runner.invoke(app, ["reconcile", "--dry-run"])
    assert result.exit_code == 0, result.stdout
    assert captured["dry_run"] is True
    assert captured["verbose"] is False


def test_reconcile_apply_no_dry_run(runner: CliRunner, monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}

    async def fake_reconcile(*, dry_run: bool, verbose: bool) -> None:
        captured["dry_run"] = dry_run
        captured["verbose"] = verbose

    monkeypatch.setattr("midas.cli._reconcile", fake_reconcile)

    result = runner.invoke(app, ["reconcile", "-v"])
    assert result.exit_code == 0
    assert captured["dry_run"] is False
    assert captured["verbose"] is True


def test_graph_render_defaults(runner: CliRunner, monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}

    async def fake_render(sector: str | None, as_of: date | None, output: Path, title: str) -> None:
        captured["sector"] = sector
        captured["as_of"] = as_of
        captured["output"] = output

    monkeypatch.setattr("midas.cli._render_graph", fake_render)

    result = runner.invoke(app, ["graph", "render"])
    assert result.exit_code == 0, result.stdout
    assert captured["sector"] is None
    assert captured["as_of"] is None
    assert captured["output"] == Path("graph.html")


# Quiet unused-import lint when individual tests are run.
_ = AsyncMock
