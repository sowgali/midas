"""Smoke tests: package imports and CLI surface is wired correctly."""

from __future__ import annotations

from typer.testing import CliRunner

from midas import __version__
from midas.cli import app


def test_package_version_is_set() -> None:
    assert __version__ != ""


def test_cli_version_command_prints_version() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_cli_help_lists_subcommands() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for sub in ("ingest", "graph", "init", "version"):
        assert sub in result.stdout


def test_settings_has_expected_defaults() -> None:
    from midas.config import settings

    assert "postgresql+asyncpg" in settings.database_url
    assert settings.cache_dir.parts[-2:] == ("data", "raw")
