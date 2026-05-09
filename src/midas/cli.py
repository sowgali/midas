"""Entry point for the ``midas`` CLI.

Composes the storage, registry, sources, extractors, pipeline, and graph
layers behind a small typer-based command surface. The CLI assumes it's
run from the repo root (or a checkout) so that ``alembic.ini`` and the
``alembic/`` directory are reachable for migrations; pass
``--alembic-ini`` to override.

Commands:

* ``midas version``            — print version
* ``midas init``               — run migrations + seed entity registry
* ``midas ingest sec``         — fetch SEC filings + extract + persist
* ``midas graph render``       — build the graph + write interactive HTML
"""

from __future__ import annotations

import asyncio
from datetime import date
from pathlib import Path
from typing import Annotated

import structlog
import typer
from sqlalchemy.ext.asyncio import AsyncSession

from midas import __version__
from midas.extractors.base import Extractor
from midas.graph.builder import build_graph
from midas.graph.viz import render_pyvis
from midas.pipeline import IngestStats, ingest_sec_filings_for_ticker
from midas.registry import load_seed_registry
from midas.sources.http_client import HttpClient
from midas.storage.db import make_engine

log = structlog.get_logger(__name__)

DEFAULT_ALEMBIC_INI = Path("alembic.ini")

app = typer.Typer(
    name="midas",
    help="Map cash flow between companies in a sector.",
    no_args_is_help=True,
    add_completion=False,
)

ingest_app = typer.Typer(help="Fetch raw documents from sources.", no_args_is_help=True)
graph_app = typer.Typer(help="Build and render the cash-flow graph.", no_args_is_help=True)
app.add_typer(ingest_app, name="ingest")
app.add_typer(graph_app, name="graph")


# ---------- Helpers ----------


def _build_extractor(name: str) -> Extractor:
    """Lazy-build an extractor; deferred imports keep ``midas version`` fast."""
    if name == "regex":
        from midas.extractors.regex import RegexExtractor

        return RegexExtractor()
    if name == "claude":
        from midas.extractors.claude import ClaudeExtractor

        return ClaudeExtractor()
    raise typer.BadParameter(f"unknown extractor: {name!r} (expected 'regex' or 'claude')")


def _run_alembic_upgrade(ini_path: Path) -> None:
    if not ini_path.exists():
        raise typer.BadParameter(
            f"{ini_path} not found — run from the repo root or pass --alembic-ini.",
        )
    # Imported lazily so the CLI starts fast and so ``midas version`` works
    # in a checkout that doesn't have alembic installed in some odd path.
    from alembic import command
    from alembic.config import Config

    config = Config(str(ini_path))
    command.upgrade(config, "head")


def _print_stats(label: str, stats: IngestStats) -> None:
    typer.echo(
        f"{label}: documents={stats.documents_seen} "
        f"sources(+/dup)={stats.sources_added}/{stats.sources_skipped_duplicate} "
        f"deals(+/skip)={stats.deals_added}/{stats.deals_skipped_unknown_party} "
        f"evidence={stats.evidence_spans_added} "
        f"errors={len(stats.errors)}",
    )
    for err in stats.errors:
        typer.echo(f"  ! {err}", err=True)


# ---------- Top-level commands ----------


@app.command()
def version() -> None:
    """Print the installed midas version."""
    typer.echo(__version__)


@app.command()
def init(
    seed_only: Annotated[
        bool,
        typer.Option("--seed-only", help="Skip migrations; only re-seed the entity registry."),
    ] = False,
    alembic_ini: Annotated[
        Path,
        typer.Option("--alembic-ini", help="Path to alembic.ini.", show_default=True),
    ] = DEFAULT_ALEMBIC_INI,
) -> None:
    """Create the database schema and seed the entity registry."""
    if not seed_only:
        typer.echo("Running database migrations...")
        _run_alembic_upgrade(alembic_ini)
        typer.echo("Migrations complete.")

    asyncio.run(_seed_registry())


async def _seed_registry() -> None:
    engine = make_engine()
    try:
        async with AsyncSession(engine, expire_on_commit=False) as session:
            inserted, skipped = await load_seed_registry(session)
        typer.echo(f"Registry: {inserted} inserted, {skipped} already present.")
    finally:
        await engine.dispose()


# ---------- ingest ----------


@ingest_app.command("sec")
def ingest_sec(
    ticker: Annotated[str, typer.Option(..., help="Ticker symbol, e.g. MSFT.")],
    since: Annotated[
        str | None,
        typer.Option(help="ISO date — only ingest filings on/after this, e.g. 2024-01-01."),
    ] = None,
    forms: Annotated[
        list[str],
        typer.Option("--form", help="SEC form codes; repeat to select multiple."),
    ] = ["10-K", "10-Q", "8-K"],  # noqa: B006 — typer reads list defaults this way
    extractor_name: Annotated[
        str,
        typer.Option("--extractor", help="Extractor: 'regex' (default, free) or 'claude'."),
    ] = "regex",
) -> None:
    """Fetch SEC EDGAR filings, extract deals, persist."""
    since_date = date.fromisoformat(since) if since else None
    extractor = _build_extractor(extractor_name)
    asyncio.run(_ingest_sec(ticker, since_date, list(forms), extractor))


async def _ingest_sec(
    ticker: str,
    since: date | None,
    forms: list[str],
    extractor: Extractor,
) -> None:
    engine = make_engine()
    try:
        async with (
            HttpClient() as http_client,
            AsyncSession(engine, expire_on_commit=False) as session,
        ):
            stats = await ingest_sec_filings_for_ticker(
                session=session,
                http_client=http_client,
                extractor=extractor,
                ticker=ticker,
                forms=forms,
                since=since,
            )
        _print_stats(f"SEC/{ticker}", stats)
    finally:
        await engine.dispose()


# ---------- graph ----------


@graph_app.command("render")
def graph_render(
    sector: Annotated[
        str | None,
        typer.Option(help="Sector tag, e.g. 'ai'. Omit to include every entity."),
    ] = None,
    as_of: Annotated[
        str | None,
        typer.Option("--as-of", help="ISO date — temporal filter on announced_at."),
    ] = None,
    output: Annotated[
        Path,
        typer.Option("--output", help="Output HTML path."),
    ] = Path("graph.html"),
    title: Annotated[
        str,
        typer.Option(help="Title shown in the rendered HTML."),
    ] = "midas",
) -> None:
    """Build the cash-flow graph and write it as interactive HTML."""
    as_of_date = date.fromisoformat(as_of) if as_of else None
    asyncio.run(_render_graph(sector, as_of_date, output, title))


async def _render_graph(
    sector: str | None,
    as_of: date | None,
    output: Path,
    title: str,
) -> None:
    engine = make_engine()
    try:
        async with AsyncSession(engine, expire_on_commit=False) as session:
            graph = await build_graph(session, sector=sector, as_of=as_of)
        n_nodes = graph.number_of_nodes()
        n_edges = graph.number_of_edges()
        if n_nodes == 0:
            typer.echo("Empty graph — nothing to render. (Run `midas ingest` first.)")
            return
        render_pyvis(graph, output, title=title)
        typer.echo(f"Rendered {n_nodes} nodes / {n_edges} edges to {output}")
    finally:
        await engine.dispose()


if __name__ == "__main__":  # pragma: no cover
    app()
