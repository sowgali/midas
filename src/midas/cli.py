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
import uuid
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
from midas.pipeline import (
    IngestStats,
    ingest_playwright_source,
    ingest_rss_feed,
    ingest_sec_filings_for_ticker,
)
from midas.registry import (
    PlaywrightSourceConfig as YamlPlaywrightSourceConfig,
)
from midas.registry import (
    RssSourceConfig,
    load_seed_registry,
    parse_ir_sources,
)
from midas.sources.http_client import HttpClient
from midas.sources.ir_press import IrPressConfig
from midas.sources.playwright_source import PlaywrightSourceConfig
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
        f"deals(+/merged/skip)="
        f"{stats.deals_added}/{stats.deals_merged}/{stats.deals_skipped_unknown_party} "
        f"discovered={stats.entities_discovered} "
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
def serve(
    host: Annotated[
        str,
        typer.Option(help="Bind host for the HTTP server."),
    ] = "127.0.0.1",
    port: Annotated[
        int,
        typer.Option(help="Bind port for the HTTP server."),
    ] = 8000,
    reload: Annotated[
        bool,
        typer.Option("--reload", help="Restart the server on source changes (dev only)."),
    ] = False,
) -> None:
    """Run the FastAPI server (point a frontend at http://localhost:8000)."""
    import uvicorn

    uvicorn.run("midas.api.app:app", host=host, port=port, reload=reload)


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
        typer.Option(
            "--form",
            help="SEC form codes; repeat to select multiple. Default covers the three "
            "earnings-bearing forms (10-K annual, 10-Q quarterly, 8-K material events).",
        ),
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


@ingest_app.command("ir")
def ingest_ir(
    entity: Annotated[
        str | None,
        typer.Option(help="Canonical entity name; default = every configured source."),
    ] = None,
    since: Annotated[
        str | None,
        typer.Option(help="ISO date — only ingest items published on/after this."),
    ] = None,
    extractor_name: Annotated[
        str,
        typer.Option("--extractor", help="Extractor: 'regex' (default, free) or 'claude'."),
    ] = "regex",
) -> None:
    """Fetch IR / news / blog feeds, extract deals, persist."""
    since_date = date.fromisoformat(since) if since else None
    extractor = _build_extractor(extractor_name)
    asyncio.run(_ingest_ir(entity, since_date, extractor))


async def _ingest_ir(
    entity_filter: str | None,
    since: date | None,
    extractor: Extractor,
) -> None:
    from midas.pipeline import ingest_ir_press
    from midas.storage.repository import EntityRepository

    configs = parse_ir_sources()
    if entity_filter is not None:
        configs = [c for c in configs if c.entity_canonical_name.lower() == entity_filter.lower()]
        if not configs:
            typer.echo(f"No IR sources configured for {entity_filter!r}.")
            raise typer.Exit(code=1)

    engine = make_engine()
    try:
        async with (
            HttpClient() as http_client,
            AsyncSession(engine, expire_on_commit=False) as session,
        ):
            entity_map = {
                e.canonical_name: e.id for e in await EntityRepository(session).list_all()
            }

            total = IngestStats()
            for cfg in configs:
                entity_id = entity_map.get(cfg.entity_canonical_name)
                if entity_id is None:
                    typer.echo(
                        f"  ! unknown entity {cfg.entity_canonical_name!r} — "
                        "run `midas init --seed-only` first.",
                        err=True,
                    )
                    continue

                if isinstance(cfg, RssSourceConfig):
                    stats = await ingest_rss_feed(
                        session=session,
                        http_client=http_client,
                        extractor=extractor,
                        entity_id=entity_id,
                        feed_url=cfg.feed_url,
                        publisher=cfg.publisher,
                        source_type=cfg.source_type,
                        since=since,
                    )
                elif isinstance(cfg, YamlPlaywrightSourceConfig):
                    pw_config = PlaywrightSourceConfig(
                        entity_id=entity_id,
                        publisher=cfg.publisher,
                        index_url=cfg.index_url,
                        item_selector=cfg.item_selector,
                        title_selector=cfg.title_selector,
                        date_selector=cfg.date_selector,
                        date_format=cfg.date_format,
                        article_body_selector=cfg.article_body_selector,
                        link_base_url=cfg.link_base_url,
                        wait_after_load_ms=cfg.wait_after_load_ms,
                        navigation_timeout_ms=cfg.navigation_timeout_ms,
                        source_type=cfg.source_type,
                    )
                    stats = await ingest_playwright_source(
                        session=session,
                        extractor=extractor,
                        config=pw_config,
                        since=since,
                    )
                else:  # IrPressSourceConfig
                    ir_config = IrPressConfig(
                        entity_id=entity_id,
                        publisher=cfg.publisher,
                        index_url=cfg.index_url,
                        item_selector=cfg.item_selector,
                        link_selector=cfg.link_selector,
                        title_selector=cfg.title_selector,
                        date_selector=cfg.date_selector,
                        date_format=cfg.date_format,
                        article_body_selector=cfg.article_body_selector,
                        link_base_url=cfg.link_base_url,
                    )
                    stats = await ingest_ir_press(
                        session=session,
                        http_client=http_client,
                        extractor=extractor,
                        config=ir_config,
                        since=since,
                    )
                _print_stats(f"IR/{cfg.entity_canonical_name}", stats)
                total += stats
            _print_stats("IR/total", total)
    finally:
        await engine.dispose()


# ---------- graph ----------


review_app = typer.Typer(
    help="Inspect / promote auto-discovered entities (V1.8 open-world resolution).",
    no_args_is_help=True,
)
app.add_typer(review_app, name="review")


@review_app.command("list")
def review_list(
    limit: Annotated[
        int,
        typer.Option(help="Max rows to print."),
    ] = 50,
) -> None:
    """List discovered entities by deal-volume (most-flow first)."""
    asyncio.run(_review_list(limit=limit))


async def _review_list(*, limit: int) -> None:
    from collections import Counter

    from sqlmodel import col, select

    from midas.models import Deal, Entity

    engine = make_engine()
    try:
        async with AsyncSession(engine, expire_on_commit=False) as session:
            discovered = (
                (
                    await session.execute(
                        select(Entity).where(col(Entity.discovered).is_(True)),
                    )
                )
                .scalars()
                .all()
            )
            # Pulling all deals and counting in Python is fine here: the
            # whole point of this command is interactive review of a small
            # set (tens — at most low hundreds — of pending entities).
            deals = list((await session.execute(select(Deal))).scalars().all())
            from_count: Counter[uuid.UUID] = Counter(d.from_entity_id for d in deals)
            to_count: Counter[uuid.UUID] = Counter(d.to_entity_id for d in deals)

            ranked = sorted(
                discovered,
                key=lambda e: from_count[e.id] + to_count[e.id],
                reverse=True,
            )
            typer.echo(f"=== {len(ranked)} discovered entities ===")
            if not ranked:
                typer.echo("(none — registry is fully resolved)")
                return
            for ent in ranked[:limit]:
                typer.echo(
                    f"  {str(ent.id)[:8]}  "
                    f"out={from_count[ent.id]:>3}  in={to_count[ent.id]:>3}  "
                    f"{ent.canonical_name}",
                )
    finally:
        await engine.dispose()


@review_app.command("promote")
def review_promote(
    name_or_id: Annotated[str, typer.Argument(help="Canonical name or UUID prefix.")],
    canonical: Annotated[
        str | None,
        typer.Option(help="Override the canonical name (e.g. fix capitalization)."),
    ] = None,
    ticker: Annotated[str | None, typer.Option(help="Set ticker.")] = None,
    cik: Annotated[str | None, typer.Option(help="Set CIK (10-char zero-padded).")] = None,
    sector: Annotated[
        list[str] | None,
        typer.Option("--sector", help="Replace sector_tags. Repeat for multiple."),
    ] = None,
    alias: Annotated[
        list[str] | None,
        typer.Option("--alias", help="Add aliases. Repeat for multiple."),
    ] = None,
) -> None:
    """Promote a discovered entity to curated (sets ``discovered=False``)."""
    asyncio.run(
        _review_promote(
            name_or_id=name_or_id,
            canonical=canonical,
            ticker=ticker,
            cik=cik,
            sectors=sector,
            new_aliases=alias,
        ),
    )


async def _review_promote(
    *,
    name_or_id: str,
    canonical: str | None,
    ticker: str | None,
    cik: str | None,
    sectors: list[str] | None,
    new_aliases: list[str] | None,
) -> None:
    from sqlmodel import col, or_, select

    from midas.models import Entity

    engine = make_engine()
    try:
        async with AsyncSession(engine, expire_on_commit=False) as session:
            # Match by id-prefix OR canonical_name OR alias-membership.
            stmt = select(Entity).where(
                or_(
                    col(Entity.canonical_name) == name_or_id,
                    col(Entity.canonical_name).ilike(name_or_id),
                ),
            )
            ent = (await session.execute(stmt)).scalars().first()
            if ent is None:
                # Fall back to UUID-prefix scan over the discovered set.
                discovered = (
                    (
                        await session.execute(
                            select(Entity).where(col(Entity.discovered).is_(True)),
                        )
                    )
                    .scalars()
                    .all()
                )
                ent = next(
                    (e for e in discovered if str(e.id).startswith(name_or_id.lower())),
                    None,
                )
            if ent is None:
                typer.echo(f"No entity matched {name_or_id!r}.", err=True)
                raise typer.Exit(code=1)

            ent.discovered = False
            if canonical is not None:
                ent.canonical_name = canonical
            if ticker is not None:
                ent.ticker = ticker
            if cik is not None:
                ent.cik = cik
            if sectors is not None:
                ent.sector_tags = list(sectors)
            if new_aliases:
                ent.aliases = sorted({*ent.aliases, *new_aliases})
            session.add(ent)
            await session.commit()
            typer.echo(
                f"Promoted: {ent.canonical_name} "
                f"(ticker={ent.ticker} cik={ent.cik} sectors={ent.sector_tags})",
            )
    finally:
        await engine.dispose()


@app.command()
def reconcile(
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Show the plan but don't write changes.",
        ),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Print one line per merge."),
    ] = False,
) -> None:
    """Find duplicate Deal rows and merge them per the V1.6 dedup policy.

    Use this once to clean up data ingested before V1.6; the live
    pipeline now dedups at ingest time.
    """
    asyncio.run(_reconcile(dry_run=dry_run, verbose=verbose))


async def _reconcile(*, dry_run: bool, verbose: bool) -> None:
    from sqlalchemy import update as sa_update
    from sqlmodel import col, select

    from midas.dedup import deals_match, merge_duplicate_into
    from midas.models import Deal, EvidenceSpan
    from midas.storage.repository import EntityRepository

    engine = make_engine()
    try:
        async with AsyncSession(engine, expire_on_commit=False) as session:
            entity_name = {
                e.id: e.canonical_name for e in await EntityRepository(session).list_all()
            }

            deals = list(
                (await session.execute(select(Deal).order_by(col(Deal.created_at))))
                .scalars()
                .all(),
            )

            absorbed: set[object] = set()
            plan: list[tuple[Deal, Deal]] = []
            for i, canonical in enumerate(deals):
                if canonical.id in absorbed:
                    continue
                for later in deals[i + 1 :]:
                    if later.id in absorbed:
                        continue
                    if deals_match(
                        canonical,
                        from_entity_id=later.from_entity_id,
                        to_entity_id=later.to_entity_id,
                        deal_type=later.deal_type,
                        announced_at=later.announced_at,
                        amount_usd=later.amount_usd,
                    ):
                        absorbed.add(later.id)
                        plan.append((canonical, later))

            typer.echo("=== Reconciliation plan ===")
            typer.echo(
                f"  {len(plan)} candidate merges; "
                f"{len(deals)} -> {len(deals) - len(plan)} deals after apply",
            )
            if verbose and plan:
                typer.echo("")
                for canonical, dup in plan:
                    fr = entity_name.get(canonical.from_entity_id, "?")
                    to = entity_name.get(canonical.to_entity_id, "?")
                    cf = _fmt_amount(canonical.amount_usd)
                    df = _fmt_amount(dup.amount_usd)
                    cd = canonical.announced_at or "—"
                    dd = dup.announced_at or "—"
                    typer.echo(f"  {fr} → {to} ({canonical.deal_type})")
                    typer.echo(
                        f"    canon  {canonical.id}  {cf}  {canonical.status}  {cd}",
                    )
                    typer.echo(
                        f"    absorb {dup.id}  {df}  {dup.status}  {dd}",
                    )

            if dry_run:
                typer.echo("\n(dry-run; no changes written. Re-run without --dry-run to apply.)")
                return

            if not plan:
                typer.echo("Nothing to reconcile.")
                return

            for canonical, dup in plan:
                merge_duplicate_into(canonical, dup)
                session.add(canonical)
                # Move all EvidenceSpans from the duplicate onto the canonical.
                await session.execute(
                    sa_update(EvidenceSpan)
                    .where(col(EvidenceSpan.deal_id) == dup.id)
                    .values(deal_id=canonical.id),
                )
                await session.delete(dup)

            await session.commit()
            typer.echo(f"\nApplied {len(plan)} merges.")
    finally:
        await engine.dispose()


def _fmt_amount(amount: object) -> str:
    """Render a Decimal/float amount as $1.5B / $500M / —."""
    if amount is None:
        return "—"
    val = float(amount)  # type: ignore[arg-type]
    if abs(val) >= 1e9:
        return f"${val / 1e9:.1f}B"
    if abs(val) >= 1e6:
        return f"${val / 1e6:.0f}M"
    return f"${val:,.0f}"


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
    strict: Annotated[
        bool,
        typer.Option(
            "--strict/--transitive",
            help=(
                "Strict mode keeps only edges with BOTH endpoints inside the "
                "sector filter; the default --transitive expansion BFS-walks "
                "the deal graph so cash chains don't get truncated at the "
                "filter boundary."
            ),
        ),
    ] = False,
) -> None:
    """Build the cash-flow graph and write it as interactive HTML."""
    as_of_date = date.fromisoformat(as_of) if as_of else None
    asyncio.run(_render_graph(sector, as_of_date, output, title, expand=not strict))


async def _render_graph(
    sector: str | None,
    as_of: date | None,
    output: Path,
    title: str,
    *,
    expand: bool = True,
) -> None:
    engine = make_engine()
    try:
        async with AsyncSession(engine, expire_on_commit=False) as session:
            graph = await build_graph(
                session,
                sector=sector,
                as_of=as_of,
                expand_transitively=expand,
            )
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
