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
from decimal import Decimal
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
    IrPressSourceConfig,
    RssSourceConfig,
    load_seed_registry,
    parse_ir_sources,
)
from midas.registry import (
    PlaywrightSourceConfig as YamlPlaywrightSourceConfig,
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
    from sqlmodel import select

    from midas.models import DiscoveredSource, Entity
    from midas.pipeline import ingest_ir_press
    from midas.storage.repository import EntityRepository

    configs: list[
        RssSourceConfig | IrPressSourceConfig | YamlPlaywrightSourceConfig
    ] = list(parse_ir_sources())

    # V1.9.2: also pick up auto-discovered feeds from the DB. Each
    # DiscoveredSource row contributes an in-memory RssSourceConfig
    # alongside the YAML bootstrap. ``parse_ir_sources`` stays pure
    # (still tested against the YAML); the union happens here.
    engine_for_disc = make_engine()
    try:
        async with AsyncSession(engine_for_disc, expire_on_commit=False) as disc_session:
            rows = list(
                (
                    await disc_session.execute(
                        select(DiscoveredSource, Entity)
                        .join(Entity, DiscoveredSource.entity_id == Entity.id)  # type: ignore[arg-type]
                        .where(DiscoveredSource.status == "valid"),
                    )
                )
                .all(),
            )
        for row, ent in rows:
            configs.append(
                RssSourceConfig(
                    entity_canonical_name=ent.canonical_name,
                    type="rss",
                    feed_url=row.feed_url,
                    publisher=row.publisher,
                    source_type=row.source_type,
                ),
            )
    finally:
        await engine_for_disc.dispose()

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


# ---------- discover ----------


discover_app = typer.Typer(
    help="Auto-discover feed URLs for entities lacking a curated source.",
    no_args_is_help=True,
)
app.add_typer(discover_app, name="discover")


@discover_app.command("frontier")
def discover_frontier(
    max_rounds: Annotated[
        int,
        typer.Option(help="Cap on BFS rounds (safety bound; loop also stops on convergence)."),
    ] = 3,
    per_round_limit: Annotated[
        int,
        typer.Option(help="Max entities probed per round."),
    ] = 30,
    extractor_name: Annotated[
        str,
        typer.Option("--extractor", help="Extractor to use for downstream ingest."),
    ] = "regex",
    since: Annotated[
        str | None,
        typer.Option(help="ISO date — only ingest items on/after this."),
    ] = None,
) -> None:
    """Run the V1.9.2 BFS frontier loop until convergence.

    Each round:

    1. Probe heuristic feed URLs for every discovered entity that
       doesn't yet have a validated source.
    2. Ingest from the just-validated feeds (and any prior DB-discovered
       feeds that haven't been polled).
    3. New entities surface via the open-world resolver. Goto 1.

    Stops on the first round where no new sources land OR no new
    entities are discovered, or at ``--max-rounds``. Use ``--extractor
    regex`` (default, free) for cheap dry-runs; switch to ``claude`` to
    actually extract deals.
    """
    since_date = date.fromisoformat(since) if since else None
    extractor = _build_extractor(extractor_name)
    asyncio.run(
        _discover_frontier(
            max_rounds=max_rounds,
            per_round_limit=per_round_limit,
            extractor=extractor,
            since=since_date,
        ),
    )


async def _discover_frontier(
    *,
    max_rounds: int,
    per_round_limit: int,
    extractor: Extractor,
    since: date | None,
) -> None:
    from sqlmodel import col, select

    from midas.models import DiscoveredSource, Entity

    engine = make_engine()
    try:
        for round_n in range(1, max_rounds + 1):
            typer.echo(f"\n=========  Round {round_n} / {max_rounds}  =========")

            # Snapshot pre-state to detect convergence later.
            async with AsyncSession(engine, expire_on_commit=False) as snap_session:
                pre_entity_count = len(
                    list(
                        (
                            await snap_session.execute(
                                select(Entity).where(col(Entity.discovered).is_(True)),
                            )
                        )
                        .scalars()
                        .all(),
                    ),
                )
                pre_source_count = len(
                    list(
                        (
                            await snap_session.execute(
                                select(DiscoveredSource).where(
                                    DiscoveredSource.status == "valid",
                                ),
                            )
                        )
                        .scalars()
                        .all(),
                    ),
                )

            # Phase 1 — discover sources for discovered entities.
            typer.echo("  [phase 1] discover sources for discovered entities...")
            await _discover_sources(
                limit=per_round_limit,
                discovered_only=True,
                entity_filter=None,
                dry_run=False,
            )

            # Phase 2 — ingest only the just-added sources (cheap proxy:
            # re-run full IR ingest; the dedup layer skips already-seen
            # documents by content_sha256, so re-runs are no-ops on the
            # unchanged feeds).
            typer.echo("  [phase 2] ingest the validated feeds...")
            await _ingest_ir(None, since, extractor)

            # Convergence check.
            async with AsyncSession(engine, expire_on_commit=False) as snap_session:
                post_entity_count = len(
                    list(
                        (
                            await snap_session.execute(
                                select(Entity).where(col(Entity.discovered).is_(True)),
                            )
                        )
                        .scalars()
                        .all(),
                    ),
                )
                post_source_count = len(
                    list(
                        (
                            await snap_session.execute(
                                select(DiscoveredSource).where(
                                    DiscoveredSource.status == "valid",
                                ),
                            )
                        )
                        .scalars()
                        .all(),
                    ),
                )
            new_entities = post_entity_count - pre_entity_count
            new_sources = post_source_count - pre_source_count
            typer.echo(
                f"  round summary: +{new_sources} sources, +{new_entities} discovered entities",
            )
            if new_sources == 0 and new_entities == 0:
                typer.echo("Converged — no new sources or entities. Stopping.")
                break
        else:
            typer.echo("\nReached max-rounds cap; the frontier may still have work to do.")
    finally:
        await engine.dispose()


@discover_app.command("sources")
def discover_sources(
    limit: Annotated[
        int,
        typer.Option(help="Max entities to probe."),
    ] = 25,
    discovered_only: Annotated[
        bool,
        typer.Option(
            "--discovered-only/--all",
            help="Only probe entities with discovered=true (the BFS frontier). "
            "Pass --all to also probe curated entities (e.g. backfilling Vertiv).",
        ),
    ] = True,
    entity: Annotated[
        str | None,
        typer.Option(help="Only probe this canonical name (substring match)."),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Print hits but don't persist."),
    ] = False,
) -> None:
    """Probe heuristic feed URLs for entities lacking a curated source.

    Runs the V1.9.2 BFS source-discovery loop: for each candidate
    entity, derive likely domains from its canonical_name and probe
    common feed URL patterns. Valid hits are persisted to the
    ``discovered_source`` table so the next ``midas ingest ir`` pass
    picks them up alongside the YAML bootstrap.
    """
    asyncio.run(
        _discover_sources(
            limit=limit,
            discovered_only=discovered_only,
            entity_filter=entity,
            dry_run=dry_run,
        ),
    )


async def _discover_sources(
    *,
    limit: int,
    discovered_only: bool,
    entity_filter: str | None,
    dry_run: bool,
) -> None:
    import httpx
    from sqlmodel import col, select

    from midas.discovery.sources import discover_for_entity
    from midas.models import DiscoveredSource, Entity

    engine = make_engine()
    try:
        async with AsyncSession(engine, expire_on_commit=False) as session:
            # Build the candidate set.
            stmt = select(Entity)
            if discovered_only:
                stmt = stmt.where(col(Entity.discovered).is_(True))
            entities = list((await session.execute(stmt)).scalars().all())
            if entity_filter is not None:
                ef = entity_filter.lower()
                entities = [e for e in entities if ef in e.canonical_name.lower()]

            # Skip entities that already have at least one validated source.
            already = (
                (
                    await session.execute(
                        select(DiscoveredSource.entity_id).where(
                            DiscoveredSource.status == "valid",
                        ),
                    )
                )
                .scalars()
                .all()
            )
            done_ids = set(already)
            candidates = [e for e in entities if e.id not in done_ids]
            if len(candidates) > limit:
                candidates = candidates[:limit]

            typer.echo(
                f"=== Discovering feeds for {len(candidates)} of {len(entities)} entities ===",
            )
            if not candidates:
                typer.echo("Nothing to probe — every candidate already has a discovered source.")
                return

            ua = (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                "(KHTML, like Gecko) midas/1.9.2 Safari/537.36"
            )
            new_rows: list[DiscoveredSource] = []
            async with httpx.AsyncClient(
                headers={"User-Agent": ua},
                follow_redirects=True,
            ) as client:
                for ent in candidates:
                    hits = await discover_for_entity(client, ent, max_results=1)
                    if not hits:
                        typer.echo(f"  ✗  {ent.canonical_name}")
                        continue
                    hit = hits[0]
                    typer.echo(f"  ✓  {ent.canonical_name:<40}  {hit.feed_url}")
                    new_rows.append(
                        DiscoveredSource(
                            entity_id=ent.id,
                            feed_url=hit.feed_url,
                            publisher=hit.publisher,
                            source_type=hit.source_type,
                            status="valid",
                        ),
                    )

            typer.echo(f"\n{len(new_rows)} feed(s) discovered.")
            if dry_run:
                typer.echo("(dry-run; nothing written.)")
                return
            for row in new_rows:
                session.add(row)
            await session.commit()
            typer.echo(f"Wrote {len(new_rows)} rows to discovered_source.")
    finally:
        await engine.dispose()


@review_app.command("prune")
def review_prune(
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Print plan only."),
    ] = False,
) -> None:
    """Soft-delete discovered entities that violate the V1.9 filter rules.

    The filter (:func:`midas.entity_resolution.is_extractable_entity_name`)
    was tightened in V1.9.1 but pre-existing rows in the DB didn't get
    cleaned up retroactively. This walks every discovered entity, runs
    each through the current filter, and deletes the entity *plus its
    incident deals + evidence_spans* if the filter would reject it now.

    Use ``--dry-run`` first; the plan is printed either way.
    """
    asyncio.run(_review_prune(dry_run=dry_run))


async def _review_prune(*, dry_run: bool) -> None:
    from sqlalchemy import delete as sa_delete
    from sqlmodel import col, or_, select

    from midas.entity_resolution import is_extractable_entity_name
    from midas.models import Deal, Entity, EvidenceSpan

    engine = make_engine()
    try:
        async with AsyncSession(engine, expire_on_commit=False) as session:
            discovered = list(
                (
                    await session.execute(
                        select(Entity).where(col(Entity.discovered).is_(True)),
                    )
                )
                .scalars()
                .all(),
            )
            violators = [e for e in discovered if not is_extractable_entity_name(e.canonical_name)]

            if not violators:
                typer.echo("No violators — discovered set is clean.")
                return

            ent_ids = [e.id for e in violators]
            deals = list(
                (
                    await session.execute(
                        select(Deal).where(
                            or_(
                                col(Deal.from_entity_id).in_(ent_ids),
                                col(Deal.to_entity_id).in_(ent_ids),
                            ),
                        ),
                    )
                )
                .scalars()
                .all(),
            )
            deal_ids = [d.id for d in deals]

            typer.echo("=== Prune plan ===")
            typer.echo(
                f"  {len(violators)} entities → drop "
                f"{len(deals)} incident deals + their evidence",
            )
            for ent in sorted(violators, key=lambda e: e.canonical_name):
                typer.echo(f"  - {ent.canonical_name}")

            if dry_run:
                typer.echo("\n(dry-run; no changes written.)")
                return

            # Order: evidence → deals → entities (FK dependency).
            if deal_ids:
                await session.execute(
                    sa_delete(EvidenceSpan).where(col(EvidenceSpan.deal_id).in_(deal_ids)),
                )
                await session.execute(
                    sa_delete(Deal).where(col(Deal.id).in_(deal_ids)),
                )
            await session.execute(sa_delete(Entity).where(col(Entity.id).in_(ent_ids)))
            await session.commit()
            typer.echo(f"\nPruned {len(violators)} entities + {len(deals)} deals.")
    finally:
        await engine.dispose()


# ---------- insights ----------


insights_app = typer.Typer(
    help="Investment-decision queries: who's getting paid by the labs, "
    "how deep do the chains run.",
    no_args_is_help=True,
)
app.add_typer(insights_app, name="insights")


_DEFAULT_LAB_NAMES: tuple[str, ...] = (
    "OpenAI",
    "Anthropic",
    "Microsoft Corporation",
    "Alphabet Inc.",
    "Amazon.com, Inc.",
    "Meta Platforms, Inc.",
    "NVIDIA Corporation",
    "Oracle Corporation",
)


@insights_app.command("inflow")
def insights_inflow(
    payer: Annotated[
        list[str] | None,
        typer.Option(
            "--payer",
            help=(
                "Canonical name of a payer to include. Repeat for multiple. "
                "Defaults to the major labs (OpenAI/Anthropic/Microsoft/Alphabet/"
                "Amazon/Meta/NVIDIA/Oracle)."
            ),
        ),
    ] = None,
    limit: Annotated[
        int,
        typer.Option(help="Max rows to print."),
    ] = 30,
    as_of: Annotated[
        str | None,
        typer.Option("--as-of", help="ISO date — only consider deals on or before this."),
    ] = None,
) -> None:
    """Rank entities by total inbound $ from a set of payers.

    The default payer set is the major labs + hyperscalers, so this is
    the "who's catching the AI-capex wave" view. Override with one or
    more ``--payer`` to chase any other source of flows.
    """
    payers = tuple(payer) if payer else _DEFAULT_LAB_NAMES
    as_of_date = date.fromisoformat(as_of) if as_of else None
    asyncio.run(_insights_inflow(payers=payers, limit=limit, as_of=as_of_date))


async def _insights_inflow(
    *,
    payers: tuple[str, ...],
    limit: int,
    as_of: date | None,
) -> None:
    from sqlmodel import col, select

    from midas.graph.builder import build_graph
    from midas.insights import inflow_ranking
    from midas.models import Entity

    engine = make_engine()
    try:
        async with AsyncSession(engine, expire_on_commit=False) as session:
            # Resolve payer names → entity ids.
            stmt = select(Entity).where(col(Entity.canonical_name).in_(payers))
            payer_rows = list((await session.execute(stmt)).scalars().all())
            if not payer_rows:
                typer.echo(f"No payers matched any of {payers!r}.", err=True)
                raise typer.Exit(code=1)
            payer_ids = {e.id for e in payer_rows}

            # Build the full transitive graph (every deal everywhere). The
            # ranking restricts to edges *from* the payer set, so this is
            # the right primitive even though it's "wider" than needed.
            graph = await build_graph(session, as_of=as_of, expand_transitively=False)
        rows = inflow_ranking(graph, payer_ids=payer_ids)

        typer.echo(
            f"=== Inflow ranking — top {min(limit, len(rows))} of "
            f"{len(rows)} recipients ===",
        )
        typer.echo(f"Payers: {', '.join(e.canonical_name for e in payer_rows)}")
        if as_of is not None:
            typer.echo(f"as_of:  {as_of.isoformat()}")
        typer.echo("")
        typer.echo(f"  {'recipient':<48} {'total $':>14}  deals  payers")
        typer.echo(f"  {'-' * 48} {'-' * 14}  -----  ------")
        for row in rows[:limit]:
            total = _fmt_amount(row.total_usd) if row.total_usd > 0 else "—"
            payer_label = ",".join(row.payers)
            if len(payer_label) > 30:
                payer_label = payer_label[:27] + "..."
            name = row.canonical_name
            if len(name) > 47:
                name = name[:44] + "..."
            typer.echo(
                f"  {name:<48} {total:>14}  {row.deal_count:>5}  {payer_label}",
            )
    finally:
        await engine.dispose()


@insights_app.command("chain")
def insights_chain(
    seed: Annotated[str, typer.Argument(help="Canonical name (or substring) of the seed entity.")],
    max_hops: Annotated[
        int,
        typer.Option("--max-hops", help="How many BFS rings to traverse."),
    ] = 3,
    as_of: Annotated[
        str | None,
        typer.Option("--as-of", help="ISO date — only consider deals on or before this."),
    ] = None,
) -> None:
    """BFS-walk the outbound chain from ``seed`` and print each ring.

    Each hop lists the deals crossing into that ring, biggest-amount
    first. Use this to chase: ``Anthropic → AWS → ??? → ???`` and see
    where disclosed flows run out (i.e. where the next IR source needs
    to be added).
    """
    as_of_date = date.fromisoformat(as_of) if as_of else None
    asyncio.run(_insights_chain(seed=seed, max_hops=max_hops, as_of=as_of_date))


async def _insights_chain(
    *,
    seed: str,
    max_hops: int,
    as_of: date | None,
) -> None:
    from sqlmodel import col, or_, select

    from midas.graph.builder import build_graph
    from midas.insights import outbound_chain
    from midas.models import Entity

    engine = make_engine()
    try:
        async with AsyncSession(engine, expire_on_commit=False) as session:
            # Resolve seed by exact, then ilike-substring.
            stmt = select(Entity).where(
                or_(
                    col(Entity.canonical_name) == seed,
                    col(Entity.canonical_name).ilike(f"%{seed}%"),
                ),
            )
            matches = list((await session.execute(stmt)).scalars().all())
            if not matches:
                typer.echo(f"No entity matched {seed!r}.", err=True)
                raise typer.Exit(code=1)
            if len(matches) > 1:
                exact = [e for e in matches if e.canonical_name.lower() == seed.lower()]
                if exact:
                    seed_entity = exact[0]
                else:
                    typer.echo(
                        f"Ambiguous {seed!r}: matched {len(matches)} entities. "
                        f"First few: {[e.canonical_name for e in matches[:5]]}",
                        err=True,
                    )
                    raise typer.Exit(code=2)
            else:
                seed_entity = matches[0]

            graph = await build_graph(
                session,
                entity_ids={seed_entity.id},
                as_of=as_of,
                expand_transitively=True,
            )
        hops = outbound_chain(graph, seed_entity.id, max_hops=max_hops)

        typer.echo(f"=== Outbound chain from {seed_entity.canonical_name} ===")
        if as_of is not None:
            typer.echo(f"as_of: {as_of.isoformat()}")
        if not hops:
            typer.echo("(no outbound deals)")
            return
        for hop in hops:
            n = len(hop.edges)
            total_disclosed = sum(
                (e.amount_usd for e in hop.edges if e.amount_usd is not None),
                start=Decimal("0"),
            )
            typer.echo(
                f"\nHop {hop.hop}  ({n} deals, "
                f"disclosed total {_fmt_amount(total_disclosed)})",
            )
            for e in hop.edges:
                amt = _fmt_amount(e.amount_usd) if e.amount_usd is not None else "—"
                desc = e.description[:80] + "…" if len(e.description) > 80 else e.description
                typer.echo(
                    f"  {e.from_name:<28} → {e.to_name:<32} {e.deal_type:<20} {amt:>14}  {desc}",
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
