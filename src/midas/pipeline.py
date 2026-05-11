"""End-to-end ingestion pipeline.

Wires the layered components together:

    Source.fetch() -> RawDocument
        -> SourceRepository.upsert  (idempotent on content_sha256)
        -> Extractor.extract        (regex / Claude / ...)
        -> EntityResolver.resolve   (name -> Entity.id, from registry)
        -> DealRepository.add + EvidenceRepository.add_many

V1 deliberately does NOT dedup deals across sources. Each extraction
produces a fresh :class:`Deal` row + its supporting :class:`EvidenceSpan`.
A later pass (V1.5) can collapse parallel deals between the same parties
with the same key (deal_type, announced_at, amount); for now the graph
layer's ``aggregate_by_pair`` covers the common visualization need.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from midas.dedup import apply_merge, find_matching_deal
from midas.entity_resolution import EntityResolver
from midas.extractors.base import ExtractedDeal, ExtractionContext, Extractor
from midas.models import Deal, EvidenceSpan, Source
from midas.models.types import SourceType
from midas.parsers import Parser, select_parser
from midas.sources.base import RawDocument
from midas.sources.blog_rss import RssFeed
from midas.sources.http_client import HttpClient
from midas.sources.ir_press import IrPress, IrPressConfig
from midas.sources.playwright_source import PlaywrightSource, PlaywrightSourceConfig
from midas.sources.sec_edgar import SecEdgar
from midas.storage.repository import (
    DealRepository,
    EvidenceRepository,
    SourceRepository,
)

log = structlog.get_logger(__name__)


# ---------- Stats ----------


@dataclass
class IngestStats:
    """Counters returned by every ingest_* call.

    Stats from independent calls compose with ``+``.
    """

    documents_seen: int = 0
    sources_added: int = 0
    sources_skipped_duplicate: int = 0
    deals_added: int = 0
    deals_merged: int = 0
    # V1.8 — number of NEW Entity rows the resolver auto-created during
    # this ingest (each tagged ``discovered=True``, ready for review).
    entities_discovered: int = 0
    # V1.8 — extracted parties whose names failed the quality filter
    # ("we", "the Company", etc.) and were dropped without creating a row.
    deals_skipped_unknown_party: int = 0
    evidence_spans_added: int = 0
    errors: list[str] = field(default_factory=list)

    def __add__(self, other: IngestStats) -> IngestStats:
        return IngestStats(
            documents_seen=self.documents_seen + other.documents_seen,
            sources_added=self.sources_added + other.sources_added,
            sources_skipped_duplicate=self.sources_skipped_duplicate
            + other.sources_skipped_duplicate,
            deals_added=self.deals_added + other.deals_added,
            deals_merged=self.deals_merged + other.deals_merged,
            entities_discovered=self.entities_discovered + other.entities_discovered,
            deals_skipped_unknown_party=self.deals_skipped_unknown_party
            + other.deals_skipped_unknown_party,
            evidence_spans_added=self.evidence_spans_added + other.evidence_spans_added,
            errors=[*self.errors, *other.errors],
        )


# ---------- Core unit: one document -> deals ----------


async def _upsert_source_and_parse(
    session: AsyncSession,
    raw: RawDocument,
    parser: Parser | None,
) -> tuple[Source, str, bool]:
    """Upsert the Source row and parse raw bytes to clean prose.

    Returns ``(source, document_text, is_new_source)``. Factored out so
    both the single-doc and batch ingest paths can reuse it before
    diverging on how they call the extractor.
    """
    source_repo = SourceRepository(session)
    pre_existing = await source_repo.get_by_content_sha256(raw.content_sha256)
    source = await source_repo.upsert(
        Source(
            url=raw.url,
            source_type=raw.source_type,
            publisher=raw.publisher,
            title=raw.title,
            published_at=raw.published_at,
            content_sha256=raw.content_sha256,
        ),
    )
    active_parser = parser if parser is not None else select_parser(raw)
    document_text = active_parser.parse(raw)
    return source, document_text, pre_existing is None


async def _resolve_and_persist_deals(
    session: AsyncSession,
    *,
    source: Source,
    deals: list[ExtractedDeal],
    resolver: EntityResolver,
    stats: IngestStats,
) -> None:
    """Walk extracted deals through entity resolution + dedup + persist.

    Updates ``stats`` in-place. Same logic the single-doc path used
    inline pre-V1.9.4; factored out so the batch path can reuse it
    without diverging on dedup semantics.
    """
    deal_repo = DealRepository(session)
    evidence_repo = EvidenceRepository(session)

    for ed in deals:
        before_known = len(resolver.known_parties)
        from_id = await resolver.resolve_or_create(session, ed.source_party_name)
        to_id = await resolver.resolve_or_create(session, ed.target_party_name)
        stats.entities_discovered += len(resolver.known_parties) - before_known

        if from_id is None or to_id is None:
            stats.deals_skipped_unknown_party += 1
            log.debug(
                "pipeline.skip.unknown_party",
                source_party=ed.source_party_name,
                target_party=ed.target_party_name,
                from_resolved=from_id is not None,
                to_resolved=to_id is not None,
            )
            continue

        existing = await find_matching_deal(
            session,
            from_entity_id=from_id,
            to_entity_id=to_id,
            deal_type=ed.deal_type,
            announced_at=ed.announced_at,
            amount_usd=ed.amount_usd,
        )

        if existing is None:
            deal = Deal(
                from_entity_id=from_id,
                to_entity_id=to_id,
                deal_type=ed.deal_type,
                amount_usd=ed.amount_usd,
                amount_native=ed.amount_native,
                currency=ed.currency,
                announced_at=ed.announced_at,
                closes_at=ed.closes_at,
                status=ed.status,
                confidence=ed.confidence,
                description=ed.description,
            )
            await deal_repo.add(deal)
            stats.deals_added += 1
            target_deal_id = deal.id
        else:
            apply_merge(existing, ed)
            session.add(existing)
            await session.flush()
            stats.deals_merged += 1
            log.debug(
                "pipeline.deal.merged",
                deal_id=str(existing.id),
                from_=ed.source_party_name,
                to=ed.target_party_name,
            )
            target_deal_id = existing.id

        await evidence_repo.add_many(
            [
                EvidenceSpan(
                    deal_id=target_deal_id,
                    source_id=source.id,
                    text_snippet=ed.evidence_text_snippet,
                    char_start=ed.char_start,
                    char_end=ed.char_end,
                    extractor=ed.extractor_name,
                ),
            ],
        )
        stats.evidence_spans_added += 1


async def ingest_raw_document(
    *,
    session: AsyncSession,
    raw: RawDocument,
    extractor: Extractor,
    resolver: EntityResolver,
    parser: Parser | None = None,
) -> IngestStats:
    """Persist one fetched document end-to-end.

    Idempotent at the Source level (via ``content_sha256`` upsert); when
    the same document is ingested twice the second call is mostly a
    no-op for sources but will re-run extraction and re-insert deals.
    Deal-level dedup is deferred to V1.5.

    The optional ``parser`` strips raw bytes (Inline XBRL, generic HTML)
    down to clean prose before extraction. When omitted, the pipeline
    picks one via :func:`midas.parsers.select_parser` based on
    ``raw.source_type`` — XBRL parser for SEC forms, pass-through for
    sources that already deliver UTF-8 prose.

    Commits on success.
    """
    stats = IngestStats(documents_seen=1)
    source, document_text, is_new = await _upsert_source_and_parse(session, raw, parser)
    if is_new:
        stats.sources_added += 1
    else:
        stats.sources_skipped_duplicate += 1

    context = ExtractionContext(
        source_id=source.id,
        source_url=raw.url,
        source_type=raw.source_type,
        known_parties=resolver.known_parties,
        document_text=document_text,
    )
    extracted: list[ExtractedDeal] = await extractor.extract(context)

    await _resolve_and_persist_deals(
        session,
        source=source,
        deals=extracted,
        resolver=resolver,
        stats=stats,
    )
    await session.commit()
    return stats


async def ingest_raw_documents_batched(
    *,
    session: AsyncSession,
    raws: list[RawDocument],
    extractor: Any,  # must expose ``extract_many``; runtime-checked below
    resolver: EntityResolver,
    parser: Parser | None = None,
) -> IngestStats:
    """Ingest many documents in one batched extraction round.

    All raws first get source-upserted + parsed serially (cheap), then a
    single ``extractor.extract_many(contexts)`` call extracts deals for
    every doc in one Anthropic batch (50% cost saving). Per-doc resolve
    + persist runs sequentially in order so entity discoveries from
    earlier docs are visible to the resolver before later docs commit.

    Same correctness model as the per-doc :func:`ingest_raw_document`:
    one session.commit() at the end after all docs persist successfully,
    so a single broken doc rolls the whole batch back. The caller is
    expected to batch by feed (typically tens of docs), not the entire
    ingest run.
    """
    if not hasattr(extractor, "extract_many"):
        raise TypeError(
            f"extractor {type(extractor).__name__!r} has no extract_many; "
            "use ingest_raw_document for non-batchable extractors.",
        )
    if not raws:
        return IngestStats()

    stats = IngestStats(documents_seen=len(raws))

    # Step 1: upsert sources + parse text for every raw (serial, cheap).
    per_doc: list[tuple[Source, ExtractionContext]] = []
    for raw in raws:
        source, document_text, is_new = await _upsert_source_and_parse(session, raw, parser)
        if is_new:
            stats.sources_added += 1
        else:
            stats.sources_skipped_duplicate += 1
        per_doc.append(
            (
                source,
                ExtractionContext(
                    source_id=source.id,
                    source_url=raw.url,
                    source_type=raw.source_type,
                    known_parties=resolver.known_parties,
                    document_text=document_text,
                ),
            ),
        )

    # Step 2: ONE batch call for every doc's extraction.
    contexts = [ctx for _, ctx in per_doc]
    extracted_per_doc: list[list[ExtractedDeal]] = await extractor.extract_many(contexts)

    # Step 3: resolve + persist each doc's deals in order, so cross-doc
    # entity discovery still flows through the resolver.
    for (source, _ctx), deals in zip(per_doc, extracted_per_doc, strict=True):
        await _resolve_and_persist_deals(
            session,
            source=source,
            deals=deals,
            resolver=resolver,
            stats=stats,
        )

    await session.commit()
    return stats


# ---------- High-level convenience entry points ----------


async def ingest_sec_filings_for_ticker(
    *,
    session: AsyncSession,
    http_client: HttpClient,
    extractor: Extractor,
    ticker: str,
    forms: Iterable[str] | None = None,
    since: date | None = None,
) -> IngestStats:
    """Fetch + ingest all matching SEC filings for ``ticker``.

    ``forms`` defaults to the three most useful (10-K, 10-Q, 8-K).
    """
    edgar = SecEdgar(http_client)
    cik = await edgar.get_cik(ticker)
    if cik is None:
        return IngestStats(errors=[f"unknown ticker: {ticker}"])

    filings = await edgar.list_filings(
        cik,
        forms=list(forms) if forms is not None else ["10-K", "10-Q", "8-K"],
        since=since,
    )
    resolver = await EntityResolver.from_session(session)

    total = IngestStats()
    for filing in filings:
        try:
            raw = await edgar.fetch_filing(filing)
        except Exception as exc:
            total.errors.append(f"{ticker}/{filing.accession_number}: {exc}")
            log.warning("pipeline.fetch_failed", ticker=ticker, error=str(exc))
            continue
        total += await ingest_raw_document(
            session=session,
            raw=raw,
            extractor=extractor,
            resolver=resolver,
        )
    return total


async def ingest_ir_press(
    *,
    session: AsyncSession,
    http_client: HttpClient,
    extractor: Extractor,
    config: IrPressConfig,
    since: date | None = None,
) -> IngestStats:
    """Fetch + ingest one company's IR press feed.

    Resilient at the index level: a 403 / 5xx / parse-error on the
    index page returns an :class:`IngestStats` with one error entry
    rather than raising — keeps a multi-source run going past one
    bad site.
    """
    press = IrPress(config, http_client=http_client)
    try:
        items = await press.list_items(since=since)
    except Exception as exc:
        log.warning(
            "pipeline.ir_press.list_items_failed",
            index_url=config.index_url,
            error=str(exc),
        )
        return IngestStats(
            errors=[f"{config.publisher}/{config.index_url}: list_items failed: {exc}"],
        )

    resolver = await EntityResolver.from_session(session)

    raws: list[RawDocument] = []
    total = IngestStats()
    for item in items:
        try:
            raws.append(await press.fetch_article(item))
        except Exception as exc:
            total.errors.append(f"{config.publisher}/{item.url}: {exc}")

    if hasattr(extractor, "extract_many"):
        total += await ingest_raw_documents_batched(
            session=session,
            raws=raws,
            extractor=extractor,
            resolver=resolver,
        )
    else:
        for raw in raws:
            total += await ingest_raw_document(
                session=session,
                raw=raw,
                extractor=extractor,
                resolver=resolver,
            )
    return total


async def ingest_rss_feed(
    *,
    session: AsyncSession,
    http_client: HttpClient,
    extractor: Extractor,
    entity_id: uuid.UUID,
    feed_url: str,
    publisher: str,
    source_type: SourceType = SourceType.BLOG,
    since: date | None = None,
) -> IngestStats:
    """Fetch + ingest one company's RSS / Atom feed.

    Mirrors :func:`ingest_ir_press` but for the RSS path. Body
    extraction is the generic "drop-script-find-article" fallback
    inside :class:`RssFeed`; configurable per-site selectors live on
    the IR-press path.

    Resilient at the feed-index level: a 403 / 5xx / parse-error on
    the feed itself returns an :class:`IngestStats` with one error
    entry rather than raising — keeps a multi-feed run going past
    one bad feed.
    """
    feed = RssFeed(
        entity_id=entity_id,
        feed_url=feed_url,
        publisher=publisher,
        source_type=source_type,
        http_client=http_client,
    )
    try:
        items = await feed.list_items(since=since)
    except Exception as exc:
        log.warning("pipeline.rss.list_items_failed", feed_url=feed_url, error=str(exc))
        return IngestStats(errors=[f"{publisher}/{feed_url}: list_items failed: {exc}"])

    resolver = await EntityResolver.from_session(session)

    # Fetch all articles up front so we can either feed them one-by-one
    # to ingest_raw_document (real-time) or hand the whole batch to
    # ingest_raw_documents_batched.
    raws: list[RawDocument] = []
    total = IngestStats()
    for item in items:
        try:
            raws.append(await feed.fetch_article(item))
        except Exception as exc:
            total.errors.append(f"{publisher}/{item.url}: {exc}")
            log.warning("pipeline.rss.fetch_failed", url=item.url, error=str(exc))

    if hasattr(extractor, "extract_many"):
        total += await ingest_raw_documents_batched(
            session=session,
            raws=raws,
            extractor=extractor,
            resolver=resolver,
        )
    else:
        for raw in raws:
            total += await ingest_raw_document(
                session=session,
                raw=raw,
                extractor=extractor,
                resolver=resolver,
            )
    return total


async def ingest_playwright_source(
    *,
    session: AsyncSession,
    extractor: Extractor,
    config: PlaywrightSourceConfig,
    since: date | None = None,
) -> IngestStats:
    """Fetch + ingest one Playwright-backed source (Cloudflare / JS-rendered).

    Used for OpenAI / Anthropic news where the RSS / IR-press paths
    don't work. Spins up one Chromium for the entire feed; that's the
    expensive bit, so don't fan out per-item.
    """
    resolver = await EntityResolver.from_session(session)
    total = IngestStats()
    try:
        async with PlaywrightSource(config) as source:
            try:
                items = await source.list_items(since=since)
            except Exception as exc:
                log.warning(
                    "pipeline.playwright.list_items_failed",
                    index_url=config.index_url,
                    error=str(exc),
                )
                return IngestStats(
                    errors=[
                        f"{config.publisher}/{config.index_url}: list_items failed: {exc}",
                    ],
                )

            raws: list[RawDocument] = []
            for item in items:
                try:
                    raws.append(await source.fetch_article(item))
                except Exception as exc:
                    total.errors.append(f"{config.publisher}/{item.url}: {exc}")
                    log.warning(
                        "pipeline.playwright.fetch_failed",
                        url=item.url,
                        error=str(exc),
                    )
            if hasattr(extractor, "extract_many"):
                total += await ingest_raw_documents_batched(
                    session=session,
                    raws=raws,
                    extractor=extractor,
                    resolver=resolver,
                )
            else:
                for raw in raws:
                    total += await ingest_raw_document(
                        session=session,
                        raw=raw,
                        extractor=extractor,
                        resolver=resolver,
                    )
    except Exception as exc:
        # Browser launch failure (Chromium missing, OS error).
        log.warning(
            "pipeline.playwright.browser_failed",
            index_url=config.index_url,
            error=str(exc),
        )
        total.errors.append(
            f"{config.publisher}/{config.index_url}: browser unavailable: {exc}",
        )
    return total
