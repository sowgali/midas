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

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from midas.dedup import apply_merge, find_matching_deal
from midas.extractors.base import ExtractedDeal, ExtractionContext, Extractor, KnownParty
from midas.models import Deal, Entity, EvidenceSpan, Source
from midas.models.types import SourceType
from midas.parsers import Parser, select_parser
from midas.sources.base import RawDocument
from midas.sources.blog_rss import RssFeed
from midas.sources.http_client import HttpClient
from midas.sources.ir_press import IrPress, IrPressConfig
from midas.sources.sec_edgar import SecEdgar
from midas.storage.repository import (
    DealRepository,
    EntityRepository,
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
            deals_skipped_unknown_party=self.deals_skipped_unknown_party
            + other.deals_skipped_unknown_party,
            evidence_spans_added=self.evidence_spans_added + other.evidence_spans_added,
            errors=[*self.errors, *other.errors],
        )


# ---------- Entity resolution ----------


class EntityResolver:
    """Case-insensitive name → :class:`Entity` lookup.

    Built once per ingest run from the registry; cheap to query, no DB
    round-trips after construction.
    """

    def __init__(self, entities: Iterable[Entity]) -> None:
        self._by_name: dict[str, uuid.UUID] = {}
        self._known_parties: list[KnownParty] = []
        for entity in entities:
            self._index(entity, entity.canonical_name)
            for alias in entity.aliases:
                self._index(entity, alias)
            self._known_parties.append(
                KnownParty(
                    entity_id=entity.id,
                    canonical_name=entity.canonical_name,
                    aliases=list(entity.aliases),
                ),
            )

    def _index(self, entity: Entity, key: str) -> None:
        norm = key.strip().lower()
        if not norm:
            return
        if norm in self._by_name and self._by_name[norm] != entity.id:
            # Surface alias collisions loudly rather than silently winning.
            log.warning(
                "resolver.alias_collision",
                key=key,
                first_id=str(self._by_name[norm]),
                second_id=str(entity.id),
            )
            return
        self._by_name[norm] = entity.id

    def resolve(self, name: str) -> uuid.UUID | None:
        return self._by_name.get(name.strip().lower())

    @property
    def known_parties(self) -> list[KnownParty]:
        return list(self._known_parties)

    @classmethod
    async def from_session(cls, session: AsyncSession) -> EntityResolver:
        entities = await EntityRepository(session).list_all()
        return cls(entities)


# ---------- Core unit: one document -> deals ----------


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

    # 1. Source: upsert by content_sha256.
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
    if pre_existing is None:
        stats.sources_added += 1
    else:
        stats.sources_skipped_duplicate += 1

    # 2. Parse: raw bytes -> clean prose. SEC iXBRL gets the XBRL strip
    # treatment; press releases / RSS pass through (they're already prose).
    active_parser = parser if parser is not None else select_parser(raw)
    document_text = active_parser.parse(raw)

    # 3. Extract.
    context = ExtractionContext(
        source_id=source.id,
        source_url=raw.url,
        source_type=raw.source_type,
        known_parties=resolver.known_parties,
        document_text=document_text,
    )
    extracted: list[ExtractedDeal] = await extractor.extract(context)

    # 4. Resolve + persist.
    deal_repo = DealRepository(session)
    evidence_repo = EvidenceRepository(session)

    for ed in extracted:
        from_id = resolver.resolve(ed.source_party_name)
        to_id = resolver.resolve(ed.target_party_name)
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

        # V1.6 dedup: do we already have this deal?
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

    total = IngestStats()
    for item in items:
        try:
            raw = await press.fetch_article(item)
        except Exception as exc:
            total.errors.append(f"{config.publisher}/{item.url}: {exc}")
            continue
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

    total = IngestStats()
    for item in items:
        try:
            raw = await feed.fetch_article(item)
        except Exception as exc:
            total.errors.append(f"{publisher}/{item.url}: {exc}")
            log.warning("pipeline.rss.fetch_failed", url=item.url, error=str(exc))
            continue
        total += await ingest_raw_document(
            session=session,
            raw=raw,
            extractor=extractor,
            resolver=resolver,
        )
    return total
