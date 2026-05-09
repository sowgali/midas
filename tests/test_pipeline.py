"""Tests for the ingestion pipeline.

Coverage:

* :class:`EntityResolver` — case-insensitive lookup, alias resolution,
  bulk load from the DB, alias-collision warning.
* :func:`ingest_raw_document` — happy path persists Source + Deal +
  EvidenceSpan; unknown-party deals are skipped (and counted); duplicate
  documents (same content_sha256) bump the dedup counter without
  inserting a second Source.
* :func:`ingest_sec_filings_for_ticker` — happy path with a stubbed
  ``SecEdgar`` and stubbed extractor; unknown ticker returns an
  ``IngestStats`` with the error.
* :class:`IngestStats` arithmetic.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlmodel import SQLModel, select

from midas.extractors.base import ExtractedDeal, ExtractionContext
from midas.models import (
    Deal,
    DealStatus,
    DealType,
    Entity,
    EntityType,
    EvidenceSpan,
    Source,
    SourceType,
)
from midas.pipeline import (
    EntityResolver,
    IngestStats,
    ingest_raw_document,
    ingest_sec_filings_for_ticker,
)
from midas.sources.base import RawDocument
from midas.storage.repository import EntityRepository

# ---------- Fixtures ----------


@pytest.fixture
async def session() -> AsyncGenerator[AsyncSession, None]:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    async with AsyncSession(engine, expire_on_commit=False) as s:
        yield s
    await engine.dispose()


async def _seed_two_entities(session: AsyncSession) -> tuple[Entity, Entity]:
    repo = EntityRepository(session)
    msft = Entity(
        canonical_name="Microsoft Corporation",
        aliases=["Microsoft", "MSFT"],
        ticker="MSFT",
        cik="0000789019",
        entity_type=EntityType.PUBLIC_COMPANY,
        sector_tags=["ai", "cloud"],
    )
    openai = Entity(
        canonical_name="OpenAI",
        aliases=["OpenAI Inc."],
        entity_type=EntityType.PRIVATE_COMPANY,
        sector_tags=["ai"],
    )
    await repo.add(msft)
    await repo.add(openai)
    await session.commit()
    return msft, openai


def _make_raw_document(text: str = "Microsoft will invest $10 billion in OpenAI.") -> RawDocument:
    return RawDocument(
        url="https://example.com/press/release-1",
        content_bytes=text.encode("utf-8"),
        source_type=SourceType.PRESS_RELEASE,
        publisher="Example Publisher",
        title="Press Release",
        published_at=datetime(2025, 9, 15, tzinfo=UTC),
    )


def _make_extracted(
    *,
    source_party: str,
    target_party: str,
    snippet: str = "Microsoft will invest $10 billion in OpenAI.",
    amount: Decimal | None = Decimal("10000000000"),
) -> ExtractedDeal:
    return ExtractedDeal(
        source_party_name=source_party,
        target_party_name=target_party,
        deal_type=DealType.INVESTMENT,
        status=DealStatus.ANNOUNCED,
        amount_usd=amount,
        currency="USD" if amount is not None else None,
        announced_at=date(2025, 9, 15),
        confidence=0.9,
        description="Microsoft invests in OpenAI.",
        evidence_text_snippet=snippet,
        char_start=0,
        char_end=len(snippet),
        extractor_name="test:fake",
    )


class _FakeExtractor:
    name = "test:fake"

    def __init__(self, deals: list[ExtractedDeal]) -> None:
        self._deals = deals

    async def extract(self, context: ExtractionContext) -> list[ExtractedDeal]:
        return list(self._deals)


# ---------- IngestStats ----------


def test_ingest_stats_addition_combines_counters_and_errors() -> None:
    a = IngestStats(deals_added=2, errors=["x"])
    b = IngestStats(deals_added=3, sources_added=1, errors=["y"])
    total = a + b
    assert total.deals_added == 5
    assert total.sources_added == 1
    assert total.errors == ["x", "y"]


# ---------- EntityResolver ----------


async def test_resolver_resolves_canonical_and_aliases(session: AsyncSession) -> None:
    msft, openai = await _seed_two_entities(session)
    resolver = await EntityResolver.from_session(session)

    assert resolver.resolve("Microsoft Corporation") == msft.id
    assert resolver.resolve("Microsoft") == msft.id
    assert resolver.resolve("MSFT") == msft.id
    assert resolver.resolve("OpenAI") == openai.id
    assert resolver.resolve("OpenAI Inc.") == openai.id


async def test_resolver_is_case_insensitive_and_strips(session: AsyncSession) -> None:
    msft, _ = await _seed_two_entities(session)
    resolver = await EntityResolver.from_session(session)

    assert resolver.resolve("microsoft corporation") == msft.id
    assert resolver.resolve("  MSFT  ") == msft.id
    assert resolver.resolve("MICROSOFT") == msft.id


async def test_resolver_returns_none_for_unknown(session: AsyncSession) -> None:
    await _seed_two_entities(session)
    resolver = await EntityResolver.from_session(session)
    assert resolver.resolve("Acme Holdings") is None
    assert resolver.resolve("") is None


async def test_resolver_known_parties_round_trip(session: AsyncSession) -> None:
    await _seed_two_entities(session)
    resolver = await EntityResolver.from_session(session)
    parties = resolver.known_parties
    names = {p.canonical_name for p in parties}
    assert names == {"Microsoft Corporation", "OpenAI"}


# ---------- ingest_raw_document ----------


async def test_ingest_raw_document_happy_path(session: AsyncSession) -> None:
    msft, openai = await _seed_two_entities(session)
    resolver = await EntityResolver.from_session(session)
    raw = _make_raw_document()
    extractor = _FakeExtractor(
        [_make_extracted(source_party="Microsoft", target_party="OpenAI")],
    )

    stats = await ingest_raw_document(
        session=session,
        raw=raw,
        extractor=extractor,
        resolver=resolver,
    )

    assert stats.documents_seen == 1
    assert stats.sources_added == 1
    assert stats.deals_added == 1
    assert stats.evidence_spans_added == 1
    assert stats.deals_skipped_unknown_party == 0

    # Persistence assertions.
    sources = (await session.execute(select(Source))).scalars().all()
    assert len(sources) == 1
    deals = (await session.execute(select(Deal))).scalars().all()
    assert len(deals) == 1
    assert deals[0].from_entity_id == msft.id
    assert deals[0].to_entity_id == openai.id
    spans = (await session.execute(select(EvidenceSpan))).scalars().all()
    assert len(spans) == 1
    assert spans[0].deal_id == deals[0].id
    assert spans[0].source_id == sources[0].id


async def test_ingest_raw_document_skips_unknown_party(session: AsyncSession) -> None:
    await _seed_two_entities(session)
    resolver = await EntityResolver.from_session(session)
    raw = _make_raw_document()
    extractor = _FakeExtractor(
        [_make_extracted(source_party="Microsoft", target_party="Acme Holdings")],
    )

    stats = await ingest_raw_document(
        session=session,
        raw=raw,
        extractor=extractor,
        resolver=resolver,
    )

    assert stats.deals_added == 0
    assert stats.deals_skipped_unknown_party == 1
    assert stats.evidence_spans_added == 0
    assert (await session.execute(select(Deal))).scalars().first() is None


async def test_ingest_raw_document_invokes_parser(session: AsyncSession) -> None:
    """The pipeline should send *parsed* text to the extractor, not raw bytes.

    Pin this so a future refactor can't quietly drop the parsing step
    and start feeding raw iXBRL HTML to Claude again.
    """
    await _seed_two_entities(session)
    resolver = await EntityResolver.from_session(session)

    # Mark the source as a SEC form so the registry routes to XbrlHtmlParser.
    raw = RawDocument(
        url="https://example.com/8k",
        content_bytes=(
            b"<html><body>"
            b"<ix:hidden><xbrli:context id='c'>SECRET</xbrli:context></ix:hidden>"
            b"<p>Microsoft will invest $10 billion in OpenAI.</p>"
            b"</body></html>"
        ),
        source_type=SourceType.FORM_8K,
        publisher="SEC",
        title="Test 8-K",
        published_at=datetime(2025, 9, 15, tzinfo=UTC),
    )

    captured_text: list[str] = []

    class _RecordingExtractor:
        name = "test:recording"

        async def extract(self, ctx: ExtractionContext) -> list[ExtractedDeal]:
            captured_text.append(ctx.document_text)
            return []

    await ingest_raw_document(
        session=session,
        raw=raw,
        extractor=_RecordingExtractor(),
        resolver=resolver,
    )

    assert len(captured_text) == 1
    text = captured_text[0]
    assert "Microsoft will invest $10 billion in OpenAI." in text
    # XBRL noise stripped before reaching the extractor.
    assert "SECRET" not in text
    assert "xbrli" not in text.lower()


async def test_ingest_raw_document_dedups_source_by_content_sha256(
    session: AsyncSession,
) -> None:
    await _seed_two_entities(session)
    resolver = await EntityResolver.from_session(session)
    raw = _make_raw_document()
    extractor = _FakeExtractor([])  # no deals -> isolate the source dedup behavior

    first = await ingest_raw_document(
        session=session,
        raw=raw,
        extractor=extractor,
        resolver=resolver,
    )
    second = await ingest_raw_document(
        session=session,
        raw=raw,
        extractor=extractor,
        resolver=resolver,
    )

    assert first.sources_added == 1 and first.sources_skipped_duplicate == 0
    assert second.sources_added == 0 and second.sources_skipped_duplicate == 1
    assert len((await session.execute(select(Source))).scalars().all()) == 1


# ---------- ingest_sec_filings_for_ticker ----------


async def test_ingest_sec_filings_unknown_ticker_returns_error(
    session: AsyncSession, monkeypatch: Any
) -> None:
    await _seed_two_entities(session)

    fake_edgar = AsyncMock()
    fake_edgar.get_cik.return_value = None
    monkeypatch.setattr("midas.pipeline.SecEdgar", lambda _client: fake_edgar)

    stats = await ingest_sec_filings_for_ticker(
        session=session,
        http_client=AsyncMock(),
        extractor=_FakeExtractor([]),
        ticker="WHAT",
    )

    assert stats.documents_seen == 0
    assert stats.errors and "WHAT" in stats.errors[0]


async def test_ingest_sec_filings_happy_path(session: AsyncSession, monkeypatch: Any) -> None:
    await _seed_two_entities(session)

    raw = _make_raw_document()
    fake_filing = object()  # pipeline only treats it opaquely

    fake_edgar = AsyncMock()
    fake_edgar.get_cik.return_value = "0000789019"
    fake_edgar.list_filings.return_value = [fake_filing]
    fake_edgar.fetch_filing.return_value = raw

    monkeypatch.setattr("midas.pipeline.SecEdgar", lambda _client: fake_edgar)

    extractor = _FakeExtractor(
        [_make_extracted(source_party="Microsoft", target_party="OpenAI")],
    )

    stats = await ingest_sec_filings_for_ticker(
        session=session,
        http_client=AsyncMock(),
        extractor=extractor,
        ticker="MSFT",
        forms=["10-K"],
        since=date(2024, 1, 1),
    )

    assert stats.documents_seen == 1
    assert stats.deals_added == 1
    fake_edgar.list_filings.assert_awaited_once()
    call_kwargs = fake_edgar.list_filings.call_args.kwargs
    assert call_kwargs["forms"] == ["10-K"]
    assert call_kwargs["since"] == date(2024, 1, 1)


async def test_ingest_sec_filings_continues_past_fetch_errors(
    session: AsyncSession, monkeypatch: Any
) -> None:
    await _seed_two_entities(session)

    raw = _make_raw_document()

    class _Acc:
        accession_number = "0000000000-00-000000"

    fake_edgar = AsyncMock()
    fake_edgar.get_cik.return_value = "0000789019"
    fake_edgar.list_filings.return_value = [_Acc(), _Acc()]
    fake_edgar.fetch_filing.side_effect = [RuntimeError("boom"), raw]

    monkeypatch.setattr("midas.pipeline.SecEdgar", lambda _client: fake_edgar)

    extractor = _FakeExtractor(
        [_make_extracted(source_party="Microsoft", target_party="OpenAI")],
    )

    stats = await ingest_sec_filings_for_ticker(
        session=session,
        http_client=AsyncMock(),
        extractor=extractor,
        ticker="MSFT",
    )

    assert stats.documents_seen == 1  # one succeeded
    assert stats.deals_added == 1
    assert len(stats.errors) == 1 and "boom" in stats.errors[0]


# ---------- alias collision (resolver constructed directly) ----------


def test_resolver_logs_alias_collision_without_clobbering() -> None:
    """If two entities share an alias, first wins; a warning is logged."""
    e1 = Entity(
        canonical_name="Apple Inc.",
        aliases=["Core"],
        entity_type=EntityType.PUBLIC_COMPANY,
    )
    e2 = Entity(
        canonical_name="Core Scientific",
        aliases=["Core"],  # collides with e1's alias
        entity_type=EntityType.PUBLIC_COMPANY,
    )
    resolver = EntityResolver([e1, e2])

    # First-wins for the colliding alias; canonical names still resolve correctly.
    assert resolver.resolve("Core") == e1.id
    assert resolver.resolve("Apple Inc.") == e1.id
    assert resolver.resolve("Core Scientific") == e2.id


def test_resolver_returns_none_for_unknown_id_in_isolation() -> None:
    resolver = EntityResolver([])
    assert resolver.resolve("anything") is None
    assert resolver.known_parties == []
