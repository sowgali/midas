"""Async storage-layer tests against an in-memory SQLite database.

These tests don't exercise Postgres-specific features (JSONB, partial
unique indexes) — the migration layer covers that against a real
Postgres in CI later. Here we just verify the session/repo plumbing
round-trips correctly on a portable backend.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import date
from decimal import Decimal

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession
from sqlmodel import SQLModel

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
from midas.storage import (
    DealRepository,
    EntityRepository,
    EvidenceRepository,
    SourceRepository,
    make_engine,
    make_session_factory,
)

SQLITE_URL = "sqlite+aiosqlite:///:memory:"


@pytest.fixture
async def engine() -> AsyncIterator[AsyncEngine]:
    eng = make_engine(SQLITE_URL)
    async with eng.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    try:
        yield eng
    finally:
        await eng.dispose()


@pytest.fixture
async def db_session(engine: AsyncEngine) -> AsyncIterator[AsyncSession]:
    factory = make_session_factory(engine)
    async with factory() as s:
        yield s


# ---------- session factory + schema round-trip ----------


async def test_session_factory_round_trip(engine: AsyncEngine) -> None:
    factory = make_session_factory(engine)
    async with factory() as s:
        repo = EntityRepository(s)
        e = Entity(canonical_name="NVIDIA", entity_type=EntityType.PUBLIC_COMPANY)
        await repo.add(e)
        await s.commit()
        fetched = await repo.get(e.id)
        assert fetched is not None
        assert fetched.canonical_name == "NVIDIA"


# ---------- EntityRepository ----------


async def test_entity_repo_add_and_get_by_canonical_name(db_session: AsyncSession) -> None:
    repo = EntityRepository(db_session)
    e = Entity(
        canonical_name="Alphabet Inc.",
        aliases=["Google", "GOOGL"],
        ticker="GOOGL",
        entity_type=EntityType.PUBLIC_COMPANY,
        sector_tags=["ai", "cloud"],
    )
    await repo.add(e)
    await db_session.commit()

    by_name = await repo.get_by_canonical_name("Alphabet Inc.")
    assert by_name is not None
    assert by_name.id == e.id

    by_ticker = await repo.get_by_ticker("GOOGL")
    assert by_ticker is not None
    assert by_ticker.id == e.id

    by_sector = await repo.list_by_sector("ai")
    assert {x.id for x in by_sector} == {e.id}


# ---------- SourceRepository ----------


async def test_source_repo_upsert_is_idempotent(db_session: AsyncSession) -> None:
    repo = SourceRepository(db_session)
    digest = "c" * 64
    first = Source(
        url="https://example.com/a",
        source_type=SourceType.PRESS_RELEASE,
        publisher="Example Co.",
        content_sha256=digest,
    )
    inserted = await repo.upsert(first)
    await db_session.commit()

    duplicate = Source(
        url="https://example.com/duplicate",  # different URL, same hash
        source_type=SourceType.PRESS_RELEASE,
        publisher="Example Co.",
        content_sha256=digest,
    )
    again = await repo.upsert(duplicate)
    await db_session.commit()

    assert again.id == inserted.id
    # No second row was inserted.
    fetched = await repo.get_by_content_sha256(digest)
    assert fetched is not None
    assert fetched.id == inserted.id


# ---------- DealRepository ----------


async def test_deal_repo_list_outgoing_filters_by_as_of(db_session: AsyncSession) -> None:
    entities = EntityRepository(db_session)
    deals = DealRepository(db_session)

    payer = Entity(canonical_name="Microsoft", entity_type=EntityType.PUBLIC_COMPANY)
    payee = Entity(canonical_name="OpenAI", entity_type=EntityType.PRIVATE_COMPANY)
    await entities.add(payer)
    await entities.add(payee)

    early = Deal(
        from_entity_id=payer.id,
        to_entity_id=payee.id,
        deal_type=DealType.INVESTMENT,
        status=DealStatus.CLOSED,
        confidence=0.95,
        description="Early investment.",
        amount_usd=Decimal("1000000.00"),
        announced_at=date(2023, 1, 23),
    )
    later = Deal(
        from_entity_id=payer.id,
        to_entity_id=payee.id,
        deal_type=DealType.INVESTMENT,
        status=DealStatus.ANNOUNCED,
        confidence=0.9,
        description="Later commitment.",
        amount_usd=Decimal("10000000000.00"),
        announced_at=date(2025, 9, 15),
    )
    await deals.add(early)
    await deals.add(later)
    await db_session.commit()

    all_outgoing = await deals.list_outgoing(payer.id)
    assert {d.id for d in all_outgoing} == {early.id, later.id}

    historical = await deals.list_outgoing(payer.id, as_of=date(2024, 1, 1))
    assert {d.id for d in historical} == {early.id}

    incoming = await deals.list_incoming(payee.id, as_of=date(2024, 1, 1))
    assert {d.id for d in incoming} == {early.id}


# ---------- EvidenceRepository ----------


async def test_evidence_repo_add_many_persists_all(db_session: AsyncSession) -> None:
    entities = EntityRepository(db_session)
    sources = SourceRepository(db_session)
    deals = DealRepository(db_session)
    evidence = EvidenceRepository(db_session)

    payer = Entity(canonical_name="A Corp", entity_type=EntityType.PUBLIC_COMPANY)
    payee = Entity(canonical_name="B Corp", entity_type=EntityType.PRIVATE_COMPANY)
    await entities.add(payer)
    await entities.add(payee)

    src = Source(
        url="https://example.com/filing",
        source_type=SourceType.FORM_10K,
        publisher="SEC",
        content_sha256="d" * 64,
    )
    await sources.add(src)

    deal = Deal(
        from_entity_id=payer.id,
        to_entity_id=payee.id,
        deal_type=DealType.COMMERCIAL_CONTRACT,
        status=DealStatus.ANNOUNCED,
        confidence=0.8,
        description="Compute commitment.",
    )
    await deals.add(deal)

    spans = [
        EvidenceSpan(
            deal_id=deal.id,
            source_id=src.id,
            text_snippet="A Corp committed $1B in compute to B Corp.",
            char_start=0,
            char_end=44,
            extractor="claude:opus-4-7",
        ),
        EvidenceSpan(
            deal_id=deal.id,
            source_id=src.id,
            text_snippet="The agreement spans three years.",
            char_start=200,
            char_end=232,
            extractor="regex:duration",
        ),
    ]
    saved = await evidence.add_many(spans)
    await db_session.commit()

    assert len(saved) == 2
    assert all(s.id is not None for s in saved)
    assert {s.extractor for s in saved} == {"claude:opus-4-7", "regex:duration"}
