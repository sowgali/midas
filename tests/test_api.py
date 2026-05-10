"""End-to-end tests for the read-only HTTP API.

Strategy: stand up the real FastAPI app and override the ``get_session``
dependency to yield sessions backed by a single in-memory aiosqlite
engine populated with a small fixture. We exercise the routes through
``TestClient`` (sync) — FastAPI runs the underlying ``async def``
handlers on its own loop, so tests stay simple.

The fixture builds three entities (two AI, one not), three deals
between them (two parallel deals on one pair so aggregation has
something to collapse), and two evidence spans on the deepest deal.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Iterator
from datetime import date, datetime
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker
from sqlmodel import SQLModel

from midas.api.app import create_app
from midas.api.deps import get_session
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
from midas.storage.db import make_engine, make_session_factory

SQLITE_URL = "sqlite+aiosqlite:///:memory:"


# ---------- Fixture data ----------


class _Seed:
    """Holds the ids of seeded rows so tests can reference them."""

    def __init__(self) -> None:
        self.nvda_id: uuid.UUID = uuid.uuid4()
        self.openai_id: uuid.UUID = uuid.uuid4()
        self.acme_id: uuid.UUID = uuid.uuid4()  # non-ai entity
        self.deal_a_id: uuid.UUID = uuid.uuid4()  # NVDA -> OpenAI investment
        self.deal_b_id: uuid.UUID = uuid.uuid4()  # NVDA -> OpenAI commercial_contract
        self.deal_c_id: uuid.UUID = uuid.uuid4()  # OpenAI -> Acme (after as_of cutoff)
        self.source_id: uuid.UUID = uuid.uuid4()
        self.evidence_a_id: uuid.UUID = uuid.uuid4()
        self.evidence_b_id: uuid.UUID = uuid.uuid4()


async def _populate(engine: AsyncEngine, seed: _Seed) -> None:
    """Create schema and insert the fixture rows."""
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

    factory = make_session_factory(engine)
    async with factory() as s:
        nvda = Entity(
            id=seed.nvda_id,
            canonical_name="NVIDIA",
            aliases=["NVDA"],
            ticker="NVDA",
            cik="0001045810",
            entity_type=EntityType.PUBLIC_COMPANY,
            sector_tags=["ai", "semiconductors"],
            country="US",
        )
        openai = Entity(
            id=seed.openai_id,
            canonical_name="OpenAI",
            aliases=["OpenAI Global"],
            ticker=None,
            cik=None,
            entity_type=EntityType.PRIVATE_COMPANY,
            sector_tags=["ai"],
            country="US",
        )
        acme = Entity(
            id=seed.acme_id,
            canonical_name="Acme Holdings",
            aliases=[],
            ticker=None,
            cik=None,
            entity_type=EntityType.PRIVATE_COMPANY,
            sector_tags=["other"],
            country="US",
        )
        s.add_all([nvda, openai, acme])

        src = Source(
            id=seed.source_id,
            url="https://example.com/8k",
            source_type=SourceType.FORM_8K,
            publisher="SEC",
            title="NVIDIA Corp. 8-K",
            published_at=datetime(2025, 4, 1, 12, 0, 0),
            content_sha256="a" * 64,
        )
        s.add(src)

        deal_a = Deal(
            id=seed.deal_a_id,
            from_entity_id=seed.nvda_id,
            to_entity_id=seed.openai_id,
            deal_type=DealType.INVESTMENT,
            amount_usd=Decimal("1000000.00"),
            amount_native=Decimal("1000000.00"),
            currency="USD",
            announced_at=date(2025, 1, 15),
            closes_at=None,
            status=DealStatus.CLOSED,
            confidence=0.9,
            description="Strategic investment.",
        )
        deal_b = Deal(
            id=seed.deal_b_id,
            from_entity_id=seed.nvda_id,
            to_entity_id=seed.openai_id,
            deal_type=DealType.COMMERCIAL_CONTRACT,
            amount_usd=Decimal("500000.00"),
            amount_native=Decimal("500000.00"),
            currency="USD",
            announced_at=date(2025, 3, 1),
            closes_at=None,
            status=DealStatus.ANNOUNCED,
            confidence=0.8,
            description="Compute supply contract.",
        )
        # deal_c is announced *after* the as_of filter we'll test against.
        deal_c = Deal(
            id=seed.deal_c_id,
            from_entity_id=seed.openai_id,
            to_entity_id=seed.acme_id,
            deal_type=DealType.PARTNERSHIP,
            amount_usd=None,
            amount_native=None,
            currency=None,
            announced_at=date(2025, 9, 1),
            closes_at=None,
            status=DealStatus.ANNOUNCED,
            confidence=0.6,
            description="Joint research agreement.",
        )
        s.add_all([deal_a, deal_b, deal_c])

        ev_a = EvidenceSpan(
            id=seed.evidence_a_id,
            deal_id=seed.deal_a_id,
            source_id=seed.source_id,
            text_snippet="NVIDIA invested $1M in OpenAI.",
            char_start=0,
            char_end=30,
            extractor="claude:opus-4-7",
        )
        ev_b = EvidenceSpan(
            id=seed.evidence_b_id,
            deal_id=seed.deal_a_id,
            source_id=seed.source_id,
            text_snippet="The deal closed in Q1 2025.",
            char_start=120,
            char_end=147,
            extractor="regex:date",
        )
        s.add_all([ev_a, ev_b])

        await s.commit()


# ---------- pytest fixtures ----------


@pytest.fixture
async def engine() -> AsyncIterator[AsyncEngine]:
    eng = make_engine(SQLITE_URL)
    try:
        yield eng
    finally:
        await eng.dispose()


@pytest.fixture
async def seed(engine: AsyncEngine) -> _Seed:
    s = _Seed()
    await _populate(engine, s)
    return s


@pytest.fixture
def client(
    engine: AsyncEngine,
    seed: _Seed,  # ensures DB is populated before the client runs requests.
) -> Iterator[TestClient]:
    """A ``TestClient`` whose ``get_session`` is wired to the in-memory engine."""
    factory: async_sessionmaker[AsyncSession] = make_session_factory(engine)

    async def _override() -> AsyncIterator[AsyncSession]:
        async with factory() as session:
            yield session

    app = create_app()
    app.dependency_overrides[get_session] = _override
    with TestClient(app) as c:
        yield c


# ---------- /api/health ----------


def test_health_returns_ok(client: TestClient) -> None:
    resp = client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "version" in body


# ---------- /api/entities ----------


def test_list_entities_returns_seeded(client: TestClient, seed: _Seed) -> None:
    resp = client.get("/api/entities")
    assert resp.status_code == 200
    body = resp.json()
    ids = {row["id"] for row in body}
    assert ids == {str(seed.nvda_id), str(seed.openai_id), str(seed.acme_id)}
    nvda = next(row for row in body if row["id"] == str(seed.nvda_id))
    assert nvda["canonical_name"] == "NVIDIA"
    assert nvda["aliases"] == ["NVDA"]
    assert nvda["ticker"] == "NVDA"
    assert nvda["entity_type"] == "public_company"
    assert "ai" in nvda["sector_tags"]


def test_list_entities_filters_by_sector(client: TestClient, seed: _Seed) -> None:
    resp = client.get("/api/entities", params={"sector": "ai"})
    assert resp.status_code == 200
    ids = {row["id"] for row in resp.json()}
    assert ids == {str(seed.nvda_id), str(seed.openai_id)}


def test_get_entity_404_for_random_uuid(client: TestClient) -> None:
    missing = str(uuid.uuid4())
    resp = client.get(f"/api/entities/{missing}")
    assert resp.status_code == 404


def test_get_entity_returns_seeded(client: TestClient, seed: _Seed) -> None:
    resp = client.get(f"/api/entities/{seed.nvda_id}")
    assert resp.status_code == 200
    assert resp.json()["canonical_name"] == "NVIDIA"


# ---------- /api/graph ----------


def test_graph_aggregates_parallel_deals(client: TestClient, seed: _Seed) -> None:
    resp = client.get("/api/graph")
    assert resp.status_code == 200
    body = resp.json()

    node_ids = {n["id"] for n in body["nodes"]}
    assert node_ids == {str(seed.nvda_id), str(seed.openai_id), str(seed.acme_id)}

    # The two NVDA->OpenAI deals collapse into one aggregated edge.
    nvda_to_openai = next(
        e
        for e in body["edges"]
        if e["from_id"] == str(seed.nvda_id) and e["to_id"] == str(seed.openai_id)
    )
    assert nvda_to_openai["deal_count"] == 2
    assert nvda_to_openai["total_amount_usd"] == pytest.approx(1_500_000.0)
    assert nvda_to_openai["deal_types"] == sorted(
        {"investment", "commercial_contract"},
    )

    # The OpenAI -> Acme partnership has no amount; total stays None.
    openai_to_acme = next(
        e
        for e in body["edges"]
        if e["from_id"] == str(seed.openai_id) and e["to_id"] == str(seed.acme_id)
    )
    assert openai_to_acme["deal_count"] == 1
    assert openai_to_acme["total_amount_usd"] is None
    assert openai_to_acme["deal_types"] == ["partnership"]


def test_graph_sector_filter_strict_mode_drops_off_sector_nodes(
    client: TestClient, seed: _Seed,
) -> None:
    """``expand_transitively=False`` reproduces the old closed-world view."""
    resp = client.get(
        "/api/graph",
        params={"sector": "ai", "expand_transitively": "false"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["sector"] == "ai"
    node_ids = {n["id"] for n in body["nodes"]}
    # Acme is not tagged 'ai' so it's dropped along with its incident edge.
    assert node_ids == {str(seed.nvda_id), str(seed.openai_id)}
    edge_pairs = {(e["from_id"], e["to_id"]) for e in body["edges"]}
    assert edge_pairs == {(str(seed.nvda_id), str(seed.openai_id))}


def test_graph_sector_filter_default_expands_transitively(
    client: TestClient, seed: _Seed,
) -> None:
    """Default mode pulls Acme back in via the openai→acme partnership —
    the sector filter seeds the graph but doesn't truncate the chain.
    """
    resp = client.get("/api/graph", params={"sector": "ai"})
    assert resp.status_code == 200
    body = resp.json()
    node_ids = {n["id"] for n in body["nodes"]}
    assert node_ids == {str(seed.nvda_id), str(seed.openai_id), str(seed.acme_id)}
    edge_pairs = {(e["from_id"], e["to_id"]) for e in body["edges"]}
    assert (str(seed.openai_id), str(seed.acme_id)) in edge_pairs


def test_graph_filters_by_as_of(client: TestClient, seed: _Seed) -> None:
    resp = client.get("/api/graph", params={"as_of": "2025-06-01"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["as_of"] == "2025-06-01"
    edge_pairs = {(e["from_id"], e["to_id"]) for e in body["edges"]}
    # deal_c (announced 2025-09-01) is excluded; only NVDA -> OpenAI remains.
    assert edge_pairs == {(str(seed.nvda_id), str(seed.openai_id))}


# ---------- /api/deals ----------


def test_get_deal_detail_includes_endpoints_and_evidence(
    client: TestClient,
    seed: _Seed,
) -> None:
    resp = client.get(f"/api/deals/{seed.deal_a_id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["id"] == str(seed.deal_a_id)
    assert body["amount_usd"] == pytest.approx(1_000_000.0)
    assert body["from_entity"]["canonical_name"] == "NVIDIA"
    assert body["to_entity"]["canonical_name"] == "OpenAI"
    evidence = body["evidence"]
    assert len(evidence) == 2
    assert {ev["extractor"] for ev in evidence} == {
        "claude:opus-4-7",
        "regex:date",
    }
    # Source is inlined under each evidence span.
    assert evidence[0]["source"]["publisher"] == "SEC"


def test_get_deal_404_for_random_uuid(client: TestClient) -> None:
    resp = client.get(f"/api/deals/{uuid.uuid4()}")
    assert resp.status_code == 404


# ---------- CORS ----------


def test_cors_preflight_allows_vite_origin(client: TestClient) -> None:
    resp = client.options(
        "/api/health",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert resp.status_code == 200
    assert resp.headers.get("access-control-allow-origin") == "http://localhost:5173"
    allow_methods = resp.headers.get("access-control-allow-methods", "")
    assert "GET" in allow_methods or "*" in allow_methods
