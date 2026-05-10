"""End-to-end tests for :func:`midas.graph.build_graph` against a real DB.

We spin up an in-memory SQLite database (same pattern as
``test_storage.py``) and exercise the filtering modes — sector, as_of
date, and explicit entity_ids — through actual ORM rows.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import date
from decimal import Decimal

import networkx as nx
import pytest
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession
from sqlmodel import SQLModel

from midas.graph import build_graph
from midas.models import Deal, DealStatus, DealType, Entity, EntityType
from midas.storage import (
    DealRepository,
    EntityRepository,
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


async def _seed_four_entities_five_deals(
    session: AsyncSession,
) -> tuple[Entity, Entity, Entity, Entity, list[Deal]]:
    """Seed the canonical 4-entity / 5-deal fixture used by every test below.

    Topology::

        msft (ai, cloud) ─[$10B 2023]──> openai (ai)
        msft             ─[$1B  2024]──> openai
        nvda (ai, hw)    ─[$500M 2024]─> openai
        nvda             ─[None  2025]─> anthropic (ai)
        msft             ─[$200M 2026]─> chevron  (energy)   ← future, off-sector

    ``chevron`` deliberately has *no* ``"ai"`` tag so the sector filter
    drops it (and its deal). The future-dated deal stress-tests
    ``as_of`` filtering.
    """
    entities = EntityRepository(session)
    deals = DealRepository(session)

    msft = Entity(
        canonical_name="Microsoft",
        ticker="MSFT",
        entity_type=EntityType.PUBLIC_COMPANY,
        sector_tags=["ai", "cloud"],
    )
    openai = Entity(
        canonical_name="OpenAI",
        entity_type=EntityType.PRIVATE_COMPANY,
        sector_tags=["ai"],
    )
    nvda = Entity(
        canonical_name="NVIDIA",
        ticker="NVDA",
        entity_type=EntityType.PUBLIC_COMPANY,
        sector_tags=["ai", "hardware"],
    )
    anthropic = Entity(
        canonical_name="Anthropic",
        entity_type=EntityType.PRIVATE_COMPANY,
        sector_tags=["ai"],
    )
    chevron = Entity(
        canonical_name="Chevron",
        ticker="CVX",
        entity_type=EntityType.PUBLIC_COMPANY,
        sector_tags=["energy"],
    )
    for e in (msft, openai, nvda, anthropic, chevron):
        await entities.add(e)

    seeded = [
        Deal(
            from_entity_id=msft.id,
            to_entity_id=openai.id,
            deal_type=DealType.INVESTMENT,
            status=DealStatus.CLOSED,
            confidence=0.95,
            description="Initial investment.",
            amount_usd=Decimal("10000000000.00"),
            announced_at=date(2023, 1, 23),
        ),
        Deal(
            from_entity_id=msft.id,
            to_entity_id=openai.id,
            deal_type=DealType.COMMERCIAL_CONTRACT,
            status=DealStatus.ANNOUNCED,
            confidence=0.9,
            description="Compute commitment.",
            amount_usd=Decimal("1000000000.00"),
            announced_at=date(2024, 6, 1),
        ),
        Deal(
            from_entity_id=nvda.id,
            to_entity_id=openai.id,
            deal_type=DealType.COMMERCIAL_CONTRACT,
            status=DealStatus.ANNOUNCED,
            confidence=0.85,
            description="GPU supply.",
            amount_usd=Decimal("500000000.00"),
            announced_at=date(2024, 9, 15),
        ),
        Deal(
            from_entity_id=nvda.id,
            to_entity_id=anthropic.id,
            deal_type=DealType.PARTNERSHIP,
            status=DealStatus.ANNOUNCED,
            confidence=0.7,
            description="Strategic partnership (no disclosed amount).",
            amount_usd=None,
            announced_at=date(2025, 3, 1),
        ),
        Deal(
            from_entity_id=msft.id,
            to_entity_id=chevron.id,
            deal_type=DealType.COMMERCIAL_CONTRACT,
            status=DealStatus.ANNOUNCED,
            confidence=0.6,
            description="Energy supply for datacenters.",
            amount_usd=Decimal("200000000.00"),
            announced_at=date(2026, 2, 1),
        ),
    ]
    for d in seeded:
        await deals.add(d)
    await session.commit()

    return msft, openai, nvda, anthropic, seeded


# ---------- build_graph ----------


async def test_build_graph_loads_all_entities_and_deals(db_session: AsyncSession) -> None:
    msft, openai, nvda, anthropic, deals = await _seed_four_entities_five_deals(db_session)

    graph = await build_graph(db_session)

    assert isinstance(graph, nx.MultiDiGraph)
    # 5 entities seeded; chevron is included when no filter is given.
    assert graph.number_of_nodes() == 5
    assert graph.number_of_edges() == 5

    # Node attrs round-trip canonical_name + entity_type + sector_tags.
    msft_attrs = graph.nodes[msft.id]
    assert msft_attrs["canonical_name"] == "Microsoft"
    assert msft_attrs["entity_type"] == "public_company"
    assert msft_attrs["ticker"] == "MSFT"
    assert "ai" in msft_attrs["sector_tags"]

    # Edge keys are stringified Deal IDs and addressable individually.
    first_deal = deals[0]
    edge_data = graph.get_edge_data(msft.id, openai.id, key=str(first_deal.id))
    assert edge_data is not None
    assert edge_data["deal_id"] == first_deal.id
    assert edge_data["amount_usd"] == 1.0e10
    assert edge_data["status"] == "closed"
    assert edge_data["announced_at"] == "2023-01-23"

    # Reach exists end-to-end.
    assert graph.has_edge(nvda.id, anthropic.id)


async def test_build_graph_as_of_excludes_future_deals(db_session: AsyncSession) -> None:
    msft, openai, nvda, anthropic, _ = await _seed_four_entities_five_deals(db_session)

    graph = await build_graph(db_session, as_of=date(2024, 12, 31))

    # Three deals on or before 2024-12-31: msft->openai x2, nvda->openai.
    assert graph.number_of_edges() == 3
    assert graph.has_edge(msft.id, openai.id)
    assert graph.has_edge(nvda.id, openai.id)
    # nvda->anthropic is 2025; chevron deal is 2026; both excluded.
    assert not graph.has_edge(nvda.id, anthropic.id)


async def test_build_graph_sector_filters_with_strict_truncates_off_sector_edges(
    db_session: AsyncSession,
) -> None:
    """Strict mode (``expand_transitively=False``) — pre-V1.9.1 behaviour."""
    msft, openai, nvda, anthropic, _ = await _seed_four_entities_five_deals(db_session)

    graph = await build_graph(db_session, sector="ai", expand_transitively=False)

    # Only the four AI-tagged entities; chevron is dropped.
    assert set(graph.nodes) == {msft.id, openai.id, nvda.id, anthropic.id}
    # And so is the msft->chevron edge — both endpoints must be in the
    # entity set for an edge to land in the graph.
    assert graph.number_of_edges() == 4
    for edge in graph.edges():
        assert edge[0] in graph.nodes
        assert edge[1] in graph.nodes


async def test_build_graph_entity_ids_strict_filters_to_subset(
    db_session: AsyncSession,
) -> None:
    """Strict mode keeps the old closed-world subset semantics."""
    msft, openai, _nvda, _anthropic, _ = await _seed_four_entities_five_deals(db_session)

    graph = await build_graph(
        db_session,
        entity_ids={msft.id, openai.id},
        expand_transitively=False,
    )

    # Only msft and openai land as nodes…
    assert set(graph.nodes) == {msft.id, openai.id}
    # …and only edges with both endpoints in that set survive (the
    # 2 msft->openai deals; nvda->openai is dropped because nvda was
    # filtered out).
    assert graph.number_of_edges() == 2
    for u, v in graph.edges():
        assert u in {msft.id, openai.id}
        assert v in {msft.id, openai.id}


async def test_build_graph_empty_entity_ids_returns_empty_graph(
    db_session: AsyncSession,
) -> None:
    await _seed_four_entities_five_deals(db_session)
    graph = await build_graph(db_session, entity_ids=set())
    assert graph.number_of_nodes() == 0
    assert graph.number_of_edges() == 0


# ---------- transitive expansion ----------
#
# Pre-V1.9.1 the graph silently truncated any deal whose other endpoint
# was outside the sector / id filter — most painful for discovered
# entities (empty sector_tags by construction). These tests pin the
# chain-preserving behavior and the strict opt-out.


async def test_sector_filter_expands_transitively_to_off_sector_endpoints(
    db_session: AsyncSession,
) -> None:
    """``sector="ai"`` keeps msft→chevron because msft is in the seed
    even though chevron is tagged only ``energy``. That's the cash-chain
    behaviour we want — the user filters by sector to *seed* the graph,
    not to amputate it.
    """
    msft, openai, nvda, anthropic, _ = await _seed_four_entities_five_deals(db_session)
    # chevron lives on the ``energy`` tag; locate it.
    from sqlmodel import select as _select

    from midas.models import Entity as _Entity

    rows = await db_session.execute(_select(_Entity).where(_Entity.canonical_name == "Chevron"))
    chevron = rows.scalars().one()

    graph = await build_graph(db_session, sector="ai")

    # All four AI entities + chevron pulled in transitively.
    assert set(graph.nodes) == {msft.id, openai.id, nvda.id, anthropic.id, chevron.id}
    # All five seeded deals survive — none truncated.
    assert graph.number_of_edges() == 5
    assert graph.has_edge(msft.id, chevron.id)


async def test_strict_mode_still_truncates_at_sector_boundary(
    db_session: AsyncSession,
) -> None:
    """``expand_transitively=False`` reproduces the old closed-world
    behaviour for callers who explicitly want it.
    """
    msft, openai, nvda, anthropic, _ = await _seed_four_entities_five_deals(db_session)

    graph = await build_graph(db_session, sector="ai", expand_transitively=False)

    assert set(graph.nodes) == {msft.id, openai.id, nvda.id, anthropic.id}
    # 4 AI-internal edges; msft→chevron dropped because chevron isn't in the set.
    assert graph.number_of_edges() == 4
    for u, v in graph.edges():
        assert u in graph.nodes
        assert v in graph.nodes


async def test_entity_ids_expands_transitively_along_chain(
    db_session: AsyncSession,
) -> None:
    """Seeded with just msft, the BFS pulls in openai (msft→openai), then
    chevron (msft→chevron). nvda is also reached via openai (nvda→openai
    is incoming on openai), and from there anthropic.
    """
    msft, openai, nvda, anthropic, _ = await _seed_four_entities_five_deals(db_session)

    graph = await build_graph(db_session, entity_ids={msft.id})

    # msft → openai (frontier 1) → nvda via inbound (frontier 2) → anthropic (frontier 3)
    # plus chevron from frontier 0.
    assert msft.id in graph.nodes
    assert openai.id in graph.nodes
    assert nvda.id in graph.nodes
    assert anthropic.id in graph.nodes
    # All 5 deals are visible since every endpoint is in the expanded set.
    assert graph.number_of_edges() == 5


async def test_transitive_expansion_respects_as_of_cutoff(
    db_session: AsyncSession,
) -> None:
    """``as_of`` is applied during BFS, so a chain that only continues
    via a future-dated deal stops at the cutoff.
    """
    msft, openai, nvda, anthropic, _ = await _seed_four_entities_five_deals(db_session)
    # chevron located via canonical_name lookup.
    from sqlmodel import select as _select

    from midas.models import Entity as _Entity

    rows = await db_session.execute(_select(_Entity).where(_Entity.canonical_name == "Chevron"))
    chevron = rows.scalars().one()

    # Cutoff before chevron deal (2026-02-01) and before nvda→anthropic
    # (2025-03-01). Seed = msft.
    graph = await build_graph(
        db_session,
        entity_ids={msft.id},
        as_of=date(2024, 12, 31),
    )

    # msft → openai is 2023/2024, included.
    # nvda reached via openai (nvda→openai is 2024-09-15, ≤ cutoff).
    # anthropic only reachable via nvda→anthropic (2025-03-01) — past cutoff,
    # so anthropic should NOT be expanded in.
    assert msft.id in graph.nodes
    assert openai.id in graph.nodes
    assert nvda.id in graph.nodes
    assert anthropic.id not in graph.nodes
    assert chevron.id not in graph.nodes  # 2026 deal past cutoff


async def test_transitive_expansion_terminates_on_cycle(
    db_session: AsyncSession,
) -> None:
    """A back-edge between two seeded nodes shouldn't loop the BFS forever."""
    a = Entity(
        canonical_name="A",
        entity_type=EntityType.PRIVATE_COMPANY,
        sector_tags=["ai"],
    )
    b = Entity(
        canonical_name="B",
        entity_type=EntityType.PRIVATE_COMPANY,
        sector_tags=[],  # off-sector, only reachable via expansion
    )
    c = Entity(
        canonical_name="C",
        entity_type=EntityType.PRIVATE_COMPANY,
        sector_tags=[],
    )
    repo = EntityRepository(db_session)
    for e in (a, b, c):
        await repo.add(e)
    deals = DealRepository(db_session)
    # A → B → C → A (cycle) plus B → C (parallel edge).
    for from_id, to_id in [(a.id, b.id), (b.id, c.id), (c.id, a.id), (b.id, c.id)]:
        await deals.add(
            Deal(
                from_entity_id=from_id,
                to_entity_id=to_id,
                deal_type=DealType.INVESTMENT,
                status=DealStatus.ANNOUNCED,
                confidence=0.8,
                description="cycle",
                amount_usd=Decimal("1000000.00"),
                announced_at=date(2024, 1, 1),
            ),
        )
    await db_session.commit()

    graph = await build_graph(db_session, sector="ai")  # seed = {A}

    # Cycle expanded fully without hanging; both off-sector nodes pulled in.
    assert set(graph.nodes) == {a.id, b.id, c.id}
    assert graph.number_of_edges() == 4
