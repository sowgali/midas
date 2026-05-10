"""Postgres → :class:`networkx.MultiDiGraph` builder.

Why ``MultiDiGraph``? Two entities can be connected by many distinct
deals (a string of investments, a partnership *and* a commercial
contract, etc.) and we need each one as its own edge so provenance and
amounts don't get squashed together. We use ``Deal.id`` (as ``str``) as
the edge key so individual edges remain addressable.

Edge attribute types are chosen to be JSON-serialisable: ``Decimal``
amounts are downcast to ``float``, dates to ISO strings, enums to their
``str`` values. Callers that need ``Decimal`` precision (totals,
comparisons) should use :mod:`midas.graph.queries`, which re-constructs
``Decimal`` from the ``float`` attribute on demand.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable
from datetime import date

import networkx as nx
from sqlalchemy import or_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col, select

from midas.models import Deal, Entity


def _entity_node_attrs(entity: Entity) -> dict[str, object]:
    """Pack the subset of :class:`Entity` columns we expose on graph nodes."""
    return {
        "canonical_name": entity.canonical_name,
        "entity_type": str(entity.entity_type),
        "ticker": entity.ticker,
        "sector_tags": list(entity.sector_tags),
    }


def _deal_edge_attrs(deal: Deal) -> dict[str, object]:
    """Pack the subset of :class:`Deal` columns we expose on graph edges.

    ``amount_usd`` is converted from ``Decimal`` to ``float`` so the graph
    is JSON-serialisable; precision-sensitive callers reconstruct
    ``Decimal`` from this float in :mod:`midas.graph.queries`.
    """
    return {
        "deal_id": deal.id,
        "deal_type": str(deal.deal_type),
        "amount_usd": float(deal.amount_usd) if deal.amount_usd is not None else None,
        "announced_at": deal.announced_at.isoformat() if deal.announced_at is not None else None,
        "status": str(deal.status),
        "confidence": float(deal.confidence),
        "description": deal.description,
    }


async def _expand_transitively(
    session: AsyncSession,
    seed: set[uuid.UUID],
    *,
    as_of: date | None = None,
) -> set[uuid.UUID]:
    """BFS-expand ``seed`` through the deal graph until no new endpoints emerge.

    This is what fixes the V1.9 truncation bug: a sector or explicit-id
    filter scopes *which entities the user cares about*, but the cash
    chain we want to visualize keeps going — Alphabet → Wiz → (Wiz's
    cloud-spend counterparties) → ... — even when downstream nodes have
    no ``sector_tags`` (typical for newly-discovered entities).

    Each iteration looks for deals that touch the current frontier on
    either side, adds the *other* endpoint to the visited set, and
    advances the frontier to those newly-found endpoints. Terminates at
    leaves or once a cycle has been fully explored — the visited set
    monotonically grows and frontier shrinks to ∅.

    The ``as_of`` cutoff is applied during expansion so we don't reach
    through deals that wouldn't be in the final graph anyway.
    """
    visited = set(seed)
    frontier = set(seed)
    while frontier:
        stmt = select(Deal.from_entity_id, Deal.to_entity_id).where(
            or_(
                col(Deal.from_entity_id).in_(frontier),
                col(Deal.to_entity_id).in_(frontier),
            ),
        )
        if as_of is not None:
            stmt = stmt.where(col(Deal.announced_at).is_not(None)).where(
                col(Deal.announced_at) <= as_of,
            )
        result = await session.execute(stmt)
        new_endpoints: set[uuid.UUID] = set()
        for from_id, to_id in result.all():
            new_endpoints.add(from_id)
            new_endpoints.add(to_id)
        new_endpoints -= visited
        visited |= new_endpoints
        frontier = new_endpoints
    return visited


async def build_graph(
    session: AsyncSession,
    *,
    sector: str | None = None,
    as_of: date | None = None,
    entity_ids: Iterable[uuid.UUID] | None = None,
    expand_transitively: bool = True,
) -> nx.MultiDiGraph:
    """Build a :class:`networkx.MultiDiGraph` from the entity / deal tables.

    Parameters
    ----------
    session:
        Open :class:`AsyncSession`. The caller owns its lifecycle.
    sector:
        Restrict to entities whose ``sector_tags`` contain ``sector``.
        Filtered Python-side (matching :meth:`EntityRepository.list_by_sector`).
    as_of:
        Only include deals announced on or before this date. Deals with
        no ``announced_at`` are excluded when ``as_of`` is set, since we
        can't place them in time.
    entity_ids:
        Restrict to this exact set of entities. Takes precedence over
        ``sector`` (they aren't intended to be combined).
    expand_transitively:
        When ``True`` (default) and a ``sector`` or ``entity_ids`` filter
        is set, BFS-expand the seed set through the deal graph so every
        node reachable from the seed (until leaves or cycles) is
        included. Without this, edges with one endpoint outside the
        sector / id filter are silently dropped — which is what
        truncated the cash chain past discovered entities pre-V1.9.1.
        Set to ``False`` for the strict closed-world view.
    """
    # 1. Resolve the seed set from the user-provided filter.
    seed_ids: set[uuid.UUID] | None
    if entity_ids is not None:
        ids_set = set(entity_ids)
        if not ids_set:
            return nx.MultiDiGraph()
        seed_ids = ids_set
    elif sector is not None:
        ent_stmt = select(Entity)
        ent_result = await session.execute(ent_stmt)
        seed_ids = {e.id for e in ent_result.scalars().all() if sector in e.sector_tags}
        if not seed_ids:
            return nx.MultiDiGraph()
    else:
        seed_ids = None  # No filter at all.

    # 2. Optionally expand the seed transitively through the deal graph
    #    so chains don't get truncated at the filter boundary.
    if seed_ids is not None and expand_transitively:
        seed_ids = await _expand_transitively(session, seed_ids, as_of=as_of)

    # 3. Load the entity rows for whatever set we ended up with.
    if seed_ids is None:
        ent_stmt = select(Entity)
        ent_result = await session.execute(ent_stmt)
        entities: list[Entity] = list(ent_result.scalars().all())
    else:
        ent_stmt = select(Entity).where(col(Entity.id).in_(seed_ids))
        ent_result = await session.execute(ent_stmt)
        entities = list(ent_result.scalars().all())

    entity_id_set = {e.id for e in entities}

    graph = nx.MultiDiGraph()
    for entity in entities:
        graph.add_node(entity.id, **_entity_node_attrs(entity))

    if not entity_id_set:
        return graph

    # 4. Load deals where BOTH endpoints are in the loaded set. After
    #    transitive expansion this is no longer a truncation risk —
    #    every counterparty reachable from the seed is in the set.
    deal_stmt = (
        select(Deal)
        .where(col(Deal.from_entity_id).in_(entity_id_set))
        .where(col(Deal.to_entity_id).in_(entity_id_set))
    )
    if as_of is not None:
        deal_stmt = deal_stmt.where(col(Deal.announced_at).is_not(None)).where(
            col(Deal.announced_at) <= as_of,
        )
    deal_result = await session.execute(deal_stmt)
    deals: list[Deal] = list(deal_result.scalars().all())

    for deal in deals:
        graph.add_edge(
            deal.from_entity_id,
            deal.to_entity_id,
            key=str(deal.id),
            **_deal_edge_attrs(deal),
        )

    return graph


def aggregate_by_pair(graph: nx.MultiDiGraph) -> nx.DiGraph:
    """Collapse parallel edges between each ``(from, to)`` pair.

    Returns a :class:`networkx.DiGraph` whose edges carry:

    - ``amount_usd``: ``float`` sum of underlying deal amounts (``None``
      values are skipped). The result is ``None`` when *every* underlying
      edge has ``amount_usd is None``.
    - ``deal_count``: ``int`` total number of underlying deals on this
      pair, including those with ``amount_usd is None``.

    Useful for Sankey-style summaries where parallel deals between the
    same two entities should appear as a single thicker arrow.
    """
    aggregated = nx.DiGraph()
    for node, attrs in graph.nodes(data=True):
        aggregated.add_node(node, **attrs)

    pair_totals: dict[tuple[object, object], list[float | None]] = {}
    for u, v, data in graph.edges(data=True):
        pair_totals.setdefault((u, v), []).append(data.get("amount_usd"))

    for (u, v), amounts in pair_totals.items():
        non_null = [a for a in amounts if a is not None]
        total: float | None = sum(non_null) if non_null else None
        aggregated.add_edge(u, v, amount_usd=total, deal_count=len(amounts))

    return aggregated
