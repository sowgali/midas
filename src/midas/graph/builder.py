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


async def build_graph(
    session: AsyncSession,
    *,
    sector: str | None = None,
    as_of: date | None = None,
    entity_ids: Iterable[uuid.UUID] | None = None,
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
    """
    # 1. Load the candidate entities.
    if entity_ids is not None:
        ids_set = set(entity_ids)
        if not ids_set:
            return nx.MultiDiGraph()
        ent_stmt = select(Entity).where(col(Entity.id).in_(ids_set))
        ent_result = await session.execute(ent_stmt)
        entities: list[Entity] = list(ent_result.scalars().all())
    else:
        ent_stmt = select(Entity)
        ent_result = await session.execute(ent_stmt)
        all_entities = list(ent_result.scalars().all())
        if sector is not None:
            entities = [e for e in all_entities if sector in e.sector_tags]
        else:
            entities = all_entities

    entity_id_set = {e.id for e in entities}

    graph = nx.MultiDiGraph()
    for entity in entities:
        graph.add_node(entity.id, **_entity_node_attrs(entity))

    if not entity_id_set:
        return graph

    # 2. Load deals where BOTH endpoints are in the loaded set.
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
