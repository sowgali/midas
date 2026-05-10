"""``/api/graph`` — built graph + aggregated edges as JSON.

The handler reuses :func:`midas.graph.builder.build_graph` and
:func:`midas.graph.builder.aggregate_by_pair`. Per-deal edge attributes
already exist on the :class:`networkx.MultiDiGraph` we get back, so we
walk the multigraph once to collect the per-pair ``deal_types`` set
(which the aggregated DiGraph drops), then walk the aggregated graph to
emit one :class:`GraphEdgeDto` per pair.

``entity_ids`` is parsed as a **comma-separated** list (``?entity_ids=a,b,c``)
rather than repeated query params; that's the convention chosen for this
slice and is what the frontend should send.
"""

from __future__ import annotations

import uuid
from datetime import date
from typing import Annotated

from fastapi import APIRouter, Query
from sqlmodel import col, select

from midas.graph.builder import aggregate_by_pair, build_graph
from midas.models import Entity

from ..deps import Session
from ..schemas import EntityDto, GraphEdgeDto, GraphResponse

router = APIRouter(prefix="/graph", tags=["graph"])


def _parse_entity_ids(raw: str | None) -> list[uuid.UUID] | None:
    """Split a comma-separated ``entity_ids`` query value into UUIDs.

    Empty / ``None`` / all-whitespace yields ``None`` (no filter).
    Unparseable tokens are skipped silently — the frontend should only
    ever send ids it got from us in the first place.
    """
    if raw is None:
        return None
    tokens = [t.strip() for t in raw.split(",") if t.strip()]
    if not tokens:
        return None
    parsed: list[uuid.UUID] = []
    for tok in tokens:
        try:
            parsed.append(uuid.UUID(tok))
        except ValueError:
            continue
    return parsed or None


@router.get("", response_model=GraphResponse)
async def get_graph(
    session: Session,
    sector: Annotated[
        str | None,
        Query(description="Restrict to entities tagged with this sector."),
    ] = None,
    as_of: Annotated[
        date | None,
        Query(
            description=(
                "ISO date — exclude deals announced after this date "
                "(and deals with no announcement date)."
            ),
        ),
    ] = None,
    entity_ids: Annotated[
        str | None,
        Query(description="Comma-separated entity ids; takes precedence over ``sector``."),
    ] = None,
    expand_transitively: Annotated[
        bool,
        Query(
            description=(
                "BFS-expand the seed set through deals so chains aren't truncated "
                "at the sector / entity_ids boundary. Defaults to True; set False "
                "for the strict closed-world view."
            ),
        ),
    ] = True,
) -> GraphResponse:
    """Build the cash-flow graph and return its nodes + aggregated edges."""
    parsed_ids = _parse_entity_ids(entity_ids)

    multigraph = await build_graph(
        session,
        sector=sector,
        as_of=as_of,
        entity_ids=parsed_ids,
        expand_transitively=expand_transitively,
    )

    # Per-pair distinct ``deal_type`` values (the aggregated DiGraph drops them).
    pair_types: dict[tuple[uuid.UUID, uuid.UUID], set[str]] = {}
    for u, v, data in multigraph.edges(data=True):
        deal_type = data.get("deal_type")
        if deal_type is None:
            continue
        pair_types.setdefault((u, v), set()).add(str(deal_type))

    aggregated = aggregate_by_pair(multigraph)

    # Re-fetch the entity rows for the nodes that ended up in the graph
    # so the wire DTO has every field (graph node attrs are a subset).
    node_ids = [node_id for node_id, _ in multigraph.nodes(data=True)]
    nodes: list[EntityDto] = []
    if node_ids:
        ent_stmt = select(Entity).where(col(Entity.id).in_(node_ids))
        ent_rows = list((await session.execute(ent_stmt)).scalars().all())
        nodes = [EntityDto.model_validate(e) for e in ent_rows]

    edges: list[GraphEdgeDto] = []
    for u, v, data in aggregated.edges(data=True):
        edges.append(
            GraphEdgeDto(
                from_id=str(u),
                to_id=str(v),
                total_amount_usd=data.get("amount_usd"),
                deal_count=int(data.get("deal_count", 0)),
                deal_types=sorted(pair_types.get((u, v), set())),
            ),
        )

    return GraphResponse(nodes=nodes, edges=edges, as_of=as_of, sector=sector)
