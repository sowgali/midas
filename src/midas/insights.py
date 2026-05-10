"""Investment-decision helpers on top of the built graph.

This is the "where is the money actually going" layer. The graph
builder gives us a ``MultiDiGraph`` of disclosed flows; insights
rolls those flows up into the shapes a human reader can scan:

- :func:`inflow_ranking` — who gets paid the most by a set of payers
  (the labs, by default). The "Anthropic and OpenAI are spending
  $180B+ — who's catching it?" view.
- :func:`outbound_chain` — BFS one node outward, hop-by-hop, summing
  the dollar volume at each ring. Lets a reader walk "Anthropic →
  AWS → ??? → ???" and see how deep the disclosed chain reaches.

Both are pure: they take an already-built :class:`networkx.MultiDiGraph`
and a few parameters. The CLI wrappers in :mod:`midas.cli` are thin —
they build the graph, call these, and print.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from decimal import Decimal

import networkx as nx

# ---------- inflow ranking ----------


@dataclass(frozen=True, slots=True)
class InflowRow:
    """One row of the inflow ranking.

    ``deal_count`` is the number of distinct *deals* (edges) — useful
    so a reader can distinguish "$50B in 1 contract" from "$50B
    spread over 25 smaller commitments".
    """

    entity_id: uuid.UUID
    canonical_name: str
    total_usd: Decimal
    deal_count: int
    payers: tuple[str, ...]  # canonical_name of each distinct upstream payer


def inflow_ranking(
    graph: nx.MultiDiGraph,
    *,
    payer_ids: set[uuid.UUID],
    exclude_payers_from_result: bool = True,
) -> list[InflowRow]:
    """Rank entities by total inbound ``amount_usd`` from ``payer_ids``.

    Walks every edge whose source is in ``payer_ids`` and accumulates
    onto the *target* entity. Returns rows sorted by ``total_usd``
    descending; entities reached only by amount-less edges (the
    placeholder partnerships) still get a row with ``total_usd=0`` and
    sort below the disclosed ones.

    ``exclude_payers_from_result`` filters out rows whose own id is in
    ``payer_ids`` — i.e. the cross-flows *between* labs aren't shown
    in the ranking. Set to ``False`` if you want to surface those
    (Amazon → Anthropic, for example, when Anthropic is also a payer).
    """
    totals: dict[uuid.UUID, Decimal] = {}
    counts: dict[uuid.UUID, int] = {}
    payers_seen: dict[uuid.UUID, set[uuid.UUID]] = {}

    for u, v, data in graph.edges(data=True):
        if u not in payer_ids:
            continue
        if exclude_payers_from_result and v in payer_ids:
            continue
        totals[v] = totals.get(v, Decimal("0"))
        counts[v] = counts.get(v, 0) + 1
        payers_seen.setdefault(v, set()).add(u)
        amount = data.get("amount_usd")
        if amount is not None:
            totals[v] += Decimal(str(amount))

    rows: list[InflowRow] = []
    for entity_id, total in totals.items():
        node = graph.nodes.get(entity_id, {})
        name = node.get("canonical_name", str(entity_id))
        payer_names = tuple(
            sorted(
                graph.nodes.get(pid, {}).get("canonical_name", str(pid))
                for pid in payers_seen[entity_id]
            ),
        )
        rows.append(
            InflowRow(
                entity_id=entity_id,
                canonical_name=name,
                total_usd=total,
                deal_count=counts[entity_id],
                payers=payer_names,
            ),
        )
    rows.sort(key=lambda r: (r.total_usd, r.deal_count), reverse=True)
    return rows


# ---------- outbound chain ----------


@dataclass(frozen=True, slots=True)
class ChainHop:
    """One ring of a BFS expansion from a seed entity.

    ``hop`` is 1 for direct counterparties, 2 for their counterparties,
    etc. ``edges`` lists each disclosed deal at this hop with the
    upstream→downstream pair and ``amount_usd`` (None when not disclosed).
    """

    hop: int
    edges: list[ChainEdge]


@dataclass(frozen=True, slots=True)
class ChainEdge:
    from_id: uuid.UUID
    from_name: str
    to_id: uuid.UUID
    to_name: str
    deal_type: str
    amount_usd: Decimal | None
    description: str


def outbound_chain(
    graph: nx.MultiDiGraph,
    seed: uuid.UUID,
    *,
    max_hops: int = 3,
) -> list[ChainHop]:
    """BFS-walk outgoing edges from ``seed``, grouping edges by hop number.

    Each ``ChainHop`` lists the edges *crossing into* that ring — so
    hop=1 are deals where the payer is ``seed`` itself, hop=2 are deals
    where the payer is a hop-1 counterparty, and so on. Cycles are
    naturally terminated because BFS visits each node once.

    The hop's edge list is sorted by ``amount_usd`` descending so the
    biggest-flow lines surface first in CLI output.
    """
    if seed not in graph:
        return []

    visited: set[uuid.UUID] = {seed}
    frontier: set[uuid.UUID] = {seed}
    hops: list[ChainHop] = []

    for hop in range(1, max_hops + 1):
        next_frontier: set[uuid.UUID] = set()
        edges: list[ChainEdge] = []
        for u in frontier:
            for _u, v, data in graph.out_edges(u, data=True):
                if v in visited and v != u:
                    # Edge into an already-visited node — still surface it
                    # as a flow at this hop (cycle / cross-link), but don't
                    # advance the frontier through it.
                    pass
                else:
                    next_frontier.add(v)
                amount = data.get("amount_usd")
                amount_dec = Decimal(str(amount)) if amount is not None else None
                edges.append(
                    ChainEdge(
                        from_id=u,
                        from_name=graph.nodes[u].get("canonical_name", str(u)),
                        to_id=v,
                        to_name=graph.nodes[v].get("canonical_name", str(v)),
                        deal_type=str(data.get("deal_type", "")),
                        amount_usd=amount_dec,
                        description=str(data.get("description", "")),
                    ),
                )
        if not edges:
            break
        edges.sort(
            key=lambda e: (e.amount_usd or Decimal("-1")),
            reverse=True,
        )
        hops.append(ChainHop(hop=hop, edges=edges))
        next_frontier -= visited
        visited |= next_frontier
        if not next_frontier:
            break
        frontier = next_frontier

    return hops
