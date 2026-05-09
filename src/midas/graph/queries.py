"""Small graph-query helpers over a :class:`networkx.MultiDiGraph`.

These are pure-Python BFS / accumulator helpers. They take a graph that
was already built by :func:`midas.graph.builder.build_graph` (or
hand-constructed in tests) — there's no DB access here.

``total_inflow`` / ``total_outflow`` return :class:`decimal.Decimal` for
precision: although the graph stores ``amount_usd`` as ``float`` (so the
graph is JSON-serialisable), totals are reconstructed as ``Decimal`` via
``Decimal(str(float_value))`` so callers doing money math don't quietly
inherit float drift.
"""

from __future__ import annotations

import uuid
from collections import deque
from decimal import Decimal

import networkx as nx


def _bfs(graph: nx.MultiDiGraph, start: uuid.UUID, *, max_depth: int | None) -> set[uuid.UUID]:
    """BFS traversal of ``graph`` from ``start``, excluding the start node.

    ``max_depth`` of 1 means "direct neighbours only"; ``None`` means
    unbounded. Works on whatever direction ``graph`` already represents,
    so callers reverse the graph beforehand for upstream traversal.
    """
    if start not in graph:
        return set()

    visited: set[uuid.UUID] = {start}
    queue: deque[tuple[uuid.UUID, int]] = deque([(start, 0)])
    reached: set[uuid.UUID] = set()

    while queue:
        node, depth = queue.popleft()
        if max_depth is not None and depth >= max_depth:
            continue
        for neighbour in graph.successors(node):
            if neighbour in visited:
                continue
            visited.add(neighbour)
            reached.add(neighbour)
            queue.append((neighbour, depth + 1))

    return reached


def downstream(
    graph: nx.MultiDiGraph,
    entity_id: uuid.UUID,
    *,
    max_depth: int | None = None,
) -> set[uuid.UUID]:
    """Set of entity ids reachable by following outgoing edges from ``entity_id``.

    Excludes the start node itself. ``max_depth=1`` returns only direct
    payees; ``None`` (default) is unbounded.
    """
    return _bfs(graph, entity_id, max_depth=max_depth)


def upstream(
    graph: nx.MultiDiGraph,
    entity_id: uuid.UUID,
    *,
    max_depth: int | None = None,
) -> set[uuid.UUID]:
    """Set of entity ids that reach ``entity_id`` along directed edges.

    Implemented by traversing :meth:`networkx.MultiDiGraph.reverse` so
    the same BFS body can serve both directions.
    """
    reversed_graph: nx.MultiDiGraph = graph.reverse(copy=False)
    return _bfs(reversed_graph, entity_id, max_depth=max_depth)


def _sum_amounts(graph: nx.MultiDiGraph, edges: list[tuple[object, object, object]]) -> Decimal:
    """Sum ``amount_usd`` over ``edges``, skipping ``None``, returning ``Decimal``.

    Floats are routed through ``Decimal(str(...))`` so we don't inherit
    binary-float drift in the totals.
    """
    total = Decimal("0")
    for u, v, key in edges:
        data = graph.get_edge_data(u, v, key=key)
        if data is None:
            continue
        amount = data.get("amount_usd")
        if amount is None:
            continue
        total += Decimal(str(amount))
    return total


def total_inflow(graph: nx.MultiDiGraph, entity_id: uuid.UUID) -> Decimal:
    """Sum of ``amount_usd`` across edges *into* ``entity_id``.

    ``None`` amounts are skipped. Returns ``Decimal('0')`` if the entity
    has no incoming edges or isn't in the graph.
    """
    if entity_id not in graph:
        return Decimal("0")
    edges = [(u, v, k) for u, v, k in graph.in_edges(entity_id, keys=True)]
    return _sum_amounts(graph, edges)


def total_outflow(graph: nx.MultiDiGraph, entity_id: uuid.UUID) -> Decimal:
    """Sum of ``amount_usd`` across edges *out of* ``entity_id``. See :func:`total_inflow`."""
    if entity_id not in graph:
        return Decimal("0")
    edges = [(u, v, k) for u, v, k in graph.out_edges(entity_id, keys=True)]
    return _sum_amounts(graph, edges)
