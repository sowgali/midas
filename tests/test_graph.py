"""In-memory tests for graph aggregation, queries, and viz.

These don't touch the database — :func:`midas.graph.builder.build_graph`
gets exercised separately in ``test_graph_builder.py``. Here we
hand-construct a tiny :class:`networkx.MultiDiGraph` so each helper can
be unit-tested without DB plumbing.
"""

from __future__ import annotations

import json
import uuid
from decimal import Decimal
from pathlib import Path

import networkx as nx
import pytest

from midas.graph import (
    aggregate_by_pair,
    aggregate_to_sankey,
    downstream,
    render_pyvis,
    total_inflow,
    total_outflow,
    upstream,
)

# Stable, hand-picked UUIDs make assertion failures readable.
A = uuid.UUID("00000000-0000-0000-0000-0000000000a0")
B = uuid.UUID("00000000-0000-0000-0000-0000000000b0")
C = uuid.UUID("00000000-0000-0000-0000-0000000000c0")
D = uuid.UUID("00000000-0000-0000-0000-0000000000d0")


def _node_attrs(name: str, *, entity_type: str = "public_company") -> dict[str, object]:
    return {
        "canonical_name": name,
        "entity_type": entity_type,
        "ticker": None,
        "sector_tags": ["ai"],
    }


@pytest.fixture
def sample_graph() -> nx.MultiDiGraph:
    """A → B (two parallel deals), A → C, B → D, plus one None-amount A → B edge.

    Topology::

        A -[$1M, $2M, None]-> B
        A -[$10M]-> C
        B -[$3M]-> D
    """
    g = nx.MultiDiGraph()
    g.add_node(A, **_node_attrs("Alpha Corp"))
    g.add_node(B, **_node_attrs("Beta LLC", entity_type="private_company"))
    g.add_node(C, **_node_attrs("Gamma Fund", entity_type="fund"))
    g.add_node(D, **_node_attrs("Delta Gov", entity_type="government"))

    deal_1 = uuid.uuid4()
    deal_2 = uuid.uuid4()
    deal_none = uuid.uuid4()
    deal_a_c = uuid.uuid4()
    deal_b_d = uuid.uuid4()

    common_edge = {
        "deal_type": "investment",
        "announced_at": "2024-01-01",
        "status": "closed",
        "confidence": 0.9,
        "description": "edge",
    }
    g.add_edge(A, B, key=str(deal_1), deal_id=deal_1, amount_usd=1_000_000.0, **common_edge)
    g.add_edge(A, B, key=str(deal_2), deal_id=deal_2, amount_usd=2_000_000.0, **common_edge)
    g.add_edge(A, B, key=str(deal_none), deal_id=deal_none, amount_usd=None, **common_edge)
    g.add_edge(A, C, key=str(deal_a_c), deal_id=deal_a_c, amount_usd=10_000_000.0, **common_edge)
    g.add_edge(B, D, key=str(deal_b_d), deal_id=deal_b_d, amount_usd=3_000_000.0, **common_edge)
    return g


# ---------- aggregate_by_pair ----------


def test_aggregate_by_pair_sums_parallel_edges_skipping_none(
    sample_graph: nx.MultiDiGraph,
) -> None:
    aggregated = aggregate_by_pair(sample_graph)

    assert isinstance(aggregated, nx.DiGraph)
    assert not aggregated.is_multigraph()
    # All four nodes preserved.
    assert set(aggregated.nodes) == {A, B, C, D}

    # A->B has 3 underlying deals (one None amount), summed amount = 3M.
    ab = aggregated.get_edge_data(A, B)
    assert ab["amount_usd"] == 3_000_000.0
    assert ab["deal_count"] == 3

    # A->C single deal, untouched.
    ac = aggregated.get_edge_data(A, C)
    assert ac["amount_usd"] == 10_000_000.0
    assert ac["deal_count"] == 1


def test_aggregate_by_pair_preserves_direction(sample_graph: nx.MultiDiGraph) -> None:
    aggregated = aggregate_by_pair(sample_graph)
    # B->A must not exist; only A->B.
    assert aggregated.has_edge(A, B)
    assert not aggregated.has_edge(B, A)


def test_aggregate_by_pair_counts_none_amounts_in_deal_count() -> None:
    g = nx.MultiDiGraph()
    g.add_node(A, **_node_attrs("A"))
    g.add_node(B, **_node_attrs("B"))
    # All three amounts are None → aggregated amount stays None,
    # but deal_count is 3.
    for _ in range(3):
        deal = uuid.uuid4()
        g.add_edge(A, B, key=str(deal), deal_id=deal, amount_usd=None)

    aggregated = aggregate_by_pair(g)
    edge = aggregated.get_edge_data(A, B)
    assert edge["amount_usd"] is None
    assert edge["deal_count"] == 3


# ---------- downstream / upstream ----------


def test_downstream_unbounded_walks_full_reachable_set(sample_graph: nx.MultiDiGraph) -> None:
    # From A: B and C direct, D via B.
    assert downstream(sample_graph, A) == {B, C, D}


def test_downstream_max_depth_one(sample_graph: nx.MultiDiGraph) -> None:
    assert downstream(sample_graph, A, max_depth=1) == {B, C}


def test_downstream_max_depth_zero_returns_empty(sample_graph: nx.MultiDiGraph) -> None:
    # Depth 0 means we don't even step out of the start.
    assert downstream(sample_graph, A, max_depth=0) == set()


def test_downstream_unknown_node_is_empty(sample_graph: nx.MultiDiGraph) -> None:
    assert downstream(sample_graph, uuid.uuid4()) == set()


def test_upstream_walks_reverse_edges(sample_graph: nx.MultiDiGraph) -> None:
    # D's upstream is B (direct) and A (through B).
    assert upstream(sample_graph, D) == {A, B}
    assert upstream(sample_graph, D, max_depth=1) == {B}


# ---------- total_inflow / total_outflow ----------


def test_total_inflow_uses_decimal_precision(sample_graph: nx.MultiDiGraph) -> None:
    inflow_b = total_inflow(sample_graph, B)
    # 1M + 2M; None edge skipped.
    assert isinstance(inflow_b, Decimal)
    assert inflow_b == Decimal("3000000")


def test_total_outflow_uses_decimal_precision(sample_graph: nx.MultiDiGraph) -> None:
    outflow_a = total_outflow(sample_graph, A)
    # 1M + 2M (None skipped) + 10M = 13M.
    assert isinstance(outflow_a, Decimal)
    assert outflow_a == Decimal("13000000")


def test_decimal_total_avoids_float_drift() -> None:
    """Confirms ``Decimal(str(...))`` reconstruction protects against float drift.

    ``0.1 + 0.2`` is famously ``0.30000000000000004`` as a float; the
    helper should give us an exact ``0.3`` because amounts are routed
    through ``Decimal(str(...))`` before summation.
    """
    g = nx.MultiDiGraph()
    g.add_node(A, **_node_attrs("A"))
    g.add_node(B, **_node_attrs("B"))
    d1, d2 = uuid.uuid4(), uuid.uuid4()
    g.add_edge(A, B, key=str(d1), amount_usd=0.1)
    g.add_edge(A, B, key=str(d2), amount_usd=0.2)
    assert total_outflow(g, A) == Decimal("0.3")


def test_total_inflow_for_unknown_node_is_zero(sample_graph: nx.MultiDiGraph) -> None:
    assert total_inflow(sample_graph, uuid.uuid4()) == Decimal("0")


# ---------- aggregate_to_sankey ----------


def test_aggregate_to_sankey_shape_and_indices(sample_graph: nx.MultiDiGraph) -> None:
    sankey = aggregate_to_sankey(sample_graph)
    assert set(sankey.keys()) == {"nodes", "links"}

    # All four nodes get an entry.
    labels = {n["label"] for n in sankey["nodes"]}
    assert labels == {"Alpha Corp", "Beta LLC", "Gamma Fund", "Delta Gov"}

    # Build a label -> index map to verify links.
    idx_by_label = {n["label"]: i for i, n in enumerate(sankey["nodes"])}

    # A->B link should have value = sum of non-None parallel amounts (3M).
    ab = next(
        link
        for link in sankey["links"]
        if link["source"] == idx_by_label["Alpha Corp"]
        and link["target"] == idx_by_label["Beta LLC"]
    )
    assert ab["value"] == 3_000_000.0
    assert ab["deal_count"] == 3

    # Source / target are valid indices into nodes.
    n_nodes = len(sankey["nodes"])
    for link in sankey["links"]:
        assert 0 <= link["source"] < n_nodes
        assert 0 <= link["target"] < n_nodes


def test_aggregate_to_sankey_skips_none_amount_links() -> None:
    """All-None amounts → link omitted (Plotly needs numeric ``value``)."""
    g = nx.MultiDiGraph()
    g.add_node(A, **_node_attrs("A"))
    g.add_node(B, **_node_attrs("B"))
    d = uuid.uuid4()
    g.add_edge(A, B, key=str(d), amount_usd=None)
    sankey = aggregate_to_sankey(g)
    assert sankey["links"] == []
    # JSON-roundtrippable.
    json.dumps(sankey)


# ---------- render_pyvis ----------


def test_render_pyvis_writes_self_contained_html(
    sample_graph: nx.MultiDiGraph,
    tmp_path: Path,
) -> None:
    out = tmp_path / "g.html"
    written = render_pyvis(sample_graph, out, title="test-graph")

    assert written == out
    assert out.exists()
    assert out.stat().st_size > 0

    html = out.read_text()
    # Each entity's canonical_name should appear in the HTML — exact pyvis
    # structure is a moving target, so we just check substring presence.
    for label in ("Alpha Corp", "Beta LLC", "Gamma Fund", "Delta Gov"):
        assert label in html, f"expected {label!r} in rendered HTML"


def test_render_pyvis_creates_parent_dirs(
    sample_graph: nx.MultiDiGraph,
    tmp_path: Path,
) -> None:
    nested = tmp_path / "nested" / "deeper" / "g.html"
    render_pyvis(sample_graph, nested)
    assert nested.exists()
