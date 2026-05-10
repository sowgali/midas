"""Unit tests for :mod:`midas.insights`.

Pure-graph tests — we don't touch the DB; we hand-construct
:class:`networkx.MultiDiGraph` instances and assert the ranking / BFS
shape. The CLI wrappers are exercised separately in
``tests/test_cli.py``.
"""

from __future__ import annotations

import uuid
from decimal import Decimal

import networkx as nx
import pytest

from midas.insights import (
    ChainEdge,
    inflow_ranking,
    outbound_chain,
)


def _node(name: str) -> uuid.UUID:
    """Stable-uuid placeholder so we can name nodes in tests readably."""
    return uuid.uuid5(uuid.NAMESPACE_OID, name)


def _make_graph(
    nodes: list[str],
    edges: list[tuple[str, str, float | None, str]],
) -> nx.MultiDiGraph:
    """Build a multigraph from human-readable fixture data.

    ``edges`` is a list of ``(from_name, to_name, amount_usd, deal_type)``
    tuples. ``amount_usd=None`` is allowed (matches the partnership case).
    """
    g: nx.MultiDiGraph = nx.MultiDiGraph()
    for n in nodes:
        g.add_node(_node(n), canonical_name=n, entity_type="public_company", sector_tags=[])
    for i, (u, v, amt, dt) in enumerate(edges):
        g.add_edge(
            _node(u),
            _node(v),
            key=f"deal-{i}",
            amount_usd=amt,
            deal_type=dt,
            description=f"{u}→{v} {dt}",
        )
    return g


# ---------- inflow_ranking ----------


def test_inflow_ranks_recipients_by_total_dollars_from_payer_set() -> None:
    g = _make_graph(
        nodes=["Anthropic", "OpenAI", "AWS", "Broadcom", "SpaceX"],
        edges=[
            ("Anthropic", "AWS", 100e9, "commercial_contract"),
            ("Anthropic", "Broadcom", None, "commercial_contract"),
            ("Anthropic", "SpaceX", None, "commercial_contract"),
            ("OpenAI", "AWS", 5e9, "commercial_contract"),
        ],
    )
    rows = inflow_ranking(g, payer_ids={_node("Anthropic"), _node("OpenAI")})

    by_name = {r.canonical_name: r for r in rows}
    # AWS gets paid by both labs.
    assert by_name["AWS"].total_usd == Decimal(str(105e9))
    assert by_name["AWS"].deal_count == 2
    assert by_name["AWS"].payers == ("Anthropic", "OpenAI")
    # Broadcom and SpaceX have no disclosed amount but still appear.
    assert by_name["Broadcom"].total_usd == Decimal("0")
    assert by_name["SpaceX"].total_usd == Decimal("0")
    # Ranking order: AWS first by dollar.
    assert rows[0].canonical_name == "AWS"


def test_inflow_excludes_payers_from_result_by_default() -> None:
    """Cross-flows between two payers shouldn't appear in the recipient list."""
    g = _make_graph(
        nodes=["Anthropic", "Amazon", "Fluidstack"],
        edges=[
            ("Amazon", "Anthropic", 5.3e9, "investment"),  # cross-flow: both are payers
            ("Anthropic", "Fluidstack", 50e9, "investment"),
        ],
    )
    rows = inflow_ranking(g, payer_ids={_node("Anthropic"), _node("Amazon")})
    names = {r.canonical_name for r in rows}
    assert names == {"Fluidstack"}  # Anthropic excluded as a recipient


def test_inflow_can_keep_cross_flows_with_flag() -> None:
    g = _make_graph(
        nodes=["Anthropic", "Amazon", "Fluidstack"],
        edges=[
            ("Amazon", "Anthropic", 5.3e9, "investment"),
            ("Anthropic", "Fluidstack", 50e9, "investment"),
        ],
    )
    rows = inflow_ranking(
        g,
        payer_ids={_node("Anthropic"), _node("Amazon")},
        exclude_payers_from_result=False,
    )
    by_name = {r.canonical_name: r for r in rows}
    assert "Anthropic" in by_name
    assert by_name["Anthropic"].total_usd == Decimal(str(5.3e9))


def test_inflow_empty_payer_set_returns_empty() -> None:
    g = _make_graph(
        nodes=["A", "B"],
        edges=[("A", "B", 1e6, "commercial_contract")],
    )
    assert inflow_ranking(g, payer_ids=set()) == []


def test_inflow_payer_with_no_outgoing_edges_returns_empty() -> None:
    g = _make_graph(
        nodes=["Anthropic", "AWS"],
        edges=[("AWS", "Anthropic", 5e9, "investment")],
    )
    rows = inflow_ranking(g, payer_ids={_node("Anthropic")})
    assert rows == []


# ---------- outbound_chain ----------


def test_chain_walks_two_hops_from_seed() -> None:
    g = _make_graph(
        nodes=["Anthropic", "AWS", "Vertiv", "Schneider"],
        edges=[
            ("Anthropic", "AWS", 100e9, "commercial_contract"),
            ("AWS", "Vertiv", 2e9, "commercial_contract"),
            ("AWS", "Schneider", 1e9, "commercial_contract"),
        ],
    )
    hops = outbound_chain(g, _node("Anthropic"), max_hops=2)

    assert len(hops) == 2
    # Hop 1: Anthropic → AWS only.
    assert [(e.from_name, e.to_name) for e in hops[0].edges] == [("Anthropic", "AWS")]
    # Hop 2: AWS → Vertiv (bigger), AWS → Schneider — sorted by amount desc.
    hop2_pairs = [(e.from_name, e.to_name, e.amount_usd) for e in hops[1].edges]
    assert hop2_pairs[0] == ("AWS", "Vertiv", Decimal(str(2e9)))
    assert hop2_pairs[1] == ("AWS", "Schneider", Decimal(str(1e9)))


def test_chain_respects_max_hops_cap() -> None:
    g = _make_graph(
        nodes=["A", "B", "C", "D"],
        edges=[
            ("A", "B", 1e9, "x"),
            ("B", "C", 1e9, "x"),
            ("C", "D", 1e9, "x"),
        ],
    )
    hops = outbound_chain(g, _node("A"), max_hops=2)
    assert len(hops) == 2
    # D not reached at depth 2.
    reached = {e.to_name for hop in hops for e in hop.edges}
    assert "D" not in reached
    assert reached == {"B", "C"}


def test_chain_terminates_on_cycle() -> None:
    """A ↔ B cycle should surface the back-edge but not loop forever."""
    g = _make_graph(
        nodes=["A", "B"],
        edges=[
            ("A", "B", 1e9, "x"),
            ("B", "A", 2e9, "x"),
        ],
    )
    hops = outbound_chain(g, _node("A"), max_hops=5)
    assert len(hops) == 2
    # Hop 1: A→B.
    assert [(e.from_name, e.to_name) for e in hops[0].edges] == [("A", "B")]
    # Hop 2: B→A (the back-edge surfaces as a cross-link).
    assert [(e.from_name, e.to_name) for e in hops[1].edges] == [("B", "A")]


def test_chain_seed_not_in_graph_returns_empty() -> None:
    g = _make_graph(nodes=["A"], edges=[])
    assert outbound_chain(g, _node("NotThere"), max_hops=3) == []


def test_chain_disclosed_total_skips_none_amounts() -> None:
    g = _make_graph(
        nodes=["X", "Y", "Z"],
        edges=[
            ("X", "Y", None, "partnership"),
            ("X", "Z", 1e9, "investment"),
        ],
    )
    hops = outbound_chain(g, _node("X"), max_hops=1)
    assert len(hops) == 1
    amounts = [e.amount_usd for e in hops[0].edges]
    # One disclosed, one None — both present.
    assert Decimal(str(1e9)) in amounts
    assert None in amounts


def test_chain_edge_dataclass_is_frozen() -> None:
    """ChainEdge is frozen so output rows are safely hashable."""
    e = ChainEdge(
        from_id=uuid.uuid4(),
        from_name="A",
        to_id=uuid.uuid4(),
        to_name="B",
        deal_type="investment",
        amount_usd=Decimal("1000000"),
        description="x",
    )
    with pytest.raises(Exception):  # noqa: B017 — dataclass FrozenInstanceError
        e.from_name = "C"  # type: ignore[misc]
