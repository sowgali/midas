"""Interactive HTML rendering for the deal graph.

:func:`render_pyvis` writes a self-contained ``.html`` file (CDN assets
inlined) suitable for sharing without a web server. :func:`aggregate_to_sankey`
emits a JSON-shaped dict that maps cleanly onto Plotly's Sankey trace —
we don't import plotly here (it isn't a dep) so callers stay free to
choose a renderer.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, cast

import networkx as nx
from pyvis.network import Network

from .builder import aggregate_by_pair

# A small fixed palette keyed by ``Entity.entity_type`` string. Anything
# we haven't mapped falls back to ``_DEFAULT_NODE_COLOR``.
_NODE_COLORS: dict[str, str] = {
    "public_company": "#2b7bba",  # blue
    "private_company": "#e08e3a",  # orange
    "fund": "#3aa55c",  # green
    "government": "#7a7a7a",  # grey
    "nonprofit": "#8a4ba8",  # purple
}
_DEFAULT_NODE_COLOR = "#bdbdbd"

# Short labels for ``DealType`` so edge labels stay legible at zoom.
_DEAL_TYPE_ABBREV: dict[str, str] = {
    "investment": "INV",
    "acquisition": "ACQ",
    "commercial_contract": "CC",
    "partnership": "PART",
    "licensing": "LIC",
    "debt": "DEBT",
    "grant": "GRANT",
}


def _format_usd(amount: float | None) -> str:
    """Human-friendly USD formatter: ``$1.5M`` / ``$500M`` / ``$10.0B``."""
    if amount is None:
        return ""
    abs_amt = abs(amount)
    if abs_amt >= 1_000_000_000:
        return f"${amount / 1_000_000_000:.1f}B"
    if abs_amt >= 1_000_000:
        return f"${amount / 1_000_000:.1f}M"
    if abs_amt >= 1_000:
        return f"${amount / 1_000:.1f}K"
    return f"${amount:.0f}"


def _edge_width(amount: float | None) -> float:
    """Map ``amount_usd`` to a pyvis edge width via ``log10``, clamped to [1, 10].

    ``None`` amounts get a small constant (``1.0``) so they're still
    visible without dominating the view.
    """
    if amount is None or amount <= 0:
        return 1.0
    # log10($1M) = 6, log10($1B) = 9. Subtract a baseline so $1M maps near 1.
    width = math.log10(amount) - 5
    return max(1.0, min(10.0, width))


def render_pyvis(
    graph: nx.MultiDiGraph,
    output_path: Path,
    *,
    title: str = "midas",
) -> Path:
    """Render ``graph`` to a self-contained interactive HTML file.

    Returns the path that was written.
    """
    net = Network(
        directed=True,
        notebook=False,
        cdn_resources="in_line",
        heading=title,
    )

    for node, attrs in graph.nodes(data=True):
        canonical_name = cast(str, attrs.get("canonical_name", str(node)))
        entity_type = cast(str, attrs.get("entity_type", ""))
        ticker = attrs.get("ticker")
        sector_tags = attrs.get("sector_tags") or []

        tooltip_lines = [canonical_name, f"type: {entity_type}"]
        if ticker:
            tooltip_lines.append(f"ticker: {ticker}")
        if sector_tags:
            tooltip_lines.append(f"sectors: {', '.join(sector_tags)}")

        net.add_node(
            str(node),
            label=canonical_name,
            title="\n".join(tooltip_lines),
            color=_NODE_COLORS.get(entity_type, _DEFAULT_NODE_COLOR),
        )

    for u, v, data in graph.edges(data=True):
        amount = cast("float | None", data.get("amount_usd"))
        deal_type = cast(str, data.get("deal_type", ""))
        deal_abbrev = _DEAL_TYPE_ABBREV.get(deal_type, deal_type[:4].upper())
        amount_str = _format_usd(amount)
        edge_label = f"{amount_str} {deal_abbrev}".strip()

        announced = data.get("announced_at")
        description = cast(str, data.get("description", ""))
        tooltip = description
        if announced:
            tooltip = f"{tooltip}\nannounced: {announced}" if tooltip else f"announced: {announced}"

        net.add_edge(
            str(u),
            str(v),
            label=edge_label,
            title=tooltip,
            width=_edge_width(amount),
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    net.write_html(str(output_path), open_browser=False, notebook=False)
    return output_path


def aggregate_to_sankey(graph: nx.MultiDiGraph) -> dict[str, list[dict[str, Any]]]:
    """Return a Plotly-Sankey-compatible ``{"nodes": [...], "links": [...]}`` dict.

    Built from :func:`aggregate_by_pair`, so parallel deals between the
    same pair sum into one link (its ``value`` is the summed
    ``amount_usd``; ``deal_count`` is preserved). Links with a ``None``
    aggregated amount are skipped — Plotly Sankey requires numeric
    ``value`` per link.
    """
    aggregated = aggregate_by_pair(graph)

    node_index: dict[Any, int] = {}
    nodes: list[dict[str, Any]] = []
    for idx, (node, attrs) in enumerate(aggregated.nodes(data=True)):
        node_index[node] = idx
        nodes.append(
            {
                "id": str(node),
                "label": attrs.get("canonical_name", str(node)),
                "entity_type": attrs.get("entity_type"),
            },
        )

    links: list[dict[str, Any]] = []
    for u, v, data in aggregated.edges(data=True):
        amount = data.get("amount_usd")
        if amount is None:
            continue
        links.append(
            {
                "source": node_index[u],
                "target": node_index[v],
                "value": amount,
                "deal_count": data.get("deal_count", 1),
            },
        )

    return {"nodes": nodes, "links": links}
