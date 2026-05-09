"""Graph builder, queries, and visualization for midas.

The graph package converts the relational source-of-truth (entities and
deals in Postgres) into a :class:`networkx.MultiDiGraph` and renders it
to interactive HTML. Building is async because it pulls from the DB;
everything else (queries, aggregation, viz) is sync — graph operations
don't benefit from async, and keeping them sync keeps callers simple.
"""

from __future__ import annotations

from .builder import aggregate_by_pair, build_graph
from .queries import downstream, total_inflow, total_outflow, upstream
from .viz import aggregate_to_sankey, render_pyvis

__all__ = [
    "aggregate_by_pair",
    "aggregate_to_sankey",
    "build_graph",
    "downstream",
    "render_pyvis",
    "total_inflow",
    "total_outflow",
    "upstream",
]
