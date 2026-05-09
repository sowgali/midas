"""FastAPI routers, one per resource.

Each router is mounted under ``/api`` by :func:`midas.api.app.create_app`
so URL paths in this package stay relative.
"""

from __future__ import annotations

from . import deals, entities, graph

__all__ = ["deals", "entities", "graph"]
