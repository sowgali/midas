"""Source acquisition layer.

This package owns the *fetching* side of the pipeline: HTTP clients, rate
limiters, on-disk cache, and per-tier source adapters (SEC EDGAR, IR press,
RSS, etc.).

Public surface:

* :class:`RawDocument` — what every source returns to the parser layer.
* :class:`Source` — abstract base class for source adapters.
* :class:`HttpClient` — shared rate-limited / cached httpx wrapper.
* :class:`SecEdgar` — SEC EDGAR adapter (CIK lookup + filings index + fetch).
* :class:`IrPress` — configurable per-company IR press-release scraper.
* :class:`RssFeed` — generic RSS / Atom feed reader.
"""

from __future__ import annotations

from .base import RawDocument, Source
from .blog_rss import RssFeed, RssItem
from .http_client import HttpClient, get_default_client
from .ir_press import IrPress, IrPressConfig, IrPressItem
from .sec_edgar import FilingMetadata, SecEdgar

__all__ = [
    "FilingMetadata",
    "HttpClient",
    "IrPress",
    "IrPressConfig",
    "IrPressItem",
    "RawDocument",
    "RssFeed",
    "RssItem",
    "SecEdgar",
    "Source",
    "get_default_client",
]
