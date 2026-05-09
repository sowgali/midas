"""Source acquisition layer.

This package owns the *fetching* side of the pipeline: HTTP clients, rate
limiters, on-disk cache, and per-tier source adapters (SEC EDGAR, IR press,
RSS, etc.).

Public surface:

* :class:`RawDocument` — what every source returns to the parser layer.
* :class:`Source` — abstract base class for source adapters.
* :class:`HttpClient` — shared rate-limited / cached httpx wrapper.
* :class:`SecEdgar` — SEC EDGAR adapter (CIK lookup + filings index + fetch).
"""

from __future__ import annotations

from .base import RawDocument, Source
from .http_client import HttpClient, get_default_client
from .sec_edgar import FilingMetadata, SecEdgar

__all__ = [
    "FilingMetadata",
    "HttpClient",
    "RawDocument",
    "SecEdgar",
    "Source",
    "get_default_client",
]
