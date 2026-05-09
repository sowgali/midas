"""Generic RSS / Atom feed reader.

Tier-3 sources (official company blogs, project announcements) often
expose an RSS or Atom feed. Rather than maintain per-site selectors for
the *index*, we hand the feed bytes to ``feedparser`` (which understands
both formats and a long tail of malformed feeds) and only fall back to
HTML parsing for the *article* body.

For article-body extraction we don't have per-site selectors — these are
arbitrary blogs — so we apply a simple heuristic:

1. Drop noise tags (``<script>``, ``<style>``, ``<nav>``, ``<header>``,
   ``<footer>``, ``<form>``).
2. Prefer the longest ``<article>`` if any are present.
3. Else prefer the longest ``<main>``.
4. Else fall back to ``<body>``.
5. Concatenate paragraph text with ``"\n\n"``.

This is deliberately conservative; the goal is "no nav junk, no JS,
keep paragraphs distinct" — extraction (and any per-site override) is
the next layer's job.
"""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime
from time import struct_time
from typing import Any
from uuid import UUID

import feedparser
from pydantic import BaseModel, ConfigDict
from selectolax.parser import HTMLParser, Node

from ..models.types import SourceType
from .base import RawDocument, Source
from .http_client import HttpClient, get_default_client

logger = logging.getLogger(__name__)

# Tags whose text content is structurally never article prose.
_NOISE_TAGS: tuple[str, ...] = ("script", "style", "nav", "header", "footer", "form")


class RssItem(BaseModel):
    """One entry off an RSS / Atom feed."""

    model_config = ConfigDict(frozen=True)

    url: str
    title: str
    published_at: date | None = None
    summary: str | None = None


class RssFeed(Source):
    """Generic RSS / Atom adapter.

    ``feedparser`` is sync, but it parses an in-memory ``bytes`` payload
    in microseconds — calling it from inside the async :meth:`list_items`
    is fine. The network fetch itself goes through the shared
    :class:`HttpClient` so we still get caching + rate limiting.
    """

    def __init__(
        self,
        *,
        entity_id: UUID,
        feed_url: str,
        publisher: str,
        source_type: SourceType = SourceType.BLOG,
        http_client: HttpClient | None = None,
    ) -> None:
        self.entity_id = entity_id
        self.feed_url = feed_url
        self.publisher = publisher
        self.source_type = source_type
        self._http = http_client

    async def _client(self) -> HttpClient:
        if self._http is not None:
            return self._http
        return await get_default_client()

    # ---------- list_items ----------

    async def list_items(self, *, since: date | None = None) -> list[RssItem]:
        """Fetch the feed and return its entries as :class:`RssItem`s."""
        client = await self._client()
        body = await client.get_bytes(self.feed_url)
        parsed = feedparser.parse(body)

        out: list[RssItem] = []
        for entry in parsed.entries:
            url = entry.get("link") if hasattr(entry, "get") else None
            title = entry.get("title") if hasattr(entry, "get") else None
            if not url or not title:
                logger.warning(
                    "blog_rss: skipping entry in %s missing link/title (link=%r, title=%r)",
                    self.feed_url,
                    url,
                    title,
                )
                continue

            published_at = _entry_date(entry)
            if since is not None and published_at is not None and published_at < since:
                continue

            summary_raw = entry.get("summary") if hasattr(entry, "get") else None
            summary = str(summary_raw) if summary_raw else None

            out.append(
                RssItem(
                    url=str(url),
                    title=str(title),
                    published_at=published_at,
                    summary=summary,
                )
            )
        return out

    # ---------- fetch_article ----------

    async def fetch_article(self, item: RssItem) -> RawDocument:
        """Fetch one article URL, extract prose with the safe default heuristic."""
        client = await self._client()
        html = await client.get_text(item.url)
        body_text = extract_article_text(html)
        content_bytes = body_text.encode("utf-8")

        published_at_dt: datetime | None = (
            datetime.combine(item.published_at, datetime.min.time(), tzinfo=UTC)
            if item.published_at is not None
            else None
        )

        return RawDocument(
            url=item.url,
            content_bytes=content_bytes,
            source_type=self.source_type,
            publisher=self.publisher,
            title=item.title,
            published_at=published_at_dt,
        )

    # ---------- Source ABC ----------

    async def fetch(self, *args: Any, **kwargs: Any) -> RawDocument:
        """Adapter for the :class:`Source` ABC; delegates to :meth:`fetch_article`."""
        item = args[0] if args else kwargs["item"]
        if not isinstance(item, RssItem):
            raise TypeError(f"expected RssItem, got {type(item).__name__}")
        return await self.fetch_article(item)


# ---------- helpers ----------


def _entry_date(entry: Any) -> date | None:
    """Best-effort extract a ``date`` from a feedparser entry.

    Prefers ``published_parsed``, falls back to ``updated_parsed``. Both
    are ``time.struct_time`` (UTC) when present.
    """
    for attr in ("published_parsed", "updated_parsed"):
        val = entry.get(attr) if hasattr(entry, "get") else None
        if isinstance(val, struct_time):
            return date(val.tm_year, val.tm_mon, val.tm_mday)
    return None


def extract_article_text(html: str) -> str:
    """Pull the article prose out of an arbitrary HTML page.

    See module docstring for the heuristic. Always returns a string;
    falls back to the empty string if even ``<body>`` is absent (e.g. an
    XML fragment).
    """
    tree = HTMLParser(html)

    # Drop noise tags from the entire tree before measuring lengths.
    for tag in _NOISE_TAGS:
        for node in tree.css(tag):
            node.decompose()

    # Pick the densest container available.
    container = _longest_by_text(tree.css("article"))
    if container is None:
        container = _longest_by_text(tree.css("main"))
    if container is None:
        container = tree.css_first("body")
    if container is None:
        return ""

    return _join_paragraphs(container)


def _longest_by_text(nodes: list[Node]) -> Node | None:
    if not nodes:
        return None
    return max(nodes, key=lambda n: len(n.text(strip=True)))


def _join_paragraphs(container: Node) -> str:
    """Collect paragraph-ish text from ``container``, preserving paragraph breaks.

    Strategy: gather ``<p>`` and ``<li>`` text first; if there are none,
    fall back to the whole container's text. Empty fragments are dropped.
    """
    pieces: list[str] = []
    for tag in ("p", "li"):
        for node in container.css(tag):
            text = node.text(strip=True)
            if text:
                pieces.append(text)

    if pieces:
        return "\n\n".join(pieces)

    text = container.text(strip=True)
    return text
