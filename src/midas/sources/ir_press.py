"""Per-company IR press release scraper.

Most company investor-relations sites publish press releases as a static
HTML index (a list of items with title + date + permalink) and a similarly
static article page. Rather than write one bespoke scraper per company, we
parametrise the bits that vary — the CSS selectors and the date format —
and reuse a single :class:`IrPress` adapter against an :class:`IrPressConfig`.

Two-stage flow, mirroring the pattern in :mod:`midas.sources.sec_edgar`:

1. :meth:`IrPress.list_items` fetches the index page and returns one
   :class:`IrPressItem` per row (URL + title + best-effort published date),
   filtered by ``since``.
2. :meth:`IrPress.fetch_article` fetches one article URL and returns a
   :class:`RawDocument` whose ``content_bytes`` is the *extracted prose*
   (utf-8) — downstream extractors get plain text, not raw HTML. The
   underlying HTML is still cached on disk by the shared
   :class:`HttpClient`.

Robustness rule: a single bad row never tanks the whole index. If a
selector returns nothing, the row is skipped with a warning. If the
``date_format`` parse fails, ``published_at`` is ``None`` and the row
still survives.
"""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime
from typing import Any
from urllib.parse import urljoin
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field
from selectolax.parser import HTMLParser, Node

from ..models.types import SourceType
from .base import RawDocument, Source
from .http_client import HttpClient, get_default_client

logger = logging.getLogger(__name__)


class IrPressConfig(BaseModel):
    """Static, per-company configuration for an IR press-release scraper."""

    model_config = ConfigDict(frozen=True)

    entity_id: UUID = Field(description="midas Entity UUID this feed belongs to.")
    publisher: str = Field(description="Display name of the publisher, e.g. 'Microsoft'.")
    index_url: str = Field(description="URL of the press-release-list page.")

    item_selector: str = Field(
        description="CSS selector for each press-release row, e.g. 'article.press-release'.",
    )
    link_selector: str = Field(description="Within an item, the <a> whose href is the article URL.")
    title_selector: str = Field(description="Within an item, the title element.")
    date_selector: str = Field(description="Within an item, the date text.")
    date_format: str = Field(description="strptime format for the date text, e.g. '%B %d, %Y'.")
    article_body_selector: str = Field(
        description="On the article page, selector(s) whose text() is the press-release body.",
    )
    link_base_url: str | None = Field(
        default=None,
        description="Base URL for resolving relative hrefs. Defaults to index_url.",
    )


class IrPressItem(BaseModel):
    """One row off the IR index page."""

    model_config = ConfigDict(frozen=True)

    url: str
    title: str
    published_at: date | None = None


def _node_text(node: Node | None) -> str:
    """``Node.text(strip=True)`` but tolerant of ``None``."""
    if node is None:
        return ""
    return node.text(strip=True)


def _first(node: Node, selector: str) -> Node | None:
    """``css_first`` that doesn't blow up on a degenerate selector."""
    try:
        return node.css_first(selector)
    except Exception:  # pragma: no cover — selectolax surfaces parse errors here
        return None


class IrPress(Source):
    """Configurable IR press-release scraper.

    Stateless apart from the :class:`IrPressConfig` it was built with and
    a reference to a shared :class:`HttpClient` (which owns the cache and
    the rate limiter).
    """

    def __init__(self, config: IrPressConfig, http_client: HttpClient | None = None) -> None:
        self._config = config
        self._http = http_client

    @property
    def config(self) -> IrPressConfig:
        return self._config

    async def _client(self) -> HttpClient:
        if self._http is not None:
            return self._http
        return await get_default_client()

    # ---------- list_items ----------

    async def list_items(self, *, since: date | None = None) -> list[IrPressItem]:
        """Fetch the index page and return its rows as :class:`IrPressItem`s."""
        cfg = self._config
        client = await self._client()
        html = await client.get_text(cfg.index_url)
        tree = HTMLParser(html)

        base = cfg.link_base_url or cfg.index_url
        items: list[IrPressItem] = []
        for row in tree.css(cfg.item_selector):
            try:
                item = self._parse_row(row, base=base)
            except Exception as exc:  # defensive — never let one row kill the index
                logger.warning("ir_press: skipping row in %s due to %s", cfg.index_url, exc)
                continue
            if item is None:
                continue
            if since is not None and item.published_at is not None and item.published_at < since:
                continue
            items.append(item)
        return items

    def _parse_row(self, row: Node, *, base: str) -> IrPressItem | None:
        cfg = self._config

        link_node = _first(row, cfg.link_selector)
        title_node = _first(row, cfg.title_selector)
        date_node = _first(row, cfg.date_selector)

        href = link_node.attributes.get("href") if link_node is not None else None
        if not href:
            logger.warning(
                "ir_press: row in %s missing link (selector=%r); skipping",
                cfg.index_url,
                cfg.link_selector,
            )
            return None

        title = _node_text(title_node)
        if not title:
            logger.warning(
                "ir_press: row in %s missing title (selector=%r); skipping",
                cfg.index_url,
                cfg.title_selector,
            )
            return None

        url = urljoin(base, href)

        published_at: date | None = None
        date_text = _node_text(date_node)
        if date_text:
            try:
                published_at = datetime.strptime(date_text, cfg.date_format).date()
            except ValueError as exc:
                logger.warning(
                    "ir_press: row in %s date parse failed (%r vs %r): %s",
                    cfg.index_url,
                    date_text,
                    cfg.date_format,
                    exc,
                )
                published_at = None
        else:
            logger.warning(
                "ir_press: row in %s missing date (selector=%r); leaving published_at=None",
                cfg.index_url,
                cfg.date_selector,
            )

        return IrPressItem(url=url, title=title, published_at=published_at)

    # ---------- fetch_article ----------

    async def fetch_article(self, item: IrPressItem) -> RawDocument:
        """Fetch one article URL, extract its prose, and wrap it in a RawDocument."""
        cfg = self._config
        client = await self._client()
        html = await client.get_text(item.url)
        tree = HTMLParser(html)

        body_text = _join_paragraphs(tree.css(cfg.article_body_selector))
        content_bytes = body_text.encode("utf-8")

        published_at_dt: datetime | None = (
            datetime.combine(item.published_at, datetime.min.time(), tzinfo=UTC)
            if item.published_at is not None
            else None
        )

        return RawDocument(
            url=item.url,
            content_bytes=content_bytes,
            source_type=SourceType.PRESS_RELEASE,
            publisher=cfg.publisher,
            title=item.title,
            published_at=published_at_dt,
        )

    # ---------- Source ABC ----------

    async def fetch(self, *args: Any, **kwargs: Any) -> RawDocument:
        """Adapter for the :class:`Source` ABC; delegates to :meth:`fetch_article`."""
        item = args[0] if args else kwargs["item"]
        if not isinstance(item, IrPressItem):
            raise TypeError(f"expected IrPressItem, got {type(item).__name__}")
        return await self.fetch_article(item)


def _join_paragraphs(nodes: list[Node]) -> str:
    """Concatenate ``node.text(strip=True)`` across nodes with blank lines.

    Empty / whitespace-only nodes are dropped. Blank lines between paragraphs
    survive into the final utf-8 bytes so downstream extractors can still see
    paragraph boundaries.
    """
    parts: list[str] = []
    for n in nodes:
        text = n.text(strip=True)
        if text:
            parts.append(text)
    return "\n\n".join(parts)
