"""V1.9 Playwright-backed source for Cloudflare-fronted + JS-rendered sites.

OpenAI's news page is behind Cloudflare TLS-fingerprint checks; Anthropic's
news page is JS-rendered (the article list isn't in the static HTML). Both
defeat the lightweight ``httpx``-based :class:`midas.sources.ir_press.IrPress`
and :class:`midas.sources.blog_rss.RssFeed` adapters. This source spins up
a real Chromium via Playwright, executes JS, and returns the rendered DOM.

It's the third source-type in the IR registry alongside ``rss`` and
``ir_press``. Use it sparingly — Playwright is ~150x slower than httpx
for trivial pages — only for sites that don't yield to lighter approaches.

Lifecycle
---------
Browser launch is the expensive part (~1s on warm caches). The source is
an **async context manager** so the pipeline owns the browser across an
entire feed's worth of fetches:

    async with PlaywrightSource(config) as src:
        items = await src.list_items(since=...)
        for item in items:
            raw = await src.fetch_article(item)
            ...

One browser, many pages. Each fetch creates a fresh page context (cookie /
state isolation). Pipeline closes the browser at the end of the run.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, date, datetime
from types import TracebackType
from typing import TYPE_CHECKING, Any
from urllib.parse import urljoin

from pydantic import BaseModel, Field

from midas.models.types import SourceType
from midas.sources.base import RawDocument

if TYPE_CHECKING:
    from playwright.async_api import Browser, Playwright

log = logging.getLogger(__name__)

# Browser-realistic default. Playwright's own default UA still says
# "HeadlessChrome" which some sites flag.
_DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15"
)


class PlaywrightSourceConfig(BaseModel):
    """Per-site configuration for the Playwright adapter.

    Selectors operate on the **rendered** DOM (after JS hydration), not
    the static HTML. The expected pattern: each item on the index is
    represented by an anchor (often the entire card is wrapped in
    ``<a>``). ``item_selector`` returns those anchors; we read href +
    inner text / nested elements from each.
    """

    entity_id: uuid.UUID
    publisher: str
    index_url: str

    # CSS selector for each item anchor on the index page (e.g.
    # "a[href^='/news/']"). Each match yields one PlaywrightItem.
    item_selector: str

    # When None, the title is taken from the item's own ``inner_text``.
    # Provide a selector to dig deeper if the anchor wraps richer markup.
    title_selector: str | None = None

    # Date extraction is optional — many JS-rendered news indexes don't
    # surface dates on the cards. Leave as None and dates default to None
    # (the dedup pipeline handles missing dates fine).
    date_selector: str | None = None
    date_format: str | None = None

    # CSS selector for the article body on each article page.
    article_body_selector: str

    # Resolved against the index URL for relative hrefs.
    link_base_url: str | None = None

    user_agent: str = Field(default=_DEFAULT_USER_AGENT)
    # ms to wait after domcontentloaded before reading the DOM. Most JS
    # frameworks finish hydration within 2 seconds.
    wait_after_load_ms: int = 2500

    # Per-page navigation timeout. JS-rendered sites can be sluggish;
    # 30s gives plenty of headroom.
    navigation_timeout_ms: int = 30_000

    source_type: SourceType = SourceType.BLOG


@dataclass(frozen=True, slots=True)
class PlaywrightItem:
    """One item from a Playwright-rendered index."""

    url: str
    title: str
    published_at: date | None


class PlaywrightSource:
    """Headless-browser source. Use as an async context manager."""

    def __init__(self, config: PlaywrightSourceConfig) -> None:
        self._config = config
        self._pw: Playwright | None = None
        self._browser: Browser | None = None

    async def __aenter__(self) -> PlaywrightSource:
        # Imported lazily so importing this module is cheap when Playwright
        # isn't actually installed (matters for the test suite running
        # without `playwright install chromium`).
        from playwright.async_api import async_playwright

        self._pw = await async_playwright().start()
        self._browser = await self._pw.chromium.launch(headless=True)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._browser is not None:
            await self._browser.close()
            self._browser = None
        if self._pw is not None:
            await self._pw.stop()
            self._pw = None

    async def _new_page(self) -> Any:
        if self._browser is None:
            raise RuntimeError("PlaywrightSource not entered (use `async with`).")
        ctx = await self._browser.new_context(user_agent=self._config.user_agent)
        return await ctx.new_page()

    async def list_items(
        self,
        *,
        since: date | None = None,
    ) -> list[PlaywrightItem]:
        """Render the index page and return one :class:`PlaywrightItem` per match.

        Items missing a link or title are skipped with a warning. Items
        with a parseable date older than ``since`` are filtered out.
        Items with an unparseable / missing date are kept (we can't
        place them in time, and callers usually want the long tail).
        """
        page = await self._new_page()
        try:
            await page.goto(
                self._config.index_url,
                wait_until="domcontentloaded",
                timeout=self._config.navigation_timeout_ms,
            )
            await page.wait_for_timeout(self._config.wait_after_load_ms)

            anchors = await page.locator(self._config.item_selector).all()
            base = self._config.link_base_url or self._config.index_url

            items: list[PlaywrightItem] = []
            for anchor in anchors:
                href = await anchor.get_attribute("href")
                if not href:
                    continue
                url = urljoin(base, href)

                if self._config.title_selector:
                    sub = anchor.locator(self._config.title_selector).first
                    if await sub.count() == 0:
                        continue
                    title = (await sub.inner_text()).strip()
                else:
                    title = (await anchor.inner_text()).strip()
                if not title:
                    continue

                pub: date | None = None
                if self._config.date_selector and self._config.date_format:
                    try:
                        d_loc = anchor.locator(self._config.date_selector).first
                        if await d_loc.count() > 0:
                            pub = datetime.strptime(
                                (await d_loc.inner_text()).strip(),
                                self._config.date_format,
                            ).date()
                    except (ValueError, AttributeError) as exc:
                        log.warning("playwright.date_parse_failed url=%s err=%s", url, exc)

                if since is not None and pub is not None and pub < since:
                    continue

                items.append(PlaywrightItem(url=url, title=title, published_at=pub))
            return items
        finally:
            await page.context.close()

    async def fetch_article(self, item: PlaywrightItem) -> RawDocument:
        """Render one article page and return its body text as a RawDocument."""
        page = await self._new_page()
        try:
            await page.goto(
                item.url,
                wait_until="domcontentloaded",
                timeout=self._config.navigation_timeout_ms,
            )
            await page.wait_for_timeout(self._config.wait_after_load_ms)
            body_loc = page.locator(self._config.article_body_selector).first
            if await body_loc.count() == 0:
                # Fall back to <main>, then <body>.
                body_loc = page.locator("main").first
                if await body_loc.count() == 0:
                    body_loc = page.locator("body").first
            text = (await body_loc.inner_text()).strip()
            return RawDocument(
                url=item.url,
                # RawDocument auto-computes content_sha256 from content_bytes.
                content_bytes=text.encode("utf-8"),
                source_type=self._config.source_type,
                publisher=self._config.publisher,
                title=item.title,
                published_at=(
                    datetime.combine(item.published_at, datetime.min.time(), tzinfo=UTC)
                    if item.published_at
                    else None
                ),
            )
        finally:
            await page.context.close()
