"""Tests for the generic RSS / Atom feed reader.

Network is fully mocked with respx. Both RSS 2.0 and Atom feeds are
covered; article-body extraction is exercised against an inline HTML
fixture with hostile noise tags.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from uuid import uuid4

import httpx
import pytest
import respx

from midas.models.types import SourceType
from midas.sources.blog_rss import RssFeed, RssItem
from midas.sources.http_client import HttpClient

_FEED_URL = "https://example.test/feed.xml"
_ATOM_URL = "https://example.test/atom.xml"
_PUBLISHER = "ExampleBlog"

# RSS 2.0: three entries. Two have valid <link> + <pubDate>, one is
# missing <link> (must be skipped).
_RSS_BYTES = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>ExampleBlog</title>
    <link>https://example.test/</link>
    <description>x</description>
    <item>
      <title>New product</title>
      <link>https://example.test/posts/new-product</link>
      <pubDate>Mon, 02 Sep 2024 10:00:00 GMT</pubDate>
      <description>A short summary.</description>
    </item>
    <item>
      <title>Older post</title>
      <link>https://example.test/posts/older</link>
      <pubDate>Tue, 10 Jan 2023 09:00:00 GMT</pubDate>
      <description>Old.</description>
    </item>
    <item>
      <title>Broken (no link)</title>
      <pubDate>Wed, 11 Sep 2024 12:00:00 GMT</pubDate>
      <description>Should be skipped.</description>
    </item>
  </channel>
</rss>
"""

_ATOM_BYTES = b"""<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>AtomBlog</title>
  <link href="https://example.test/atom"/>
  <updated>2024-09-05T10:00:00Z</updated>
  <id>urn:uuid:atom-feed</id>
  <entry>
    <title>Atom entry one</title>
    <link href="https://example.test/atom/one"/>
    <id>urn:uuid:one</id>
    <updated>2024-09-05T10:00:00Z</updated>
    <published>2024-09-05T10:00:00Z</published>
    <summary>Atom summary</summary>
  </entry>
</feed>
"""

_ARTICLE_URL = "https://example.test/posts/new-product"
_ARTICLE_SENTINEL = "ARTICLE_PROSE_SENTINEL"
_NAV_SENTINEL = "NAV_JUNK_SHOULD_BE_DROPPED"
_SCRIPT_SENTINEL = "SCRIPT_NOISE_SHOULD_BE_DROPPED"
_ARTICLE_HTML = f"""
<html><head><title>x</title>
  <script>var x = "{_SCRIPT_SENTINEL}";</script>
</head><body>
  <nav>{_NAV_SENTINEL} navigation links</nav>
  <header>masthead</header>
  <article>
    <p>First paragraph of real content.</p>
    <p>{_ARTICLE_SENTINEL} appears here.</p>
    <p>Closing paragraph.</p>
  </article>
  <footer>copyright</footer>
</body></html>
"""


@pytest.mark.asyncio
async def test_list_items_parses_rss_and_skips_missing_link(tmp_path: Path) -> None:
    with respx.mock(assert_all_called=False) as mock:
        mock.get(_FEED_URL).mock(
            return_value=httpx.Response(
                200, content=_RSS_BYTES, headers={"content-type": "application/rss+xml"}
            )
        )
        async with HttpClient(cache_dir=tmp_path, rate_per_sec=8.0) as client:
            feed = RssFeed(
                entity_id=uuid4(),
                feed_url=_FEED_URL,
                publisher=_PUBLISHER,
                http_client=client,
            )
            items = await feed.list_items()

    # Three entries in the feed but the broken one (no <link>) is skipped.
    assert len(items) == 2
    titles = {it.title for it in items}
    assert titles == {"New product", "Older post"}

    by_title = {it.title: it for it in items}
    assert by_title["New product"].published_at == date(2024, 9, 2)
    assert by_title["Older post"].published_at == date(2023, 1, 10)
    assert by_title["New product"].summary == "A short summary."


@pytest.mark.asyncio
async def test_list_items_since_filters_older(tmp_path: Path) -> None:
    with respx.mock(assert_all_called=False) as mock:
        mock.get(_FEED_URL).mock(return_value=httpx.Response(200, content=_RSS_BYTES))
        async with HttpClient(cache_dir=tmp_path, rate_per_sec=8.0) as client:
            feed = RssFeed(
                entity_id=uuid4(),
                feed_url=_FEED_URL,
                publisher=_PUBLISHER,
                http_client=client,
            )
            items = await feed.list_items(since=date(2024, 1, 1))

    assert len(items) == 1
    assert items[0].title == "New product"


@pytest.mark.asyncio
async def test_list_items_parses_atom(tmp_path: Path) -> None:
    with respx.mock(assert_all_called=False) as mock:
        mock.get(_ATOM_URL).mock(
            return_value=httpx.Response(
                200, content=_ATOM_BYTES, headers={"content-type": "application/atom+xml"}
            )
        )
        async with HttpClient(cache_dir=tmp_path, rate_per_sec=8.0) as client:
            feed = RssFeed(
                entity_id=uuid4(),
                feed_url=_ATOM_URL,
                publisher="AtomBlog",
                http_client=client,
            )
            items = await feed.list_items()

    assert len(items) == 1
    item = items[0]
    assert item.title == "Atom entry one"
    assert item.url == "https://example.test/atom/one"
    assert item.published_at == date(2024, 9, 5)


@pytest.mark.asyncio
async def test_fetch_article_drops_script_and_nav_keeps_article(tmp_path: Path) -> None:
    item = RssItem(
        url=_ARTICLE_URL,
        title="New product",
        published_at=date(2024, 9, 2),
        summary="A short summary.",
    )
    with respx.mock(assert_all_called=False) as mock:
        mock.get(_ARTICLE_URL).mock(
            return_value=httpx.Response(
                200, text=_ARTICLE_HTML, headers={"content-type": "text/html"}
            )
        )
        async with HttpClient(cache_dir=tmp_path, rate_per_sec=8.0) as client:
            feed = RssFeed(
                entity_id=uuid4(),
                feed_url=_FEED_URL,
                publisher=_PUBLISHER,
                http_client=client,
            )
            doc = await feed.fetch_article(item)

    text = doc.content_bytes.decode("utf-8")
    assert _ARTICLE_SENTINEL in text
    assert "First paragraph of real content." in text
    # script + nav must not survive
    assert _SCRIPT_SENTINEL not in text
    assert _NAV_SENTINEL not in text
    # paragraph breaks preserved
    assert "\n\n" in text

    assert doc.url == _ARTICLE_URL
    assert doc.publisher == _PUBLISHER
    assert doc.title == "New product"
    assert doc.source_type is SourceType.BLOG
    assert doc.content_sha256


@pytest.mark.asyncio
async def test_fetch_article_falls_back_to_main_then_body(tmp_path: Path) -> None:
    """When there's no <article>, the longest <main> wins."""
    url = "https://example.test/posts/main-only"
    html = """
    <html><body>
      <nav>NAV_DROP nav links</nav>
      <main>
        <p>Main paragraph one.</p>
        <p>MAIN_SENTINEL appears here.</p>
      </main>
    </body></html>
    """
    with respx.mock(assert_all_called=False) as mock:
        mock.get(url).mock(return_value=httpx.Response(200, text=html))
        async with HttpClient(cache_dir=tmp_path, rate_per_sec=8.0) as client:
            feed = RssFeed(
                entity_id=uuid4(),
                feed_url=_FEED_URL,
                publisher=_PUBLISHER,
                http_client=client,
            )
            doc = await feed.fetch_article(
                RssItem(url=url, title="t", published_at=None, summary=None)
            )

    text = doc.content_bytes.decode("utf-8")
    assert "MAIN_SENTINEL" in text
    assert "NAV_DROP" not in text
    # published_at None -> RawDocument.published_at None
    assert doc.published_at is None
