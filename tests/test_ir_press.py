"""Tests for the configurable IR press-release scraper.

Network is fully mocked with respx. Cache lives under ``tmp_path``.
Fixtures are inline so the test file is self-contained.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from uuid import uuid4

import httpx
import pytest
import respx

from midas.models.types import SourceType
from midas.sources.http_client import HttpClient
from midas.sources.ir_press import IrPress, IrPressConfig, IrPressItem

_INDEX_URL = "https://example.test/news/"
_BASE_URL = "https://example.test"
_PUBLISHER = "ExampleCo"

# Three rows. Two are recent, one is old. The middle row has a malformed
# date — we want it to survive with published_at=None.
_INDEX_HTML = """
<html><body>
  <ul class="press-list">
    <li class="news-item">
      <a class="title" href="/news/2024-09-01-launch">Big launch</a>
      <span class="date">September 1, 2024</span>
    </li>
    <li class="news-item">
      <a class="title" href="/news/2024-06-15-update">Mid-year update</a>
      <span class="date">not-a-real-date</span>
    </li>
    <li class="news-item">
      <a class="title" href="/news/2023-01-10-old">Old news</a>
      <span class="date">January 10, 2023</span>
    </li>
  </ul>
</body></html>
"""

# An index where one row is missing the link entirely.
_INDEX_HTML_BAD_ROW = """
<html><body>
  <ul class="press-list">
    <li class="news-item">
      <a class="title" href="/news/good">Good row</a>
      <span class="date">September 1, 2024</span>
    </li>
    <li class="news-item">
      <span class="date">September 2, 2024</span>
    </li>
  </ul>
</body></html>
"""

_ARTICLE_URL = "https://example.test/news/2024-09-01-launch"
_SENTINEL = "SENTINEL_PROSE_42"
_ARTICLE_HTML = f"""
<html><body>
  <header>nav junk should not appear</header>
  <article>
    <div class="press-body">
      <p>First paragraph announces the deal.</p>
      <p>{_SENTINEL} is right here in the body.</p>
      <p>Closing remark.</p>
    </div>
  </article>
</body></html>
"""


def _config(index_url: str = _INDEX_URL, link_base_url: str | None = None) -> IrPressConfig:
    return IrPressConfig(
        entity_id=uuid4(),
        publisher=_PUBLISHER,
        index_url=index_url,
        item_selector="li.news-item",
        link_selector="a.title",
        title_selector="a.title",
        date_selector="span.date",
        date_format="%B %d, %Y",
        article_body_selector="article .press-body p",
        link_base_url=link_base_url,
    )


@pytest.mark.asyncio
async def test_list_items_filters_by_since(tmp_path: Path) -> None:
    with respx.mock(assert_all_called=False) as mock:
        mock.get(_INDEX_URL).mock(
            return_value=httpx.Response(
                200, text=_INDEX_HTML, headers={"content-type": "text/html"}
            )
        )
        async with HttpClient(cache_dir=tmp_path, rate_per_sec=8.0) as client:
            scraper = IrPress(_config(), http_client=client)
            items = await scraper.list_items(since=date(2024, 1, 1))

    # The 2023 row drops out; the malformed-date row survives because we
    # can't compare an unknown date against ``since``.
    assert len(items) == 2
    titles = {it.title for it in items}
    assert titles == {"Big launch", "Mid-year update"}

    by_title = {it.title: it for it in items}
    assert by_title["Big launch"].published_at == date(2024, 9, 1)
    # malformed date -> None, but row survives
    assert by_title["Mid-year update"].published_at is None


@pytest.mark.asyncio
async def test_list_items_no_since_returns_all_with_dates(tmp_path: Path) -> None:
    with respx.mock(assert_all_called=False) as mock:
        mock.get(_INDEX_URL).mock(return_value=httpx.Response(200, text=_INDEX_HTML))
        async with HttpClient(cache_dir=tmp_path, rate_per_sec=8.0) as client:
            scraper = IrPress(_config(), http_client=client)
            items = await scraper.list_items()

    assert len(items) == 3
    by_title = {it.title: it for it in items}
    assert by_title["Big launch"].published_at == date(2024, 9, 1)
    assert by_title["Mid-year update"].published_at is None
    assert by_title["Old news"].published_at == date(2023, 1, 10)


@pytest.mark.asyncio
async def test_list_items_skips_row_with_missing_selector(tmp_path: Path) -> None:
    with respx.mock(assert_all_called=False) as mock:
        mock.get(_INDEX_URL).mock(return_value=httpx.Response(200, text=_INDEX_HTML_BAD_ROW))
        async with HttpClient(cache_dir=tmp_path, rate_per_sec=8.0) as client:
            scraper = IrPress(_config(), http_client=client)
            items = await scraper.list_items()

    # Only the row with both link and date should survive; the other is
    # silently dropped with a warning, no exception.
    assert len(items) == 1
    assert items[0].title == "Good row"


@pytest.mark.asyncio
async def test_relative_href_resolved_via_link_base_url(tmp_path: Path) -> None:
    with respx.mock(assert_all_called=False) as mock:
        mock.get(_INDEX_URL).mock(return_value=httpx.Response(200, text=_INDEX_HTML))
        async with HttpClient(cache_dir=tmp_path, rate_per_sec=8.0) as client:
            scraper = IrPress(_config(link_base_url=_BASE_URL), http_client=client)
            items = await scraper.list_items()

    assert all(it.url.startswith("https://example.test/news/") for it in items)
    urls = {it.url for it in items}
    assert "https://example.test/news/2024-09-01-launch" in urls


@pytest.mark.asyncio
async def test_relative_href_resolved_via_index_url_fallback(tmp_path: Path) -> None:
    with respx.mock(assert_all_called=False) as mock:
        mock.get(_INDEX_URL).mock(return_value=httpx.Response(200, text=_INDEX_HTML))
        async with HttpClient(cache_dir=tmp_path, rate_per_sec=8.0) as client:
            scraper = IrPress(_config(link_base_url=None), http_client=client)
            items = await scraper.list_items()

    # urljoin against _INDEX_URL ('.../news/') resolves '/news/x' against
    # the host root, yielding the same absolute URLs.
    by_title = {it.title: it for it in items}
    assert by_title["Big launch"].url == "https://example.test/news/2024-09-01-launch"


@pytest.mark.asyncio
async def test_fetch_article_returns_raw_document_with_extracted_prose(tmp_path: Path) -> None:
    item = IrPressItem(url=_ARTICLE_URL, title="Big launch", published_at=date(2024, 9, 1))

    with respx.mock(assert_all_called=False) as mock:
        mock.get(_ARTICLE_URL).mock(
            return_value=httpx.Response(
                200, text=_ARTICLE_HTML, headers={"content-type": "text/html"}
            )
        )
        async with HttpClient(cache_dir=tmp_path, rate_per_sec=8.0) as client:
            scraper = IrPress(_config(), http_client=client)
            doc = await scraper.fetch_article(item)

    text = doc.content_bytes.decode("utf-8")
    assert _SENTINEL in text
    # paragraph breaks survive
    assert "First paragraph announces the deal." in text
    assert "\n\n" in text
    # nav junk does not appear (article_body_selector targets the body p's)
    assert "nav junk" not in text

    assert doc.url == _ARTICLE_URL
    assert doc.publisher == _PUBLISHER
    assert doc.title == "Big launch"
    assert doc.source_type is SourceType.PRESS_RELEASE
    assert doc.content_sha256
    assert doc.published_at is not None
    assert doc.published_at.tzinfo is not None
