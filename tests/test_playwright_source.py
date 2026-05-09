"""Tests for the V1.9 Playwright source.

We don't spin up real Chromium in unit tests; instead we mock the
``playwright.async_api.async_playwright`` factory and assert the
adapter drives the resulting Page / Locator API correctly. Live
end-to-end smoke is covered by an opt-in integration test gated on
``MIDAS_TEST_PLAYWRIGHT=1``.
"""

from __future__ import annotations

import os
import uuid
from datetime import date, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml
from sqlmodel import SQLModel

from midas.models.types import SourceType
from midas.registry import (
    IrPressSourceConfig,
    RssSourceConfig,
    parse_ir_sources,
)
from midas.registry import (
    PlaywrightSourceConfig as YamlPlaywrightSourceConfig,
)
from midas.sources.playwright_source import (
    PlaywrightItem,
    PlaywrightSource,
    PlaywrightSourceConfig,
)

# Touch SQLModel so the import isn't flagged as unused — registry models
# transitively depend on the metadata being initialized.
_KEEP_IMPORT_SQLMODEL = SQLModel


# ---------- Config discriminator ----------


def test_yaml_loader_routes_playwright_type(tmp_path: Any) -> None:
    """The new ``playwright`` discriminator value parses into the right
    pydantic shape (alongside ``rss`` / ``ir_press``).
    """
    p = tmp_path / "ir.yaml"
    p.write_text(
        yaml.safe_dump(
            {
                "sources": [
                    {
                        "entity_canonical_name": "OpenAI",
                        "type": "playwright",
                        "publisher": "OpenAI News",
                        "index_url": "https://openai.com/news/",
                        "item_selector": "a[href^='/index/']",
                        "article_body_selector": "main",
                        "link_base_url": "https://openai.com",
                    },
                ],
            },
        ),
    )
    parsed = parse_ir_sources(p)
    assert len(parsed) == 1
    assert isinstance(parsed[0], YamlPlaywrightSourceConfig)
    cfg = parsed[0]
    assert cfg.index_url == "https://openai.com/news/"
    assert cfg.source_type == SourceType.BLOG
    # Optional fields default sensibly.
    assert cfg.title_selector is None
    assert cfg.wait_after_load_ms == 2500


def test_yaml_loader_keeps_other_types(tmp_path: Any) -> None:
    """Adding the playwright discriminator doesn't break rss / ir_press."""
    p = tmp_path / "ir.yaml"
    p.write_text(
        yaml.safe_dump(
            {
                "sources": [
                    {
                        "entity_canonical_name": "X",
                        "type": "rss",
                        "publisher": "X",
                        "feed_url": "https://example.com/feed.xml",
                    },
                    {
                        "entity_canonical_name": "Y",
                        "type": "ir_press",
                        "publisher": "Y",
                        "index_url": "https://example.com",
                        "item_selector": "a",
                        "link_selector": "a",
                        "title_selector": "h2",
                        "date_selector": "time",
                        "date_format": "%B %d, %Y",
                        "article_body_selector": "article",
                    },
                ],
            },
        ),
    )
    parsed = parse_ir_sources(p)
    assert len(parsed) == 2
    assert isinstance(parsed[0], RssSourceConfig)
    assert isinstance(parsed[1], IrPressSourceConfig)


def test_default_yaml_now_includes_openai_and_anthropic_via_playwright() -> None:
    """The shipped registry uses the playwright path for OpenAI + Anthropic."""
    by_name = {c.entity_canonical_name: c for c in parse_ir_sources()}
    for cn in ("OpenAI", "Anthropic"):
        assert cn in by_name, f"missing {cn}"
        assert isinstance(by_name[cn], YamlPlaywrightSourceConfig), (
            f"{cn} should be a Playwright source"
        )


# ---------- Mocked PlaywrightSource ----------


def _config(**overrides: Any) -> PlaywrightSourceConfig:
    base: dict[str, Any] = {
        "entity_id": uuid.uuid4(),
        "publisher": "Test Co",
        "index_url": "https://example.com/news/",
        "item_selector": "a[href^='/news/']",
        "article_body_selector": "article",
        "link_base_url": "https://example.com",
    }
    base.update(overrides)
    return PlaywrightSourceConfig(**base)


def _mock_anchor(href: str, title: str) -> MagicMock:
    a = MagicMock()
    a.get_attribute = AsyncMock(return_value=href)
    a.inner_text = AsyncMock(return_value=title)
    a.locator = MagicMock(return_value=MagicMock(count=AsyncMock(return_value=0)))
    return a


def _mock_locator_with_anchors(anchors: list[MagicMock]) -> MagicMock:
    loc = MagicMock()
    loc.all = AsyncMock(return_value=anchors)
    return loc


def _install_mock_playwright(
    monkeypatch: Any,
    *,
    index_anchors: list[MagicMock] | None = None,
    article_body: str | None = None,
) -> dict[str, MagicMock]:
    """Patch ``async_playwright`` to a deterministic mock browser tree.

    Returns a dict of the major mocks so individual tests can assert
    that goto / wait_for_timeout / inner_text were called as expected.
    """
    # Page: locator(...) returns either the index-anchor list or a body locator.
    body_loc = MagicMock(
        count=AsyncMock(return_value=1),
        inner_text=AsyncMock(return_value=article_body or "body text"),
        first=None,  # set below
    )
    body_loc.first = body_loc

    index_loc = _mock_locator_with_anchors(index_anchors or [])

    def _page_locator(sel: str) -> MagicMock:
        # Naive routing: if it matches the configured item_selector return
        # the anchor list; otherwise it's the article-body selector.
        if "[href" in sel:
            return index_loc
        return body_loc

    page = MagicMock()
    page.goto = AsyncMock()
    page.wait_for_timeout = AsyncMock()
    page.locator = MagicMock(side_effect=_page_locator)
    page.context = MagicMock(close=AsyncMock())

    context = MagicMock()
    context.new_page = AsyncMock(return_value=page)

    browser = MagicMock()
    browser.new_context = AsyncMock(return_value=context)
    browser.close = AsyncMock()

    chromium = MagicMock()
    chromium.launch = AsyncMock(return_value=browser)

    pw = MagicMock()
    pw.chromium = chromium
    pw.stop = AsyncMock()

    factory = MagicMock()
    factory.start = AsyncMock(return_value=pw)

    def fake_async_playwright() -> MagicMock:
        return factory

    # Patch where the source imports from.
    monkeypatch.setattr(
        "playwright.async_api.async_playwright",
        fake_async_playwright,
    )
    return {
        "page": page,
        "browser": browser,
        "factory": factory,
        "index_loc": index_loc,
        "body_loc": body_loc,
    }


@pytest.mark.asyncio
async def test_list_items_returns_url_and_title_per_anchor(monkeypatch: Any) -> None:
    anchors = [
        _mock_anchor("/news/foo", "Foo announces partnership"),
        _mock_anchor("/news/bar", "Bar raises Series C"),
    ]
    _install_mock_playwright(monkeypatch, index_anchors=anchors)

    cfg = _config()
    async with PlaywrightSource(cfg) as src:
        items = await src.list_items()

    assert len(items) == 2
    assert items[0] == PlaywrightItem(
        url="https://example.com/news/foo",
        title="Foo announces partnership",
        published_at=None,
    )
    assert items[1].url == "https://example.com/news/bar"


@pytest.mark.asyncio
async def test_list_items_filters_by_since_when_date_is_known(monkeypatch: Any) -> None:
    """If ``since`` is set, items with a parseable older date are dropped."""

    # Each anchor has a nested date element returning a parseable string.
    def make_anchor_with_date(href: str, title: str, date_text: str) -> MagicMock:
        a = MagicMock()
        a.get_attribute = AsyncMock(return_value=href)
        a.inner_text = AsyncMock(return_value=title)
        date_el = MagicMock(
            count=AsyncMock(return_value=1),
            inner_text=AsyncMock(return_value=date_text),
        )
        date_el.first = date_el

        def _locator(sel: str) -> MagicMock:
            return date_el

        a.locator = MagicMock(side_effect=_locator)
        return a

    anchors = [
        make_anchor_with_date("/news/old", "Old post", "January 5, 2024"),
        make_anchor_with_date("/news/new", "New post", "March 15, 2026"),
    ]
    _install_mock_playwright(monkeypatch, index_anchors=anchors)

    cfg = _config(date_selector="time", date_format="%B %d, %Y")
    async with PlaywrightSource(cfg) as src:
        items = await src.list_items(since=date(2025, 1, 1))

    assert [i.title for i in items] == ["New post"]
    assert items[0].published_at == date(2026, 3, 15)


@pytest.mark.asyncio
async def test_list_items_skips_anchors_missing_href_or_title(monkeypatch: Any) -> None:
    anchors = [
        _mock_anchor("", "Has title but no href"),
        _mock_anchor("/news/ok", "  "),  # only whitespace
        _mock_anchor("/news/good", "Real article"),
    ]
    _install_mock_playwright(monkeypatch, index_anchors=anchors)

    async with PlaywrightSource(_config()) as src:
        items = await src.list_items()

    assert [i.title for i in items] == ["Real article"]


@pytest.mark.asyncio
async def test_fetch_article_returns_raw_document(monkeypatch: Any) -> None:
    _install_mock_playwright(
        monkeypatch,
        article_body="Microsoft will invest $10 billion in OpenAI.",
    )

    item = PlaywrightItem(
        url="https://example.com/news/foo",
        title="Foo",
        published_at=date(2026, 3, 1),
    )
    async with PlaywrightSource(_config()) as src:
        raw = await src.fetch_article(item)

    assert raw.url == item.url
    assert raw.title == "Foo"
    assert raw.publisher == "Test Co"
    assert raw.source_type == SourceType.BLOG
    assert b"Microsoft will invest" in raw.content_bytes
    assert raw.content_sha256  # auto-computed
    assert raw.published_at is not None and raw.published_at.tzinfo is not None


@pytest.mark.asyncio
async def test_fetch_article_without_published_at(monkeypatch: Any) -> None:
    _install_mock_playwright(monkeypatch, article_body="x")
    item = PlaywrightItem(url="https://example.com/news/x", title="X", published_at=None)
    async with PlaywrightSource(_config()) as src:
        raw = await src.fetch_article(item)
    assert raw.published_at is None


@pytest.mark.asyncio
async def test_browser_lifecycle_closes_on_exit(monkeypatch: Any) -> None:
    handles = _install_mock_playwright(monkeypatch)
    async with PlaywrightSource(_config()):
        pass
    handles["browser"].close.assert_awaited()
    handles["factory"].start.assert_awaited()


# ---------- Optional live integration smoke ----------


@pytest.mark.skipif(
    os.environ.get("MIDAS_TEST_PLAYWRIGHT") != "1",
    reason="Live Playwright smoke; set MIDAS_TEST_PLAYWRIGHT=1 to run.",
)
@pytest.mark.asyncio
async def test_live_openai_news_index_returns_items() -> None:  # pragma: no cover
    cfg = PlaywrightSourceConfig(
        entity_id=uuid.uuid4(),
        publisher="OpenAI News",
        index_url="https://openai.com/news/",
        item_selector="a[href^='/index/']",
        article_body_selector="main",
        link_base_url="https://openai.com",
    )
    async with PlaywrightSource(cfg) as src:
        items = await src.list_items()
    assert len(items) > 0
    assert all(i.url.startswith("https://openai.com/index/") for i in items)


# Touch unused-but-conceptually-relevant imports.
_KEEP_IMPORT_DATETIME = datetime
_KEEP_IMPORT_UUID = uuid
