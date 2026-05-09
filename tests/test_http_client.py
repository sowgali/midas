"""Tests for the shared rate-limited / cached HTTP client.

All HTTP is mocked with ``respx``; nothing in this file touches the real
network. The cache always lands under ``tmp_path``.
"""

from __future__ import annotations

import time
from pathlib import Path

import httpx
import pytest
import respx

from midas.sources.http_client import HttpClient

_URL = "https://example.test/resource"


@pytest.mark.asyncio
async def test_rate_limiter_actually_throttles(tmp_path: Path) -> None:
    """5 calls at limit=2/s should take roughly >= 2s of wall-clock."""
    with respx.mock(assert_all_called=False) as mock:
        mock.get(_URL).mock(return_value=httpx.Response(200, content=b"ok"))

        # different URLs to defeat the cache; we want the limiter, not the cache
        urls = [f"{_URL}?i={i}" for i in range(5)]
        for u in urls:
            mock.get(u).mock(return_value=httpx.Response(200, content=b"ok"))

        async with HttpClient(cache_dir=tmp_path, rate_per_sec=2.0) as client:
            t0 = time.monotonic()
            for u in urls:
                await client.get_bytes(u)
            elapsed = time.monotonic() - t0

        # at 2 req/s a 5-call run can't finish in under ~2s; tolerant lower
        # bound to stay flake-free under load
        assert elapsed >= 1.8, f"expected >= ~2s, got {elapsed:.3f}s"


@pytest.mark.asyncio
async def test_cache_hit_skips_second_network_call(tmp_path: Path) -> None:
    with respx.mock(assert_all_called=False) as mock:
        route = mock.get(_URL).mock(return_value=httpx.Response(200, content=b"hello"))

        async with HttpClient(cache_dir=tmp_path, rate_per_sec=8.0) as client:
            first = await client.get_bytes(_URL)
            second = await client.get_bytes(_URL)

        assert first == b"hello"
        assert second == b"hello"
        assert route.call_count == 1


@pytest.mark.asyncio
async def test_retries_on_500_then_succeeds(tmp_path: Path) -> None:
    with respx.mock(assert_all_called=False) as mock:
        responses = [
            httpx.Response(500, content=b"err"),
            httpx.Response(200, content=b"good"),
        ]
        route = mock.get(_URL).mock(side_effect=responses)

        async with HttpClient(cache_dir=tmp_path, rate_per_sec=8.0, max_attempts=3) as client:
            body = await client.get_bytes(_URL)

        assert body == b"good"
        assert route.call_count == 2


@pytest.mark.asyncio
async def test_get_text_and_get_json(tmp_path: Path) -> None:
    text_url = "https://example.test/page.html"
    json_url = "https://example.test/data.json"
    with respx.mock(assert_all_called=False) as mock:
        mock.get(text_url).mock(return_value=httpx.Response(200, content=b"<html>hi</html>"))
        mock.get(json_url).mock(return_value=httpx.Response(200, json={"a": 1, "b": [2, 3]}))
        async with HttpClient(cache_dir=tmp_path, rate_per_sec=8.0) as client:
            text = await client.get_text(text_url)
            data = await client.get_json(json_url)

    assert "<html>" in text
    assert data == {"a": 1, "b": [2, 3]}


@pytest.mark.asyncio
async def test_user_agent_header_is_sent(tmp_path: Path) -> None:
    with respx.mock(assert_all_called=False) as mock:
        route = mock.get(_URL).mock(return_value=httpx.Response(200, content=b"ok"))
        async with HttpClient(
            cache_dir=tmp_path, rate_per_sec=8.0, user_agent="midas-test agent@example.com"
        ) as client:
            await client.get_bytes(_URL)
        assert route.calls.last.request.headers["User-Agent"] == "midas-test agent@example.com"
