"""Tests for V1.9.2 BFS source-discovery.

Three layers covered:

1. Pure URL-generation tests (``derive_domain_candidates`` +
   ``feed_url_candidates``) — no network at all.
2. ``is_feed_response`` content sniffing — no network.
3. End-to-end ``discover_for_entity`` against a mocked
   ``httpx.AsyncClient`` so we can assert candidate-walk order +
   max-results truncation without touching real hosts.
"""

from __future__ import annotations

import uuid
from typing import Any

import httpx
import pytest

from midas.discovery.sources import (
    SourceCandidate,
    derive_domain_candidates,
    discover_for_entity,
    feed_url_candidates,
    is_discoverable_entity_name,
    is_feed_response,
    probe_feed,
)
from midas.models import Entity, EntityType

# ---------- derive_domain_candidates ----------


def test_domain_candidates_compact_and_hyphenated() -> None:
    assert derive_domain_candidates("Constellation Energy") == [
        "constellationenergy.com",
        "constellation-energy.com",
        "constellation.com",
    ]


def test_domain_candidates_strips_corporate_suffixes() -> None:
    """'Vertiv Holdings Co' → just 'vertiv.com' + holdings-compact variants."""
    candidates = derive_domain_candidates("Vertiv Holdings Co")
    # 'co' + 'holdings' stripped → only 'vertiv' remains.
    assert "vertiv.com" in candidates
    # Multi-token entries must NOT appear since we stripped everything else.
    assert all("holdings" not in c for c in candidates)


def test_domain_candidates_acronym_for_long_names() -> None:
    candidates = derive_domain_candidates("Hewlett Packard Enterprise Company")
    # 'company' stripped; first-letter acronym from remaining tokens = "hpe".
    assert "hpe.com" in candidates
    assert candidates[0].startswith("hewlett")


def test_domain_candidates_single_token() -> None:
    assert derive_domain_candidates("Anthropic") == ["anthropic.com"]


def test_domain_candidates_empty_input_returns_empty() -> None:
    assert derive_domain_candidates("") == []
    assert derive_domain_candidates("Inc.") == []  # all tokens stripped


def test_domain_candidates_dedupes() -> None:
    candidates = derive_domain_candidates("Acme Corp")
    # Compact "acme", hyphenated "acme", single-token "acme" all collapse.
    assert candidates.count("acme.com") == 1


# ---------- feed_url_candidates ----------


def test_feed_url_candidates_includes_common_paths() -> None:
    urls = feed_url_candidates("example.com")
    assert "https://example.com/feed" in urls
    assert "https://example.com/rss" in urls
    assert "https://example.com/blog/feed.xml" in urls
    assert "https://example.com/news/rss" in urls


def test_feed_url_candidates_includes_common_subdomains() -> None:
    urls = feed_url_candidates("example.com")
    assert "https://blog.example.com/feed" in urls
    assert "https://news.example.com/feed" in urls
    assert "https://investors.example.com/feed" in urls
    assert "https://newsroom.example.com/feed" in urls


def test_feed_url_candidates_prioritizes_root_paths() -> None:
    """First few candidates should be the conventional /feed shapes."""
    urls = feed_url_candidates("example.com")
    assert urls[0] == "https://example.com/feed"


# ---------- is_feed_response ----------


@pytest.mark.parametrize(
    "ct,body,expected",
    [
        ("application/rss+xml", b"", True),
        ("application/atom+xml; charset=utf-8", b"", True),
        ("application/xml", b"<?xml version='1.0'?><rss>", True),
        ("text/xml", b"<?xml?><feed xmlns='http://www.w3.org/2005/Atom'>", True),
        # Body marker alone is enough.
        ("text/html", b"<?xml version='1.0'?><rss version='2.0'>", True),
        ("text/html", b"<feed xmlns='http://www.w3.org/2005/Atom'>", True),
        ("text/html", b"<rdf:RDF xmlns:rdf='...'>", True),
        # Negatives.
        ("text/html", b"<!DOCTYPE html><html>", False),
        (None, b"", False),
        ("application/json", b'{"hello": "world"}', False),
    ],
)
def test_is_feed_response(ct: str | None, body: bytes, expected: bool) -> None:
    assert is_feed_response(ct, body) is expected


# ---------- probe_feed (mocked transport) ----------


def _mock_transport(handler: Any) -> httpx.MockTransport:
    """Build a httpx transport that returns whatever ``handler`` produces."""

    def fn(request: httpx.Request) -> httpx.Response:
        return handler(request)

    return httpx.MockTransport(fn)


@pytest.mark.asyncio
async def test_probe_feed_returns_response_on_rss_body() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "application/rss+xml"},
            content=b"<?xml version='1.0'?><rss><channel/></rss>",
        )

    async with httpx.AsyncClient(transport=_mock_transport(handler)) as client:
        resp = await probe_feed(client, "https://example.com/feed")
    assert resp is not None
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_probe_feed_returns_none_on_html() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/html"},
            content=b"<!DOCTYPE html><html><body>nope</body></html>",
        )

    async with httpx.AsyncClient(transport=_mock_transport(handler)) as client:
        resp = await probe_feed(client, "https://example.com/feed")
    assert resp is None


@pytest.mark.asyncio
async def test_probe_feed_returns_none_on_404() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(404, content=b"not found")

    async with httpx.AsyncClient(transport=_mock_transport(handler)) as client:
        resp = await probe_feed(client, "https://example.com/feed")
    assert resp is None


@pytest.mark.asyncio
async def test_probe_feed_swallows_network_errors() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("DNS resolution failed")

    async with httpx.AsyncClient(transport=_mock_transport(handler)) as client:
        resp = await probe_feed(client, "https://nonexistent.example.com/feed")
    assert resp is None


# ---------- discover_for_entity (mocked transport) ----------


def _entity(name: str) -> Entity:
    return Entity(
        id=uuid.uuid4(),
        canonical_name=name,
        entity_type=EntityType.PUBLIC_COMPANY,
    )


@pytest.mark.asyncio
async def test_discover_returns_first_hit_when_max_results_is_one() -> None:
    """First domain that yields a hit wins; later domains aren't tried.

    Note: candidate URLs *within* a domain are probed concurrently for
    speed, so a single domain produces ~36 in-flight requests before
    one of them returns a hit. The contract this test pins is *no
    requests to the second domain* once the first domain hits.
    """
    calls: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(str(req.url))
        if str(req.url) == "https://vertiv.com/feed":
            return httpx.Response(
                200,
                headers={"content-type": "application/rss+xml"},
                content=b"<?xml?><rss/>",
            )
        return httpx.Response(404, content=b"")

    async with httpx.AsyncClient(transport=_mock_transport(handler)) as client:
        found = await discover_for_entity(
            client,
            _entity("Vertiv Holdings Co"),
            max_results=1,
        )

    assert len(found) == 1
    assert isinstance(found[0], SourceCandidate)
    assert found[0].feed_url == "https://vertiv.com/feed"
    # All probes confined to the first parent domain (vertiv.com + its
    # blog./news./etc subdomains) — no spillover to fallback parent domains.
    domains_called = {httpx.URL(u).host for u in calls}
    assert all(h == "vertiv.com" or h.endswith(".vertiv.com") for h in domains_called)


@pytest.mark.asyncio
async def test_discover_walks_to_subdomain_when_paths_miss() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        if str(req.url) == "https://blog.acme.com/feed":
            return httpx.Response(
                200,
                headers={"content-type": "application/atom+xml"},
                content=b"<?xml?><feed/>",
            )
        return httpx.Response(404, content=b"")

    async with httpx.AsyncClient(transport=_mock_transport(handler)) as client:
        found = await discover_for_entity(client, _entity("Acme"))

    assert len(found) == 1
    assert found[0].feed_url == "https://blog.acme.com/feed"


@pytest.mark.asyncio
async def test_discover_returns_empty_when_nothing_resolves() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(404, content=b"")

    async with httpx.AsyncClient(transport=_mock_transport(handler)) as client:
        found = await discover_for_entity(client, _entity("Phantom Co"))

    assert found == []


@pytest.mark.asyncio
async def test_discover_max_results_truncates() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        # Every probe is a hit — exercise the truncation guard.
        return httpx.Response(
            200,
            headers={"content-type": "application/rss+xml"},
            content=b"<?xml?><rss/>",
        )

    async with httpx.AsyncClient(transport=_mock_transport(handler)) as client:
        # "Acme Beta" yields 3 distinct domains (acmebeta.com, acme-beta.com,
        # acme.com) — none of "beta" / "acme" are in the strip list.
        found = await discover_for_entity(
            client,
            _entity("Acme Beta"),
            max_results=2,
        )

    # Stops at max_results even if many domains/patterns would all hit.
    assert len(found) == 2
    # Two distinct domains since we break out after the first per-domain hit.
    assert found[0].feed_url != found[1].feed_url


# ---------- is_discoverable_entity_name ----------


@pytest.mark.parametrize(
    "name",
    [
        "Vertiv Holdings Co",
        "Constellation Energy",
        "ASML",
        "Hewlett Packard Enterprise Company",
        "Crusoe Energy Systems",
    ],
)
def test_discoverable_passes_real_companies(name: str) -> None:
    assert is_discoverable_entity_name(name)


@pytest.mark.parametrize(
    "name",
    [
        "Prof. Iryna Gurevych (UKP Lab, TU Darmstadt)",
        "Dr. John Smith",
        "HELMET (Princeton Language and Intelligence)",
        "EduRénov program (public school renovation projects)",
        "ChatGPT Futures Class of 2026 honorees",
        "FilBench authors",
        "Survey respondents",
        "AIMO Prize",
        # Pathologically long (>60).
        "X" * 80,
        "",  # empty
    ],
)
def test_discoverable_rejects_unfriendly_names(name: str) -> None:
    assert not is_discoverable_entity_name(name)


@pytest.mark.asyncio
async def test_discover_skips_unfriendly_names_without_probing() -> None:
    """If the name fails the pre-filter, no HTTP at all."""
    calls: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(str(req.url))
        return httpx.Response(200, content=b"<?xml?><rss/>")

    async with httpx.AsyncClient(transport=_mock_transport(handler)) as client:
        found = await discover_for_entity(
            client,
            _entity("Prof. Some Researcher (Some Lab, Some University)"),
        )

    assert found == []
    assert calls == []


@pytest.mark.asyncio
async def test_discover_returns_empty_when_no_domain_candidates() -> None:
    """All-tokens-stripped entity produces no domains → empty without probing."""
    calls: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(str(req.url))
        return httpx.Response(200, content=b"<?xml?><rss/>")

    async with httpx.AsyncClient(transport=_mock_transport(handler)) as client:
        # "Inc" / "Corp" / "Ltd" alone — every token is a strip token.
        found = await discover_for_entity(client, _entity("Inc Corp Ltd"))

    assert found == []
    assert calls == []  # never tried a probe
