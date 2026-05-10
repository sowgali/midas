"""Heuristic RSS-feed discovery for entities lacking a curated source.

Three pure-ish primitives compose into :func:`discover_for_entity`:

* :func:`derive_domain_candidates` — entity canonical_name → plausible
  domains. We're permissive: strip corporate suffixes, lowercase, try
  ``.com`` then a small set of fallbacks.
* :func:`feed_url_candidates` — domain → ordered list of likely feed
  URLs (``/feed``, ``/news/rss``, ``investors.{d}/feed``, etc.).
* :func:`probe_feed` — async GET; returns the response if the body
  sniffs as RSS / Atom XML. Network errors and non-XML responses
  return ``None``.

:func:`discover_for_entity` chains them, returning at most ``max_results``
:class:`SourceCandidate` rows ready for DB persistence.

Designed for testability: every external call is async + mockable, and
the URL-generation steps are pure functions so they can be tested
without any HTTP at all.

Per-entity probes run concurrently within a domain (asyncio.gather),
so a domain that hosts a feed at ``/feed`` and 35 dead URLs takes ~one
HTTP timeout instead of ~36. The first hit on a domain wins; later
domains aren't tried unless the first comes up empty.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Iterable
from dataclasses import dataclass

import httpx
import structlog

from midas.models import Entity
from midas.models.types import SourceType

log = structlog.get_logger(__name__)

# Per-probe HTTP timeout. 2.5s is short enough that a dead domain costs
# under 3s total (with concurrent probing) and long enough that a real
# slow CDN still answers.
_PROBE_TIMEOUT_S = 2.5

# Per-entity total wall-clock cap. A degenerate name that generates many
# domains shouldn't be allowed to monopolise the discovery phase.
_PER_ENTITY_TIMEOUT_S = 12.0


# Names that won't yield a useful corporate domain. We skip discovery
# for these instead of burning probe time on garbage. Most common
# pattern: extracted-prose names that are people, programs, or
# parenthetical phrases — nothing the open-world resolver can validate
# until human review reclassifies them.
_DOMAIN_BLOCKLIST_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\s*(prof|dr|mr|mrs|ms|sir)\.?\s", re.IGNORECASE),  # personal titles
    re.compile(r"\(.*?\)"),  # parentheticals — long taxonomic names
    re.compile(r"\b(authors|respondents|attendees|sponsors|recipients)\s*$", re.IGNORECASE),
    re.compile(r"\b(program|programme|initiative|fellowship|prize|class|cohort)\b", re.IGNORECASE),
    re.compile(r"\b(gold medal|first place|second place|honoree|honoree)\b", re.IGNORECASE),
)


def is_discoverable_entity_name(canonical_name: str) -> bool:
    """Cheap pre-filter: does this name look like it could resolve to a domain?

    Conservative — we'd rather skip a real but unusual name than burn
    seconds on a junk one. The :func:`is_extractable_entity_name`
    filter is the *first* line of defense (we already pruned the
    obvious garbage); this is a *second* line that catches
    domain-unfriendly shapes specifically.

    Reject criteria:

    - Personal titles ("Prof.", "Dr.")
    - Parentheticals (taxonomic descriptors, lab affiliations)
    - Tail words signalling an aggregate / programme rather than a co
    - Pathologically long (>60 chars; real corporate names rarely are)
    """
    if not canonical_name or len(canonical_name) > 60:
        return False
    return all(not pat.search(canonical_name) for pat in _DOMAIN_BLOCKLIST_PATTERNS)


# ---------- pure URL generation ----------

# Tokens we strip from canonical_name before deriving a domain. Same
# list as ``entity_resolution._CORP_SUFFIXES`` but specialised — we
# also strip "the" / "platforms" / "technologies" since those are
# decorative.
_DOMAIN_STRIP_TOKENS: frozenset[str] = frozenset(
    {
        "the",
        "company",
        "corporation",
        "corp",
        "incorporated",
        "inc",
        "ltd",
        "limited",
        "plc",
        "llc",
        "lp",
        "co",
        "holdings",
        "holding",
        "group",
        "platforms",
        "technologies",
        "tech",
        "labs",
        "ai",
        "sa",
        "ag",
        "nv",
        "gmbh",
        "pbc",
    },
)

_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def _slug(text: str) -> str:
    """Lowercase, replace non-alnum runs with a single dash, strip edges."""
    return _NON_ALNUM_RE.sub("-", text.lower()).strip("-")


def derive_domain_candidates(canonical_name: str) -> list[str]:
    """Plausible domains for ``canonical_name``, in priority order.

    Strategy: drop punctuation + decorative tokens, then build a few
    common shapes (compacted, hyphenated, single-token). Always returns
    a list with no duplicates and at least one element when the input
    is non-empty.

    >>> derive_domain_candidates("Vertiv Holdings Co")
    ['vertiv.com', 'vertivholdings.com', 'vertiv-holdings.com']
    >>> derive_domain_candidates("Constellation Energy")
    ['constellationenergy.com', 'constellation-energy.com', 'constellation.com']
    >>> derive_domain_candidates("Hewlett Packard Enterprise Company")
    ['hewlettpackardenterprise.com', 'hewlett-packard-enterprise.com', 'hpe.com', 'hewlettpackard.com']
    """
    cleaned = re.sub(r"[(),.&/\\]+", " ", canonical_name)
    raw_tokens = cleaned.lower().split()
    tokens = [t for t in raw_tokens if t and t not in _DOMAIN_STRIP_TOKENS]
    if not tokens:
        return []

    candidates: list[str] = []

    def _push(domain: str) -> None:
        if domain and domain not in candidates:
            candidates.append(domain)

    # Compact: vertivholdings.com, constellationenergy.com
    if len(tokens) > 1:
        _push(_slug("".join(tokens)) + ".com")
        _push(_slug("-".join(tokens)) + ".com")
        # Single-token fallback — first token is usually the brand.
        _push(_slug(tokens[0]) + ".com")
        # Acronym from initials (HPE, AMD, etc.) — only when ≥3 chars so we
        # don't generate spammy 2-letter domains like "ce.com" for ordinary
        # two-word names.
        acronym = "".join(t[0] for t in tokens if t)
        if len(acronym) >= 3:
            _push(acronym + ".com")
        # First-two-tokens compact, in case a meaningful suffix was kept.
        _push(_slug("".join(tokens[:2])) + ".com")
    else:
        _push(_slug(tokens[0]) + ".com")

    return candidates


# Ordered: stop at the first hit per domain. Listed by approximate
# real-world prevalence (top is most common feed convention).
_FEED_PATH_TEMPLATES: tuple[str, ...] = (
    "/feed",
    "/feed/",
    "/feed.xml",
    "/rss",
    "/rss.xml",
    "/index.xml",
    "/atom.xml",
    "/blog/feed",
    "/blog/feed/",
    "/blog/feed.xml",
    "/blog/rss",
    "/news/feed",
    "/news/rss",
    "/news/rss.xml",
    "/newsroom/feed",
    "/newsroom/rss",
    "/press/feed",
    "/press/rss",
)

# Common feed-hosting subdomains: these get the same suffix list, but
# prefixed with the subdomain in front of the bare domain.
_FEED_SUBDOMAINS: tuple[str, ...] = (
    "blog",
    "blogs",
    "news",
    "newsroom",
    "investors",
    "investor",
    "press",
)


def feed_url_candidates(domain: str) -> list[str]:
    """Ordered list of probable feed URLs to try for ``domain``.

    Combines path-style (``https://{domain}/feed``) and subdomain-style
    (``https://blog.{domain}/feed``) candidates. The first hit wins,
    so order matters: cheap / common patterns first.
    """
    urls: list[str] = []
    for path in _FEED_PATH_TEMPLATES:
        urls.append(f"https://{domain}{path}")
    for sub in _FEED_SUBDOMAINS:
        for path in _FEED_PATH_TEMPLATES[:5]:  # subdomain only with the top-5 paths.
            urls.append(f"https://{sub}.{domain}{path}")
    return urls


# ---------- probe + sniff ----------

# Sniff window — RSS/Atom feeds are XML; the document tag is usually in
# the first few KB. We avoid downloading the whole body for big feeds.
_SNIFF_BYTES = 4096

_FEED_MARKERS: tuple[bytes, ...] = (
    b"<rss",
    b"<feed",
    b"<channel",
    b"<rdf:rdf",  # lowercase: we lowercase the body before matching
    b"application/rss",
)

_FEED_CONTENT_TYPES: tuple[str, ...] = (
    "application/rss+xml",
    "application/atom+xml",
    "application/xml",
    "text/xml",
)


def is_feed_response(content_type: str | None, body_head: bytes) -> bool:
    """Return ``True`` if a response looks like an RSS / Atom feed.

    Either ``content_type`` matches a known feed MIME, or the body
    head contains an RSS/Atom marker tag. Pure function so it's
    trivially testable without a network.
    """
    if content_type:
        ct = content_type.split(";", 1)[0].strip().lower()
        if ct in _FEED_CONTENT_TYPES:
            return True
    lowered = body_head.lower()
    return any(marker in lowered for marker in _FEED_MARKERS)


async def probe_feed(client: httpx.AsyncClient, url: str) -> httpx.Response | None:
    """Fetch ``url`` and return the response iff it parses as a feed.

    On HTTP error, non-feed body, or network failure: returns ``None``
    (caller treats as "not a feed here, try next").
    """
    try:
        resp = await client.get(url, timeout=httpx.Timeout(_PROBE_TIMEOUT_S))
    except (httpx.HTTPError, httpx.InvalidURL) as exc:
        log.debug("discover.probe.network_error", url=url, err=str(exc))
        return None
    if resp.status_code != 200:
        log.debug("discover.probe.bad_status", url=url, status=resp.status_code)
        return None
    body_head = resp.content[:_SNIFF_BYTES]
    if is_feed_response(resp.headers.get("content-type"), body_head):
        return resp
    log.debug("discover.probe.not_a_feed", url=url, ct=resp.headers.get("content-type"))
    return None


# ---------- composed: per-entity discovery ----------


@dataclass(frozen=True, slots=True)
class SourceCandidate:
    """A validated feed URL discovered for an entity.

    ``publisher`` is auto-derived from canonical_name; the caller is
    free to override before persisting. ``source_type`` defaults to
    :class:`SourceType.BLOG` — corporate news feeds and IR-press feeds
    both map there for the purposes of the extractor.
    """

    entity_id: object  # uuid.UUID, but kept loose to avoid import cycle in tests
    canonical_name: str
    feed_url: str
    publisher: str
    source_type: SourceType


async def _probe_domain_concurrently(
    client: httpx.AsyncClient,
    urls: list[str],
) -> str | None:
    """Probe every URL for one domain concurrently; return the first hit.

    Concurrency makes a *huge* difference on dead domains: instead of
    paying timeout-seconds times 36 candidates serially, one timeout
    window covers all probes in parallel. Real domains still return
    their first hit fast (~100-300 ms).
    """
    import contextlib

    tasks = [asyncio.create_task(probe_feed(client, url)) for url in urls]
    try:
        for finished in asyncio.as_completed(tasks):
            resp = await finished
            if resp is None:
                continue
            # We have a hit — cancel pending probes and return its URL.
            for t in tasks:
                if not t.done():
                    t.cancel()
            return str(resp.request.url)
        return None
    finally:
        # Belt-and-suspenders: ensure no orphan tasks survive the function.
        for t in tasks:
            if not t.done():
                t.cancel()
        # Drain so cancellations propagate cleanly.
        for t in tasks:
            with contextlib.suppress(asyncio.CancelledError, httpx.HTTPError, httpx.InvalidURL):
                await t


async def discover_for_entity(
    client: httpx.AsyncClient,
    entity: Entity,
    *,
    max_results: int = 1,
    domain_overrides: Iterable[str] | None = None,
) -> list[SourceCandidate]:
    """Probe candidate feed URLs for ``entity``; return validated ones.

    Per-domain probes run concurrently (one HTTP timeout window covers
    all 36 candidate URLs). Domains are tried in order; the first
    domain that yields a hit is enough for ``max_results=1``. The whole
    call is wrapped in a wall-clock timeout so a degenerate name can't
    monopolise the discovery phase.
    """
    found: list[SourceCandidate] = []
    if domain_overrides is not None:
        domains = list(domain_overrides)
    elif not is_discoverable_entity_name(entity.canonical_name):
        log.debug("discover.entity.skipped_unfriendly_name", entity=entity.canonical_name)
        return found
    else:
        domains = derive_domain_candidates(entity.canonical_name)

    async def _walk() -> None:
        for domain in domains:
            urls = feed_url_candidates(domain)
            hit = await _probe_domain_concurrently(client, urls)
            if hit is None:
                continue
            found.append(
                SourceCandidate(
                    entity_id=entity.id,
                    canonical_name=entity.canonical_name,
                    feed_url=hit,
                    publisher=f"{entity.canonical_name} (auto-discovered)",
                    source_type=SourceType.BLOG,
                ),
            )
            log.info(
                "discover.feed.hit",
                entity=entity.canonical_name,
                url=hit,
            )
            if len(found) >= max_results:
                return

    try:
        await asyncio.wait_for(_walk(), timeout=_PER_ENTITY_TIMEOUT_S)
    except TimeoutError:
        log.warning(
            "discover.entity.timeout",
            entity=entity.canonical_name,
            partial_hits=len(found),
        )
    return found
