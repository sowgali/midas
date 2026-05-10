"""Unit + integration tests for V1.8 open-world entity resolution.

Three layers covered:

1. Pure-function tests for ``normalize_entity_name`` and
   ``is_extractable_entity_name``.
2. In-memory ``EntityResolver`` tests (no DB) for exact / normalized
   match behavior.
3. DB-backed tests for ``resolve_or_create`` — auto-creation,
   filter-rejection, in-run cache update.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncGenerator

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlmodel import SQLModel

from midas.entity_resolution import (
    EntityResolver,
    is_extractable_entity_name,
    normalize_entity_name,
)
from midas.models import Entity, EntityType
from midas.storage.repository import EntityRepository

# ---------- normalize_entity_name ----------


def test_normalize_strips_inc_corporation_llc() -> None:
    assert normalize_entity_name("Microsoft Corporation") == "microsoft"
    assert normalize_entity_name("Microsoft Corp.") == "microsoft"
    assert normalize_entity_name("Microsoft Corp") == "microsoft"
    assert normalize_entity_name("Microsoft Inc.") == "microsoft"
    assert normalize_entity_name("Microsoft, Inc.") == "microsoft"


def test_normalize_strips_pbc_holdings() -> None:
    assert normalize_entity_name("Anthropic, PBC") == "anthropic"
    assert normalize_entity_name("Berkshire Hathaway Holdings") == "berkshire hathaway"


def test_normalize_handles_multiword_names() -> None:
    assert normalize_entity_name("Hugging Face") == "hugging face"
    assert normalize_entity_name("Taiwan Semiconductor Manufacturing Company") == (
        "taiwan semiconductor manufacturing"
    )


def test_normalize_keeps_short_acronyms() -> None:
    assert normalize_entity_name("MSFT") == "msft"
    assert normalize_entity_name("AMD") == "amd"


def test_normalize_collapses_whitespace_and_punctuation() -> None:
    assert normalize_entity_name("  Wiz,  Inc.  ") == "wiz"
    assert normalize_entity_name("Alphabet (Google) Inc.") == "alphabet google"


def test_normalize_empty_input() -> None:
    assert normalize_entity_name("") == ""
    assert normalize_entity_name("   ") == ""


# ---------- is_extractable_entity_name ----------


@pytest.mark.parametrize(
    "name",
    [
        "we",
        "us",
        "our",
        "ourselves",
        "the Company",
        "the Issuer",
        "the Registrant",
        "Public Investors",
        "Bond Investors",
        "bondholders",
        "noteholders",
        "shareholders",
        "third parties",
        "the public",
        "the Group",
        "Privacy litigation plaintiffs",
        "settlement claimants",
        "competitors",
        "regulators",
        "officers",
        "directors",
    ],
)
def test_filter_rejects_generic_referents(name: str) -> None:
    assert not is_extractable_entity_name(name), f"should reject: {name!r}"


@pytest.mark.parametrize(
    "name",
    [
        "Wiz, Inc.",
        "Constellation Energy",
        "CoreWeave",
        "Vertiv Holdings Co.",
        "Schneider Electric SE",
        "Crusoe Energy Systems",
        "Lambda, Inc.",
        "Talen Energy Corporation",
    ],
)
def test_filter_passes_real_company_names(name: str) -> None:
    assert is_extractable_entity_name(name), f"should pass: {name!r}"


def test_filter_rejects_too_short() -> None:
    assert not is_extractable_entity_name("X")
    assert not is_extractable_entity_name("AB")
    assert is_extractable_entity_name("ABC")  # 3 chars OK


def test_filter_rejects_punctuation_only() -> None:
    assert not is_extractable_entity_name("???")
    assert not is_extractable_entity_name("---")


def test_filter_rejects_lowercase_single_word() -> None:
    assert not is_extractable_entity_name("vendors")
    assert not is_extractable_entity_name("creditors")


# V1.9 false-positive cleanup — learned from observed HF blog
# extractions that slipped through V1.8's filter.


@pytest.mark.parametrize(
    "name",
    [
        "Undisclosed investors",
        "Undisclosed Investor",
        "Unspecified IMO gold medal AI winner",
        "Unnamed party",
        "Anonymous donor",
        "Various investors",
    ],
)
def test_filter_rejects_placeholder_prefix(name: str) -> None:
    assert not is_extractable_entity_name(name), f"should reject: {name!r}"


@pytest.mark.parametrize(
    "name",
    [
        "1st Place Winner",
        "2nd Place Winner",
        "3rd Place Winner",
        "Top Student Participants",
        "FilBench authors",
        "Survey respondents",
        "North Africa region",
        "Annual sponsors",
        "Conference attendees",
        # V1.9.1 additions — learned from playwright ingest false-positives
        "ChatGPT Futures Class of 2026 honorees",
        "2025 Anthropic Fellows awardees",
        "Grant recipients",
        "Award nominees",
    ],
)
def test_filter_rejects_aggregate_suffix(name: str) -> None:
    assert not is_extractable_entity_name(name), f"should reject: {name!r}"


def test_filter_still_passes_borderline_real_entities() -> None:
    """The new aggregate-suffix rule must NOT reject real orgs that
    happen to have a similar-shaped tail.
    """
    # "Winners Inc." would canonically be a single-token thing if anyone
    # named a company that — we're filtering on the last *word*.
    assert is_extractable_entity_name("Together AI")
    assert is_extractable_entity_name("DeepSeek")
    assert is_extractable_entity_name("Sequoia Capital")
    assert is_extractable_entity_name("U.S. Government")
    # Government / nonprofit-shaped: keep, human-review can reclassify.
    assert is_extractable_entity_name("European Commission")


# ---------- EntityResolver: in-memory exact + normalized match ----------


def _make_entity(
    canonical: str,
    *,
    aliases: list[str] | None = None,
    entity_type: EntityType = EntityType.PRIVATE_COMPANY,
) -> Entity:
    return Entity(
        canonical_name=canonical,
        aliases=aliases or [],
        entity_type=entity_type,
        sector_tags=["ai"],
    )


def test_resolver_exact_match_canonical() -> None:
    e = _make_entity("Microsoft Corporation", aliases=["MSFT"])
    r = EntityResolver([e])
    assert r.resolve("Microsoft Corporation") == e.id
    assert r.resolve("microsoft corporation") == e.id


def test_resolver_exact_match_alias() -> None:
    e = _make_entity("Microsoft Corporation", aliases=["MSFT", "Microsoft"])
    r = EntityResolver([e])
    assert r.resolve("MSFT") == e.id
    assert r.resolve("Microsoft") == e.id


def test_resolver_normalized_match_strips_corp_suffix() -> None:
    """The V1.8 unlock: 'Microsoft' resolves to 'Microsoft Corporation'
    even when 'Microsoft' isn't an alias.
    """
    e = _make_entity("Microsoft Corporation", aliases=[])
    r = EntityResolver([e])
    assert r.resolve("Microsoft") == e.id
    assert r.resolve("Microsoft Corp") == e.id
    assert r.resolve("Microsoft Inc.") == e.id


def test_resolver_normalized_does_not_collide_across_distinct_orgs() -> None:
    a = _make_entity("Apple Inc.")  # norm: "apple"
    b = _make_entity("Apple Hospitality, Inc.")  # norm: "apple hospitality"
    r = EntityResolver([a, b])
    assert r.resolve("Apple") == a.id
    assert r.resolve("Apple Hospitality") == b.id


def test_resolver_unknown_returns_none() -> None:
    r = EntityResolver([_make_entity("Microsoft Corporation")])
    assert r.resolve("Constellation Energy") is None


# ---------- DB-backed resolve_or_create ----------


@pytest.fixture
async def session() -> AsyncGenerator[AsyncSession, None]:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    async with AsyncSession(engine, expire_on_commit=False) as s:
        yield s
    await engine.dispose()


async def test_resolve_or_create_returns_existing_for_known(session: AsyncSession) -> None:
    msft = _make_entity("Microsoft Corporation", aliases=["MSFT"])
    await EntityRepository(session).add(msft)
    await session.commit()
    resolver = await EntityResolver.from_session(session)

    found = await resolver.resolve_or_create(session, "MSFT")
    assert found == msft.id
    # No new row was created.
    all_entities = await EntityRepository(session).list_all()
    assert len(all_entities) == 1


async def test_resolve_or_create_creates_discovered_for_unknown(
    session: AsyncSession,
) -> None:
    resolver = await EntityResolver.from_session(session)
    new_id = await resolver.resolve_or_create(session, "Constellation Energy")
    assert new_id is not None
    await session.commit()

    # Row exists, flagged discovered.
    fetched = await EntityRepository(session).get(new_id)
    assert fetched is not None
    assert fetched.canonical_name == "Constellation Energy"
    assert fetched.discovered is True
    assert fetched.entity_type == EntityType.PRIVATE_COMPANY


async def test_resolve_or_create_caches_in_run(session: AsyncSession) -> None:
    """Subsequent calls in the same run should hit the in-memory cache,
    not create a duplicate row.
    """
    resolver = await EntityResolver.from_session(session)
    a = await resolver.resolve_or_create(session, "Constellation Energy")
    b = await resolver.resolve_or_create(session, "Constellation Energy")
    c = await resolver.resolve_or_create(session, "constellation energy")  # case-insensitive
    d = await resolver.resolve_or_create(session, "Constellation Energy Corp.")  # normalized
    assert a == b == c == d


async def test_resolve_or_create_returns_none_for_filtered_name(
    session: AsyncSession,
) -> None:
    resolver = await EntityResolver.from_session(session)
    for bad in ("we", "the Company", "Public Investors", "shareholders"):
        result = await resolver.resolve_or_create(session, bad)
        assert result is None, f"should not auto-create for {bad!r}"
    # And no rows landed in the DB.
    rows = await EntityRepository(session).list_all()
    assert rows == []


async def test_resolve_or_create_promotes_to_known_parties(
    session: AsyncSession,
) -> None:
    """A discovered entity should appear in known_parties for downstream extractors."""
    resolver = await EntityResolver.from_session(session)
    assert resolver.known_parties == []
    new_id = await resolver.resolve_or_create(session, "Constellation Energy")
    assert new_id is not None
    names = [p.canonical_name for p in resolver.known_parties]
    assert "Constellation Energy" in names


# ---------- Suppress unused-import warning when running standalone ----------
_ = uuid
