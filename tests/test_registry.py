"""Tests for the entity registry seed + loader."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from pathlib import Path

import pytest
import yaml
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlmodel import SQLModel

from midas.models import Entity, EntityType
from midas.registry import default_seed_path, load_seed_registry, parse_seed
from midas.storage.repository import EntityRepository


@pytest.fixture
async def session() -> AsyncGenerator[AsyncSession, None]:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    async with AsyncSession(engine, expire_on_commit=False) as s:
        yield s
    await engine.dispose()


# ---------- seed.yaml ----------


def test_seed_yaml_parses_into_entities() -> None:
    entities = parse_seed()
    assert len(entities) >= 15  # ~20 companies seeded
    # All rows must validate as Entity (parse_seed does this).
    assert all(isinstance(e, Entity) for e in entities)


def test_seed_yaml_has_pyramid_top() -> None:
    """The seed must include the hyperscalers + a few key labs."""
    names = {e.canonical_name for e in parse_seed()}
    expected = {
        "Microsoft Corporation",
        "Alphabet Inc.",
        "Amazon.com, Inc.",
        "Meta Platforms, Inc.",
        "OpenAI",
        "Anthropic",
        "NVIDIA Corporation",
    }
    assert expected.issubset(names)


def test_seed_yaml_public_companies_have_cik() -> None:
    # CIK is a US-SEC concept; non-US public cos (Schneider, Eaton, ASML, TSMC ADR
    # entry, etc.) may legitimately lack one even when seeded as public_company.
    for e in parse_seed():
        if e.entity_type == EntityType.PUBLIC_COMPANY and e.country == "US":
            assert e.cik is not None, f"US public company {e.canonical_name} missing CIK"
            assert len(e.cik) == 10, f"{e.canonical_name}: CIK must be 10-char zero-padded"
            assert e.cik.isdigit(), f"{e.canonical_name}: CIK must be all digits"


def test_seed_yaml_no_duplicate_canonical_names() -> None:
    names = [e.canonical_name for e in parse_seed()]
    assert len(names) == len(set(names))


def test_seed_yaml_aliases_dont_collide_across_entities() -> None:
    """No alias should resolve to two different entities (case-insensitive)."""
    seen: dict[str, str] = {}
    for e in parse_seed():
        for key in [e.canonical_name, *e.aliases]:
            norm = key.lower()
            if norm in seen and seen[norm] != e.canonical_name:
                pytest.fail(
                    f"alias collision: {key!r} maps to {seen[norm]!r} and {e.canonical_name!r}"
                )
            seen[norm] = e.canonical_name


def test_default_seed_path_exists() -> None:
    path = default_seed_path()
    assert path.exists()
    assert path.suffix == ".yaml"


# ---------- loader ----------


async def test_load_seed_registry_inserts_all(session: AsyncSession) -> None:
    inserted, skipped = await load_seed_registry(session)
    assert inserted == len(parse_seed())
    assert skipped == 0


async def test_load_seed_registry_is_idempotent(session: AsyncSession) -> None:
    first_inserted, _ = await load_seed_registry(session)
    second_inserted, second_skipped = await load_seed_registry(session)
    assert first_inserted > 0
    assert second_inserted == 0
    assert second_skipped == first_inserted


async def test_load_seed_does_not_overwrite_existing(session: AsyncSession, tmp_path: Path) -> None:
    """Manual DB edits win over the seed file (no clobber)."""
    # Pre-insert a row with edited aliases.
    repo = EntityRepository(session)
    edited = Entity(
        canonical_name="OpenAI",
        aliases=["custom-alias"],
        entity_type=EntityType.PRIVATE_COMPANY,
        sector_tags=["ai"],
    )
    await repo.add(edited)
    await session.commit()

    await load_seed_registry(session)

    found = await repo.get_by_canonical_name("OpenAI")
    assert found is not None
    assert found.aliases == ["custom-alias"]


async def test_load_custom_seed_path(session: AsyncSession, tmp_path: Path) -> None:
    custom = tmp_path / "tiny.yaml"
    custom.write_text(
        yaml.safe_dump(
            {
                "entities": [
                    {
                        "canonical_name": "Acme Co.",
                        "aliases": ["Acme"],
                        "entity_type": "private_company",
                        "sector_tags": ["test"],
                    },
                ]
            }
        ),
    )

    inserted, _ = await load_seed_registry(session, seed_path=custom)
    assert inserted == 1

    repo = EntityRepository(session)
    acme = await repo.get_by_canonical_name("Acme Co.")
    assert acme is not None
    assert acme.aliases == ["Acme"]


def test_parse_seed_rejects_bad_top_level(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("entities: not-a-list\n")
    with pytest.raises(ValueError, match="must be a list"):
        parse_seed(bad)
