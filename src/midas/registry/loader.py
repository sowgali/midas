"""Load the YAML entity registry into the database.

The registry is the hand-curated list of companies the pipeline knows
about. Loader is **idempotent**: re-running it inserts only entities not
already present (matched by ``canonical_name``). It does NOT update
existing rows — manual edits in the DB win, so a human can correct an
entity (fix a CIK, add an alias) without the seed file overriding it.

If you actually want to overwrite existing rows, add a future
``--force`` mode. V1 keeps the rule simple.
"""

from __future__ import annotations

from importlib.resources import as_file, files
from pathlib import Path
from typing import Any

import structlog
import yaml
from sqlalchemy.ext.asyncio import AsyncSession

from midas.models import Entity
from midas.storage.repository import EntityRepository

log = structlog.get_logger(__name__)

DEFAULT_SEED_FILENAME = "seed.yaml"


def default_seed_path() -> Path:
    """Filesystem path to the bundled seed.yaml.

    Resolves through ``importlib.resources`` so it works whether the
    package is run from source or installed as a wheel.
    """
    resource = files("midas.registry") / DEFAULT_SEED_FILENAME
    with as_file(resource) as path:
        return Path(path)


def parse_seed(seed_path: Path | None = None) -> list[Entity]:
    """Read and validate the seed YAML.

    Each row is run through ``Entity.model_validate`` so unknown enum
    values, missing required fields, etc. fail loudly at load time
    rather than at insert time.
    """
    seed_path = seed_path or default_seed_path()
    with seed_path.open("rb") as f:
        data: dict[str, Any] = yaml.safe_load(f)

    rows = data.get("entities", [])
    if not isinstance(rows, list):
        raise ValueError(f"{seed_path}: top-level 'entities' must be a list")

    return [Entity.model_validate(row) for row in rows]


async def load_seed_registry(
    session: AsyncSession,
    *,
    seed_path: Path | None = None,
) -> tuple[int, int]:
    """Insert any seed entities not already present.

    Returns ``(inserted, skipped)``. Commits on success.
    """
    entities = parse_seed(seed_path)
    repo = EntityRepository(session)

    inserted = 0
    skipped = 0
    for entity in entities:
        existing = await repo.get_by_canonical_name(entity.canonical_name)
        if existing is not None:
            skipped += 1
            log.debug("registry.skip.exists", canonical_name=entity.canonical_name)
            continue
        await repo.add(entity)
        inserted += 1

    await session.commit()
    log.info("registry.loaded", inserted=inserted, skipped=skipped)
    return inserted, skipped
