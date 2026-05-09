"""``/api/entities`` — list and fetch :class:`Entity` rows."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException, status

from midas.storage import EntityRepository

from ..deps import Session
from ..schemas import EntityDto

router = APIRouter(prefix="/entities", tags=["entities"])


@router.get("", response_model=list[EntityDto])
async def list_entities(
    session: Session,
    sector: str | None = None,
) -> list[EntityDto]:
    """List entities, optionally filtered to those tagged with ``sector``."""
    repo = EntityRepository(session)
    rows = await repo.list_by_sector(sector) if sector is not None else await repo.list_all()
    return [EntityDto.model_validate(e) for e in rows]


@router.get("/{entity_id}", response_model=EntityDto)
async def get_entity(entity_id: str, session: Session) -> EntityDto:
    """Fetch a single entity by id; 404 if it doesn't exist."""
    try:
        parsed = uuid.UUID(entity_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="entity not found",
        ) from exc
    repo = EntityRepository(session)
    row = await repo.get(parsed)
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="entity not found",
        )
    return EntityDto.model_validate(row)
