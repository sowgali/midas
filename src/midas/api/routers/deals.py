"""``/api/deals`` — list deals and fetch a deal with its evidence chain."""

from __future__ import annotations

import uuid
from datetime import date
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, status
from sqlmodel import col, select

from midas.models import Deal, Entity, EvidenceSpan, Source

from ..deps import Session
from ..schemas import DealDetailDto, DealDto, EntityDto, EvidenceDto, SourceDto

router = APIRouter(prefix="/deals", tags=["deals"])


@router.get("", response_model=list[DealDto])
async def list_deals(
    session: Session,
    from_id: Annotated[
        str | None,
        Query(description="Restrict to deals where this entity is the payer."),
    ] = None,
    to_id: Annotated[
        str | None,
        Query(description="Restrict to deals where this entity is the recipient."),
    ] = None,
    as_of: Annotated[
        date | None,
        Query(
            description=(
                "ISO date — exclude deals announced after this date "
                "(and deals with no announcement date)."
            ),
        ),
    ] = None,
) -> list[DealDto]:
    """List deals, with optional payer / payee / temporal filters.

    The query is built directly here (rather than calling
    :class:`DealRepository`) because ``DealRepository`` doesn't expose a
    combined-filter list method — the existing repo API is shaped for the
    extraction pipeline, not the read-side. We don't add new SQL inside
    the repo for what is purely a read concern.
    """
    stmt = select(Deal)
    if from_id is not None:
        try:
            stmt = stmt.where(col(Deal.from_entity_id) == uuid.UUID(from_id))
        except ValueError:
            return []
    if to_id is not None:
        try:
            stmt = stmt.where(col(Deal.to_entity_id) == uuid.UUID(to_id))
        except ValueError:
            return []
    if as_of is not None:
        stmt = stmt.where(col(Deal.announced_at).is_not(None)).where(
            col(Deal.announced_at) <= as_of,
        )
    result = await session.execute(stmt)
    rows = list(result.scalars().all())
    return [DealDto.model_validate(d) for d in rows]


@router.get("/{deal_id}", response_model=DealDetailDto)
async def get_deal(deal_id: str, session: Session) -> DealDetailDto:
    """Fetch a deal with both endpoints and every supporting evidence span."""
    try:
        parsed = uuid.UUID(deal_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="deal not found",
        ) from exc

    deal = await session.get(Deal, parsed)
    if deal is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="deal not found",
        )

    from_entity = await session.get(Entity, deal.from_entity_id)
    to_entity = await session.get(Entity, deal.to_entity_id)
    if from_entity is None or to_entity is None:
        # Shouldn't happen — FK guarantees endpoints exist — but be explicit.
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="deal endpoints missing",
        )

    span_stmt = select(EvidenceSpan).where(col(EvidenceSpan.deal_id) == deal.id)
    spans = list((await session.execute(span_stmt)).scalars().all())

    # Bulk-load the underlying sources in one round trip rather than N+1.
    source_ids = {s.source_id for s in spans}
    sources_by_id: dict[uuid.UUID, Source] = {}
    if source_ids:
        src_stmt = select(Source).where(col(Source.id).in_(source_ids))
        for source_row in (await session.execute(src_stmt)).scalars().all():
            sources_by_id[source_row.id] = source_row

    evidence_dtos: list[EvidenceDto] = []
    for span in spans:
        source = sources_by_id.get(span.source_id)
        if source is None:
            continue  # orphan span — shouldn't happen with FK enabled.
        evidence_dtos.append(
            EvidenceDto(
                id=str(span.id),
                text_snippet=span.text_snippet,
                char_start=span.char_start,
                char_end=span.char_end,
                extractor=span.extractor,
                source=SourceDto.model_validate(source),
            ),
        )

    # Build the detail DTO by going through a dict so the parent
    # ``_from_row`` validator handles the deal-level UUID/Decimal coercion.
    return DealDetailDto.model_validate(
        {
            "id": deal.id,
            "from_entity_id": deal.from_entity_id,
            "to_entity_id": deal.to_entity_id,
            "deal_type": deal.deal_type,
            "status": deal.status,
            "amount_usd": deal.amount_usd,
            "amount_native": deal.amount_native,
            "currency": deal.currency,
            "announced_at": deal.announced_at,
            "closes_at": deal.closes_at,
            "confidence": deal.confidence,
            "description": deal.description,
            "from_entity": EntityDto.model_validate(from_entity),
            "to_entity": EntityDto.model_validate(to_entity),
            "evidence": evidence_dtos,
        },
    )
