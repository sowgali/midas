"""Pydantic V2 response models for the read-only HTTP API.

These DTOs are deliberately separate from the SQLModel rows in
:mod:`midas.models`: the storage layer holds ``Decimal`` and ``UUID``
types so server-side math stays precise, while the wire format uses
``str`` ids and ``float`` amounts so the JSON is directly consumable by
the browser graph viz.

Rows that hydrate from SQLModel instances set
``model_config = ConfigDict(from_attributes=True)`` so the route
handlers can do ``EntityDto.model_validate(entity_row)`` without
manual field-by-field copying. UUIDs are coerced to ``str`` and
``Decimal`` to ``float`` via a ``mode="before"`` model validator on
each affected DTO.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, ConfigDict, model_validator


def _coerce(value: Any) -> Any:
    """Stringify UUIDs and downcast Decimals; leave anything else untouched."""
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, Decimal):
        return float(value)
    return value


def _coerce_attrs(obj: Any, fields: tuple[str, ...]) -> dict[str, Any] | Any:
    """If ``obj`` looks like a SQLModel row, copy ``fields`` into a coerced dict.

    This runs in pydantic's ``mode="before"`` validator, so when callers
    pass an ORM instance we produce a dict whose UUIDs/Decimals are already
    in wire-friendly form. Mappings and unrelated objects are passed
    through untouched.
    """
    if isinstance(obj, dict):
        return {k: _coerce(v) for k, v in obj.items()}
    # Heuristic: if it has every requested attribute, treat it as an ORM row.
    if all(hasattr(obj, f) for f in fields):
        return {f: _coerce(getattr(obj, f)) for f in fields}
    return obj


class EntityDto(BaseModel):
    """A node in the cash-flow graph."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    canonical_name: str
    aliases: list[str]
    ticker: str | None
    cik: str | None
    entity_type: str
    sector_tags: list[str]
    country: str | None

    @model_validator(mode="before")
    @classmethod
    def _from_row(cls, data: Any) -> Any:
        return _coerce_attrs(
            data,
            (
                "id",
                "canonical_name",
                "aliases",
                "ticker",
                "cik",
                "entity_type",
                "sector_tags",
                "country",
            ),
        )


class SourceDto(BaseModel):
    """The provenance pointer that backs an evidence span."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    url: str
    source_type: str
    publisher: str
    title: str | None
    published_at: datetime | None

    @model_validator(mode="before")
    @classmethod
    def _from_row(cls, data: Any) -> Any:
        return _coerce_attrs(
            data,
            ("id", "url", "source_type", "publisher", "title", "published_at"),
        )


class EvidenceDto(BaseModel):
    """A single quoted span supporting a deal, with its source attached."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    text_snippet: str
    char_start: int
    char_end: int
    extractor: str
    source: SourceDto


class DealDto(BaseModel):
    """A directed money-flow claim — id-only references on the list response."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    from_entity_id: str
    to_entity_id: str
    deal_type: str
    status: str
    amount_usd: float | None
    amount_native: float | None
    currency: str | None
    announced_at: date | None
    closes_at: date | None
    confidence: float
    description: str

    @model_validator(mode="before")
    @classmethod
    def _from_row(cls, data: Any) -> Any:
        return _coerce_attrs(
            data,
            (
                "id",
                "from_entity_id",
                "to_entity_id",
                "deal_type",
                "status",
                "amount_usd",
                "amount_native",
                "currency",
                "announced_at",
                "closes_at",
                "confidence",
                "description",
            ),
        )


class DealDetailDto(DealDto):
    """A deal with its endpoints and supporting evidence eagerly inlined."""

    from_entity: EntityDto
    to_entity: EntityDto
    evidence: list[EvidenceDto]


class GraphEdgeDto(BaseModel):
    """An aggregated edge collapsing all parallel deals between a pair."""

    from_id: str
    to_id: str
    total_amount_usd: float | None
    deal_count: int
    deal_types: list[str]


class GraphResponse(BaseModel):
    """Top-level payload for ``GET /api/graph``."""

    nodes: list[EntityDto]
    edges: list[GraphEdgeDto]
    as_of: date | None
    sector: str | None
