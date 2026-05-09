"""Deal — a single directed money-flow claim between two entities.

A ``Deal`` is the edge type in the graph. It's directional (``from`` →
``to``); for partnerships without a clear payer we model them as
``deal_type=PARTNERSHIP`` with ``amount_usd=None``. Every ``Deal`` should
have at least one :class:`EvidenceSpan` pointing back to the source text
that supports it — that link is created by the extractor pipeline, not
enforced at the column level (a freshly-extracted ``Deal`` may exist in
memory before its evidence is persisted).
"""

from __future__ import annotations

import uuid
from datetime import UTC, date, datetime
from decimal import Decimal

from sqlalchemy import Column, Date, DateTime, Index, Numeric, String
from sqlmodel import Field, SQLModel

from .types import DealStatus, DealType


class Deal(SQLModel, table=True):
    __tablename__ = "deal"

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)

    from_entity_id: uuid.UUID = Field(foreign_key="entity.id", index=True)
    to_entity_id: uuid.UUID = Field(foreign_key="entity.id", index=True)

    deal_type: DealType = Field(sa_column=Column(String(32), nullable=False, index=True))

    # Decimal — never float — for money. Numeric(20, 2) handles up to 999,999,999,999,999,999.99.
    amount_usd: Decimal | None = Field(
        default=None,
        sa_column=Column(Numeric(20, 2), nullable=True),
    )
    amount_native: Decimal | None = Field(
        default=None,
        sa_column=Column(Numeric(20, 2), nullable=True),
    )
    currency: str | None = Field(default=None, max_length=3, description="ISO 4217.")

    announced_at: date | None = Field(
        default=None,
        sa_column=Column(Date(), nullable=True, index=True),
    )
    closes_at: date | None = Field(
        default=None,
        sa_column=Column(Date(), nullable=True),
    )

    status: DealStatus = Field(sa_column=Column(String(32), nullable=False))

    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="0..1; reflects extractor certainty and source tier.",
    )

    description: str = Field(max_length=2048)

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )
    # Bumped whenever reconciliation merges new evidence into this row.
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )

    __table_args__ = (
        Index("ix_deal_from_announced", "from_entity_id", "announced_at"),
        Index("ix_deal_to_announced", "to_entity_id", "announced_at"),
    )
