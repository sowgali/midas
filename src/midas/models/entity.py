"""Entity — a company or other money-handling actor in the graph.

An ``Entity`` is the canonical record we resolve mentions ("Google",
"GOOGL", "Alphabet Inc.") against. ``aliases`` is intentionally a JSON
list rather than a separate table — V1 uses a hand-curated registry that
is small enough that relational expansion would be overkill.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import Boolean, Column, DateTime, Index, String, text
from sqlmodel import Field, SQLModel

from ._columns import JSON_VARIANT
from .types import EntityType


class Entity(SQLModel, table=True):
    __tablename__ = "entity"

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)

    canonical_name: str = Field(max_length=255, index=True, unique=True)

    aliases: list[str] = Field(
        default_factory=list,
        sa_column=Column(JSON_VARIANT, nullable=False),
    )

    # Ticker and CIK are unique *if present* — partial unique indexes below.
    ticker: str | None = Field(default=None, max_length=16)
    cik: str | None = Field(default=None, max_length=16)

    entity_type: EntityType = Field(sa_column=Column(String(32), nullable=False))

    sector_tags: list[str] = Field(
        default_factory=list,
        sa_column=Column(JSON_VARIANT, nullable=False),
    )

    country: str | None = Field(default=None, max_length=2, description="ISO 3166-1 alpha-2.")

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )

    # V1.8 open-world entity resolution: True when this row was
    # auto-created from an extraction whose party name didn't match any
    # existing entity. Curated registry rows (seed.yaml, human-promoted
    # via `midas review promote`) have ``discovered=False``. Frontends
    # may render discovered entities differently to flag them as
    # "needs human review".
    discovered: bool = Field(
        default=False,
        sa_column=Column(Boolean(), nullable=False),
    )

    __table_args__ = (
        # Partial unique on ticker / cik (PG only; degrades to plain index elsewhere).
        Index(
            "ix_entity_ticker_unique",
            "ticker",
            unique=True,
            postgresql_where=text("ticker IS NOT NULL"),
        ),
        Index(
            "ix_entity_cik_unique",
            "cik",
            unique=True,
            postgresql_where=text("cik IS NOT NULL"),
        ),
    )
