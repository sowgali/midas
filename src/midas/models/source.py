"""Source — a single fetched document we extract claims from.

A ``Source`` is identified primarily by ``content_sha256`` (the digest of
the raw bytes we fetched). That makes deduplication trivial: the same
press release published at two URLs collapses into one row, and any
later re-fetch is a no-op. ``url`` and ``fetched_at`` are kept for
diagnostics.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import Column, DateTime, Index, String
from sqlmodel import Field, SQLModel

from .types import SourceType


class Source(SQLModel, table=True):
    __tablename__ = "source"

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)

    url: str = Field(max_length=2048, index=True)

    source_type: SourceType = Field(sa_column=Column(String(32), nullable=False, index=True))

    publisher: str = Field(max_length=255, description="SEC, company name, news outlet, etc.")

    title: str | None = Field(default=None, max_length=1024)

    published_at: datetime | None = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), nullable=True),
    )

    fetched_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )

    content_sha256: str = Field(
        max_length=64,
        unique=True,
        index=True,
        description="Hex SHA-256 of the raw fetched bytes; primary dedup key.",
    )

    __table_args__ = (Index("ix_source_type_published_at", "source_type", "published_at"),)
