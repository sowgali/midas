"""DiscoveredSource — auto-discovered feed URLs.

Persists what :mod:`midas.discovery.sources` finds so subsequent ingest
passes can pick the URL up alongside the YAML bootstrap set, *and* so
re-running discovery doesn't waste HTTP probes on URLs we already
validated (or rejected).

Schema choice: one row per (entity_id, feed_url) pair. ``status`` lets
us mark a feed dead later without forgetting we tried it. ``publisher``
+ ``source_type`` mirror :class:`Source` so the runtime can construct an
:class:`RssSourceConfig` from a row without further lookups.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import Column, DateTime, String, UniqueConstraint
from sqlmodel import Field, SQLModel

from .types import SourceType


class DiscoveredSource(SQLModel, table=True):
    __tablename__ = "discovered_source"
    __table_args__ = (UniqueConstraint("entity_id", "feed_url", name="uq_discovered_source"),)

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    entity_id: uuid.UUID = Field(foreign_key="entity.id", index=True)

    feed_url: str = Field(max_length=2048)
    publisher: str = Field(max_length=255)
    source_type: SourceType = Field(sa_column=Column(String(32), nullable=False))

    # Lifecycle: valid (ingest-eligible), broken (recent probe failed),
    # superseded (replaced by a hand-curated YAML entry).
    status: str = Field(default="valid", max_length=32, index=True)

    discovered_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )
    last_probed_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )
