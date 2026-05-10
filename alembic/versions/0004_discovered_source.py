"""Add discovered_source table for V1.9.2 BFS source-discovery.

Persists feed URLs that ``midas discover sources`` validates so
subsequent ingest passes pick them up alongside the YAML bootstrap.

Revision ID: 0004_discovered_source
Revises: 0003_entity_discovered
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0004_discovered_source"
down_revision: str | Sequence[str] | None = "0003_entity_discovered"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "discovered_source",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "entity_id",
            sa.Uuid(),
            sa.ForeignKey("entity.id"),
            nullable=False,
            index=True,
        ),
        sa.Column("feed_url", sa.String(length=2048), nullable=False),
        sa.Column("publisher", sa.String(length=255), nullable=False),
        sa.Column("source_type", sa.String(length=32), nullable=False),
        sa.Column(
            "status",
            sa.String(length=32),
            nullable=False,
            server_default="valid",
            index=True,
        ),
        sa.Column(
            "discovered_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "last_probed_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.UniqueConstraint("entity_id", "feed_url", name="uq_discovered_source"),
    )


def downgrade() -> None:
    op.drop_table("discovered_source")
