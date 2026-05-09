"""Add Entity.discovered for V1.8 open-world entity resolution.

Existing rows are seeded / human-curated, so they get ``discovered=false``.

Revision ID: 0003_entity_discovered
Revises: 0002_deal_updated_at
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0003_entity_discovered"
down_revision: str | Sequence[str] | None = "0002_deal_updated_at"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Add nullable, backfill, then enforce NOT NULL — same dance as 0002
    # so the migration is portable across Postgres + SQLite.
    op.add_column(
        "entity",
        sa.Column(
            "discovered",
            sa.Boolean(),
            nullable=True,
            server_default=sa.text("false"),
        ),
    )
    op.execute("UPDATE entity SET discovered = false")
    op.alter_column("entity", "discovered", nullable=False)


def downgrade() -> None:
    op.drop_column("entity", "discovered")
