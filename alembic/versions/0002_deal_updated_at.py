"""Add Deal.updated_at for V1.6 dedup tracking.

Existing rows are backfilled with their ``created_at`` value so the
post-migration invariant holds: every row has a non-null
``updated_at >= created_at``.

Revision ID: 0002_deal_updated_at
Revises: 0001_initial_schema
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0002_deal_updated_at"
down_revision: str | Sequence[str] | None = "0001_initial_schema"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Two-step dance to stay portable across Postgres + SQLite:
    # add nullable, backfill, then enforce NOT NULL.
    op.add_column(
        "deal",
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.execute("UPDATE deal SET updated_at = created_at")
    op.alter_column("deal", "updated_at", nullable=False)


def downgrade() -> None:
    op.drop_column("deal", "updated_at")
