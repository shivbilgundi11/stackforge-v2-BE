"""Keep stale Razorpay webhook snapshots from rolling subscription state back.

Revision ID: c8a5f2d7b310
Revises: f3b8d5e21c47
Create Date: 2026-08-18 09:00:00.000000+00:00
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "c8a5f2d7b310"
down_revision: str | None = "f3b8d5e21c47"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("subscriptions", sa.Column("provider_event_at", sa.DateTime(timezone=True)))


def downgrade() -> None:
    op.drop_column("subscriptions", "provider_event_at")
