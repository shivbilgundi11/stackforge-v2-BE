"""razorpay: the customer id is learned, not created

`provider_customer_id` becomes nullable. A subscription created against a
customer we made has no hosted authorization page — Razorpay serves an error on
its own `short_url` — so the customer is created on that page instead, and its
id arrives with the first webhook. Between starting a checkout and authorizing
the mandate there is no customer to record, and the NOT NULL made that state
unrepresentable.

Existing rows keep their values; nothing is backfilled and nothing is dropped.

Revision ID: e7a1c4b90d26
Revises: d5e2c9a41f83
Create Date: 2026-08-14 21:30:00.000000+00:00
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "e7a1c4b90d26"
down_revision: str | None = "d5e2c9a41f83"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.alter_column(
        "subscriptions",
        "provider_customer_id",
        existing_type=sa.String(length=80),
        nullable=True,
    )


def downgrade() -> None:
    # Rows written after the upgrade may hold NULL, and a NOT NULL cannot be
    # restored over them. Filling with a marker would put a string that is not
    # a Razorpay customer id into a column every lookup treats as one, so the
    # honest downgrade refuses rather than corrupts.
    connection = op.get_bind()
    unset = connection.execute(
        sa.text("SELECT count(*) FROM subscriptions WHERE provider_customer_id IS NULL")
    ).scalar_one()
    if unset:
        raise RuntimeError(
            f"{unset} subscription(s) have no provider_customer_id. "
            "Resolve them before restoring the NOT NULL."
        )
    op.alter_column(
        "subscriptions",
        "provider_customer_id",
        existing_type=sa.String(length=80),
        nullable=False,
    )
