"""razorpay: an upgrade's new subscription waits in its own column

Razorpay has no call that changes the plan on a subscription, so moving from
Pro to Team creates a second one — and both bill until one is cancelled. The
checkout wrote the new id straight over `provider_subscription_id`, which left
the old subscription live at the provider with its id recorded nowhere. The
account paid for both plans and nothing in the product could see it, because
the only reference to what was still charging had been overwritten.

`pending_subscription_id` holds the in-flight one until its mandate is
authorized. The row goes on tracking the subscription that is actually paying
until then, and `_on_subscription_changed` promotes the new one and cancels the
one it replaces.

Nullable with no backfill: an in-flight upgrade is a state that lasts seconds,
and there is no way to reconstruct one that was already lost. Accounts already
double-subscribed are repaired with `razorpay-reconcile`, not by a migration
guessing which subscription was meant to win.

Revision ID: f3b8d5e21c47
Revises: e7a1c4b90d26
Create Date: 2026-08-17 15:30:00.000000+00:00
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "f3b8d5e21c47"
down_revision: str | None = "e7a1c4b90d26"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "subscriptions",
        sa.Column("pending_subscription_id", sa.String(length=80), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("subscriptions", "pending_subscription_id")
