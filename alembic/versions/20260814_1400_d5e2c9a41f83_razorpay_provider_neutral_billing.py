"""razorpay: provider-neutral billing columns

Renames rather than drop-and-recreate, so existing subscriptions and the
idempotency ledger survive the provider switch. The ids inside the columns are
still Stripe's until each account re-subscribes; nothing here rewrites them,
because a `cus_…` in `provider_customer_id` is accurate history and inventing a
Razorpay id for it would not be.

Revision ID: d5e2c9a41f83
Revises: c1d4a8b37e05
Create Date: 2026-08-14 14:00:00.000000+00:00
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = 'd5e2c9a41f83'
down_revision: str | None = 'c1d4a8b37e05'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.alter_column('subscriptions', 'stripe_customer_id', new_column_name='provider_customer_id')
    op.alter_column(
        'subscriptions', 'stripe_subscription_id', new_column_name='provider_subscription_id'
    )
    op.alter_column('subscriptions', 'stripe_price_id', new_column_name='provider_plan_id')

    # The index and the unique constraint carry the old name with them; a
    # rename keeps them attached to the same physical index rather than
    # rebuilding one on a table that may be large.
    op.execute(
        'ALTER INDEX ix_subscriptions_stripe_customer_id '
        'RENAME TO ix_subscriptions_provider_customer_id'
    )
    op.execute(
        'ALTER INDEX uq_subscriptions_stripe_subscription_id '
        'RENAME TO uq_subscriptions_provider_subscription_id'
    )

    op.rename_table('stripe_events', 'billing_events')
    op.execute('ALTER INDEX pk_stripe_events RENAME TO pk_billing_events')
    op.execute('ALTER INDEX ix_stripe_events_unprocessed RENAME TO ix_billing_events_unprocessed')


def downgrade() -> None:
    op.execute('ALTER INDEX ix_billing_events_unprocessed RENAME TO ix_stripe_events_unprocessed')
    op.execute('ALTER INDEX pk_billing_events RENAME TO pk_stripe_events')
    op.rename_table('billing_events', 'stripe_events')

    op.execute(
        'ALTER INDEX uq_subscriptions_provider_subscription_id '
        'RENAME TO uq_subscriptions_stripe_subscription_id'
    )
    op.execute(
        'ALTER INDEX ix_subscriptions_provider_customer_id '
        'RENAME TO ix_subscriptions_stripe_customer_id'
    )

    op.alter_column('subscriptions', 'provider_plan_id', new_column_name='stripe_price_id')
    op.alter_column(
        'subscriptions', 'provider_subscription_id', new_column_name='stripe_subscription_id'
    )
    op.alter_column('subscriptions', 'provider_customer_id', new_column_name='stripe_customer_id')
