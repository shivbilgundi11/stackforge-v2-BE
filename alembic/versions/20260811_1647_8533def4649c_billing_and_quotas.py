"""billing and quotas

Revision ID: 8533def4649c
Revises: d72e5b18c934
Create Date: 2026-08-11 16:47:49.058962+00:00
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = '8533def4649c'
down_revision: str | None = 'd72e5b18c934'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # `plan` was created by the identity migration and is reused by three
    # columns here. create_type=False on every one of them: the first CREATE
    # TYPE already happened, and a second would fail the whole migration.
    plan = postgresql.ENUM('free', 'pro', 'team', 'enterprise', name='plan', create_type=False)

    op.create_table('plan_quotas',
    sa.Column('id', sa.String(length=64), nullable=False),
    sa.Column('plan', plan, nullable=False),
    sa.Column('anonymous', sa.Boolean(), server_default=sa.text('false'), nullable=False),
    # First use of `quota_metric`, so this column is the one that creates it.
    sa.Column('metric', sa.Enum('tool_runs_per_day', 'ai_calls_per_day', 'projects', 'saved_stacks', 'exports_per_month', 'seats', name='quota_metric'), nullable=False),
    sa.Column('limit_value', sa.Integer(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint("anonymous IS FALSE OR plan = 'free'", name=op.f('ck_plan_quotas_anonymous_is_free')),
    sa.CheckConstraint('limit_value IS NULL OR limit_value >= 0', name=op.f('ck_plan_quotas_limit_not_negative')),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_plan_quotas')),
    sa.UniqueConstraint('plan', 'anonymous', 'metric', name='uq_plan_quotas_plan_anonymous_metric')
    )

    op.create_table('stripe_events',
    sa.Column('id', sa.String(length=80), nullable=False),
    sa.Column('type', sa.String(length=80), nullable=False),
    sa.Column('payload', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('received_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('processed_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('attempts', sa.Integer(), server_default=sa.text('0'), nullable=False),
    sa.Column('error', sa.Text(), nullable=True),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_stripe_events'))
    )
    op.create_index('ix_stripe_events_unprocessed', 'stripe_events', ['received_at'], unique=False, postgresql_where=sa.text('processed_at IS NULL'))

    op.create_table('subscriptions',
    sa.Column('id', sa.String(length=64), nullable=False),
    sa.Column('user_id', sa.String(length=64), nullable=True),
    sa.Column('organization_id', sa.String(length=64), nullable=True),
    sa.Column('stripe_customer_id', sa.String(length=80), nullable=False),
    sa.Column('stripe_subscription_id', sa.String(length=80), nullable=True),
    sa.Column('stripe_price_id', sa.String(length=80), nullable=True),
    sa.Column('plan', plan, nullable=False),
    sa.Column('status', sa.Enum('trialing', 'active', 'past_due', 'canceled', 'incomplete', name='subscription_status'), nullable=False),
    sa.Column('seats', sa.Integer(), server_default=sa.text('1'), nullable=False),
    sa.Column('current_period_start', sa.DateTime(timezone=True), nullable=True),
    sa.Column('current_period_end', sa.DateTime(timezone=True), nullable=True),
    sa.Column('cancel_at_period_end', sa.Boolean(), server_default=sa.text('false'), nullable=False),
    sa.Column('trial_ends_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('canceled_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('past_due_since', sa.DateTime(timezone=True), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint('num_nonnulls(user_id, organization_id) = 1', name=op.f('ck_subscriptions_exactly_one_owner')),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], name=op.f('fk_subscriptions_user_id_users'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_subscriptions')),
    sa.UniqueConstraint('stripe_subscription_id', name=op.f('uq_subscriptions_stripe_subscription_id'))
    )
    op.create_index('ix_subscriptions_past_due_since', 'subscriptions', ['past_due_since'], unique=False, postgresql_where=sa.text("status = 'past_due'"))
    op.create_index('ix_subscriptions_stripe_customer_id', 'subscriptions', ['stripe_customer_id'], unique=False)
    op.create_index('ix_subscriptions_trial_ends_at', 'subscriptions', ['trial_ends_at'], unique=False, postgresql_where=sa.text("status = 'trialing'"))
    # Partial unique: one non-canceled subscription per user. Canceled rows
    # stay for history and must not collide with the next signup.
    op.create_index('uq_subscriptions_user_live', 'subscriptions', ['user_id'], unique=True, postgresql_where=sa.text("user_id IS NOT NULL AND status <> 'canceled'"))

    op.create_table('usage_records',
    sa.Column('id', sa.String(length=64), nullable=False),
    sa.Column('user_id', sa.String(length=64), nullable=True),
    sa.Column('anonymous_session_id', sa.String(length=64), nullable=True),
    sa.Column('organization_id', sa.String(length=64), nullable=True),
    sa.Column('metric', postgresql.ENUM('tool_runs_per_day', 'ai_calls_per_day', 'projects', 'saved_stacks', 'exports_per_month', 'seats', name='quota_metric', create_type=False), nullable=False),
    sa.Column('quantity', sa.Integer(), nullable=False),
    sa.Column('period_start', sa.Date(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint('num_nonnulls(user_id, anonymous_session_id, organization_id) = 1', name=op.f('ck_usage_records_exactly_one_owner')),
    sa.ForeignKeyConstraint(['anonymous_session_id'], ['anonymous_sessions.id'], name=op.f('fk_usage_records_anonymous_session_id_anonymous_sessions'), ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], name=op.f('fk_usage_records_user_id_users'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_usage_records'))
    )
    op.create_index('ix_usage_records_anonymous_session_id_metric_period_start', 'usage_records', ['anonymous_session_id', 'metric', 'period_start'], unique=False)
    op.create_index('ix_usage_records_user_id_metric_period_start', 'usage_records', ['user_id', 'metric', 'period_start'], unique=False)

    # Autogenerate also proposed dropping `ix_projects_search` and
    # `ix_templates_search`. Both are expression indexes declared with raw SQL
    # in their own migrations, which autogenerate cannot reflect back into a
    # model — so it reads them as orphans on every subsequent revision. Those
    # drops are deleted here, and will need deleting again next time.


def downgrade() -> None:
    op.drop_index('ix_usage_records_user_id_metric_period_start', table_name='usage_records')
    op.drop_index('ix_usage_records_anonymous_session_id_metric_period_start', table_name='usage_records')
    op.drop_table('usage_records')
    op.drop_index('uq_subscriptions_user_live', table_name='subscriptions', postgresql_where=sa.text("user_id IS NOT NULL AND status <> 'canceled'"))
    op.drop_index('ix_subscriptions_trial_ends_at', table_name='subscriptions', postgresql_where=sa.text("status = 'trialing'"))
    op.drop_index('ix_subscriptions_stripe_customer_id', table_name='subscriptions')
    op.drop_index('ix_subscriptions_past_due_since', table_name='subscriptions', postgresql_where=sa.text("status = 'past_due'"))
    op.drop_table('subscriptions')
    op.drop_index('ix_stripe_events_unprocessed', table_name='stripe_events', postgresql_where=sa.text('processed_at IS NULL'))
    op.drop_table('stripe_events')
    op.drop_table('plan_quotas')
    # Dropping the tables leaves the enum types behind, and the next upgrade
    # then fails on a type that already exists. `plan` is not dropped — it
    # belongs to `users` and predates this revision.
    op.execute("DROP TYPE IF EXISTS subscription_status")
    op.execute("DROP TYPE IF EXISTS quota_metric")
