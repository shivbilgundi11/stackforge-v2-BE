"""The anonymous tier is gone: every surface requires an account.

The product used to mint a session per visitor and grant it five tool runs a
day, so `tool_runs`, `exports`, `usage_records`, `ai_calls` and `auth_events`
each carried a nullable `anonymous_session_id` beside their nullable `user_id`,
held to exactly one by a check constraint. `plan_quotas` carried a second
discriminator for the same reason — the anonymous allowance was a row, not a
plan.

The app shell is now account-only, so there is one kind of owner. The two
nullable columns collapse to one NOT NULL column, and the check constraints
that policed the union go with them.

**This deletes data.** Any row still owned by an anonymous session — a run, an
export, a usage record — cannot be made to satisfy `user_id NOT NULL`, and
there is no account to move it to: claiming happened at login and only for a
visitor who signed up in the same browser. The deletes below are counted and
logged so the number is visible in the migration output rather than inferred
afterwards. Take a dump first if any of it matters.

`auth_event_type` keeps its `anonymous_claimed` value. Removing a value from a
Postgres enum means recreating the type and rewriting every row that uses it,
and the audit trail is precisely the thing that should not be rewritten.

Revision ID: b6d24f8ac015
Revises: a4f1c9e73b28
Create Date: 2026-09-03 12:00:00.000000+00:00
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import INET

from alembic import op

revision: str = "b6d24f8ac015"
down_revision: str | None = "a4f1c9e73b28"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _purge(table: str) -> None:
    """Delete the rows that have no account to belong to, and say how many."""
    result = op.get_bind().execute(sa.text(f"DELETE FROM {table} WHERE user_id IS NULL"))
    print(f"  {table}: deleted {result.rowcount} anonymously-owned row(s)")


def upgrade() -> None:
    # ── Rows first. Everything after this assumes an owner exists. ──────────
    for table in ("tool_runs", "exports", "usage_records"):
        _purge(table)

    # ── tool_runs ──────────────────────────────────────────────────────────
    op.drop_constraint(op.f("ck_tool_runs_exactly_one_owner"), "tool_runs", type_="check")
    op.drop_index(
        "ix_tool_runs_anonymous_session_id",
        table_name="tool_runs",
        postgresql_where=sa.text("anonymous_session_id IS NOT NULL"),
    )
    op.drop_constraint(
        op.f("fk_tool_runs_anonymous_session_id_anonymous_sessions"), "tool_runs", type_="foreignkey"
    )
    op.drop_column("tool_runs", "anonymous_session_id")
    op.alter_column("tool_runs", "user_id", existing_type=sa.String(length=64), nullable=False)

    # ── exports ────────────────────────────────────────────────────────────
    op.drop_constraint(op.f("ck_exports_exactly_one_owner"), "exports", type_="check")
    op.drop_constraint(
        op.f("fk_exports_anonymous_session_id_anonymous_sessions"), "exports", type_="foreignkey"
    )
    op.drop_column("exports", "anonymous_session_id")
    op.alter_column("exports", "user_id", existing_type=sa.String(length=64), nullable=False)

    # ── usage_records ──────────────────────────────────────────────────────
    # `organization_id` stays in the union: an org-owned meter is a real owner,
    # unlike an anonymous one.
    op.drop_constraint(op.f("ck_usage_records_exactly_one_owner"), "usage_records", type_="check")
    op.drop_index(
        "ix_usage_records_anonymous_session_id_metric_period_start", table_name="usage_records"
    )
    op.drop_constraint(
        op.f("fk_usage_records_anonymous_session_id_anonymous_sessions"),
        "usage_records",
        type_="foreignkey",
    )
    op.drop_column("usage_records", "anonymous_session_id")
    op.create_check_constraint(
        "exactly_one_owner",
        "usage_records",
        "num_nonnulls(user_id, organization_id) = 1",
    )

    # ── ai_calls, auth_events ──────────────────────────────────────────────
    # Neither column ever carried a foreign key: both tables outlive the rows
    # they describe on purpose, and a cascade would have deleted the ledger.
    op.drop_column("ai_calls", "anonymous_session_id")
    op.drop_column("auth_events", "anonymous_session_id")

    # ── plan_quotas ────────────────────────────────────────────────────────
    op.execute(sa.text("DELETE FROM plan_quotas WHERE anonymous IS TRUE"))
    op.drop_constraint(op.f("uq_plan_quotas_plan_anonymous_metric"), "plan_quotas", type_="unique")
    op.drop_constraint(op.f("ck_plan_quotas_anonymous_is_free"), "plan_quotas", type_="check")
    op.drop_column("plan_quotas", "anonymous")
    op.create_unique_constraint(op.f("uq_plan_quotas_plan_metric"), "plan_quotas", ["plan", "metric"])

    # ── anonymous_sessions ─────────────────────────────────────────────────
    op.drop_index("ix_anonymous_sessions_claimed_by_user_id", table_name="anonymous_sessions")
    op.drop_table("anonymous_sessions")


def downgrade() -> None:
    """Restores the shape, never the data.

    The rows this migration deleted are gone; what comes back is a schema that
    can hold anonymous ownership again, with every existing row owned by its
    user. That is the honest limit of a downgrade past a destructive step.
    """
    op.create_table(
        "anonymous_sessions",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("ip", INET(), nullable=True),
        sa.Column("user_agent", sa.Text(), nullable=True),
        sa.Column("last_seen_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("claimed_by_user_id", sa.String(length=64), nullable=True),
        sa.Column("claimed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["claimed_by_user_id"],
            ["users.id"],
            name=op.f("fk_anonymous_sessions_claimed_by_user_id_users"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_anonymous_sessions")),
    )
    op.create_index(
        "ix_anonymous_sessions_claimed_by_user_id",
        "anonymous_sessions",
        ["claimed_by_user_id"],
        unique=False,
    )

    # ── plan_quotas ────────────────────────────────────────────────────────
    op.drop_constraint(op.f("uq_plan_quotas_plan_metric"), "plan_quotas", type_="unique")
    op.add_column(
        "plan_quotas",
        sa.Column("anonymous", sa.Boolean(), server_default=sa.text("false"), nullable=False),
    )
    op.create_check_constraint(
        "anonymous_is_free", "plan_quotas", "anonymous IS FALSE OR plan = 'free'"
    )
    op.create_unique_constraint(
        op.f("uq_plan_quotas_plan_anonymous_metric"), "plan_quotas", ["plan", "anonymous", "metric"]
    )

    # ── ai_calls, auth_events ──────────────────────────────────────────────
    op.add_column(
        "auth_events", sa.Column("anonymous_session_id", sa.String(length=64), nullable=True)
    )
    op.add_column(
        "ai_calls", sa.Column("anonymous_session_id", sa.String(length=64), nullable=True)
    )

    # ── usage_records ──────────────────────────────────────────────────────
    op.drop_constraint(op.f("ck_usage_records_exactly_one_owner"), "usage_records", type_="check")
    op.add_column(
        "usage_records", sa.Column("anonymous_session_id", sa.String(length=64), nullable=True)
    )
    op.create_foreign_key(
        op.f("fk_usage_records_anonymous_session_id_anonymous_sessions"),
        "usage_records",
        "anonymous_sessions",
        ["anonymous_session_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_index(
        "ix_usage_records_anonymous_session_id_metric_period_start",
        "usage_records",
        ["anonymous_session_id", "metric", "period_start"],
        unique=False,
    )
    op.create_check_constraint(
        "exactly_one_owner",
        "usage_records",
        "num_nonnulls(user_id, anonymous_session_id, organization_id) = 1",
    )

    # ── exports ────────────────────────────────────────────────────────────
    op.alter_column("exports", "user_id", existing_type=sa.String(length=64), nullable=True)
    op.add_column("exports", sa.Column("anonymous_session_id", sa.String(length=64), nullable=True))
    op.create_foreign_key(
        op.f("fk_exports_anonymous_session_id_anonymous_sessions"),
        "exports",
        "anonymous_sessions",
        ["anonymous_session_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_check_constraint(
        "exactly_one_owner",
        "exports",
        "num_nonnulls(user_id, anonymous_session_id) = 1",
    )

    # ── tool_runs ──────────────────────────────────────────────────────────
    op.alter_column("tool_runs", "user_id", existing_type=sa.String(length=64), nullable=True)
    op.add_column(
        "tool_runs", sa.Column("anonymous_session_id", sa.String(length=64), nullable=True)
    )
    op.create_foreign_key(
        op.f("fk_tool_runs_anonymous_session_id_anonymous_sessions"),
        "tool_runs",
        "anonymous_sessions",
        ["anonymous_session_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_index(
        "ix_tool_runs_anonymous_session_id",
        "tool_runs",
        ["anonymous_session_id"],
        unique=False,
        postgresql_where=sa.text("anonymous_session_id IS NOT NULL"),
    )
    op.create_check_constraint(
        "exactly_one_owner",
        "tool_runs",
        "num_nonnulls(user_id, anonymous_session_id) = 1",
    )
