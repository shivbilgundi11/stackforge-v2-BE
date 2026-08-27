"""Two catalog categories for the model's own infrastructure, and residency.

`gpu-cloud` and `guardrails` are the two optional Stack Architect roles from
M25. `residency` is the only genuinely new fact in that module: `gpu_pricing`
records the region an *instance* was priced in, which is not the same claim as
"this vendor can be operated in the EU", and the tool catalog had no notion of
geography at all.

Empty is the default and is not a claim. On self-hostable software it means
unconstrained; on a managed vendor it means nothing is on file, and the
engine eliminates rather than passes. Backfilling it is editorial work with a
source per row, not something a migration can invent.

Revision ID: a4f1c9e73b28
Revises: c8a5f2d7b310
Create Date: 2026-08-27 10:00:00.000000+00:00
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "a4f1c9e73b28"
down_revision: str | None = "c8a5f2d7b310"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Postgres will not add an enum value inside a transaction block on older
    # servers; `ALTER TYPE ... ADD VALUE IF NOT EXISTS` is safe from 12 on and
    # keeps the migration re-runnable after a partial failure.
    op.execute("ALTER TYPE tool_category ADD VALUE IF NOT EXISTS 'gpu-cloud'")
    op.execute("ALTER TYPE tool_category ADD VALUE IF NOT EXISTS 'guardrails'")

    op.add_column(
        "tool_catalog",
        sa.Column(
            "residency",
            sa.ARRAY(sa.String(length=8)),
            nullable=False,
            server_default="{}",
        ),
    )
    # The server default exists only to give the rows already in the table a
    # value under NOT NULL; dropping it again keeps the column's default where
    # the rest of them live, in the model. Leaving it on is drift `alembic
    # check` reports on every later migration.
    op.alter_column("tool_catalog", "residency", server_default=None)

    op.create_index(
        "ix_tool_catalog_residency",
        "tool_catalog",
        ["residency"],
        postgresql_using="gin",
    )


def downgrade() -> None:
    op.drop_index("ix_tool_catalog_residency", table_name="tool_catalog")
    op.drop_column("tool_catalog", "residency")
    # The enum values stay. Dropping one means rebuilding the type and
    # rewriting every row that uses it, and a down-migration that rewrites a
    # table under load is worse than an unused label.
