"""plan selection at signup

Revision ID: c1d4a8b37e05
Revises: ba7501f6243d
Create Date: 2026-08-14 09:00:00.000000+00:00
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = 'c1d4a8b37e05'
down_revision: str | None = 'ba7501f6243d'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # `plan` predates this revision (identity migration); reuse, never create.
    plan = postgresql.ENUM('free', 'pro', 'team', 'enterprise', name='plan', create_type=False)

    # Nullable with no server default: null is "owes nothing", which is the
    # state every existing row is already in. Backfilling would be wrong —
    # nobody who signed up before this revision chose a plan they have not
    # paid for.
    op.add_column('users', sa.Column('pending_plan', plan, nullable=True))
    op.add_column('users', sa.Column('pending_interval', sa.String(length=10), nullable=True))

    # Partial: the wall only ever asks for the rows that owe a checkout, and
    # in a healthy account base that is a small minority of `users`.
    op.create_index(
        'ix_users_pending_plan',
        'users',
        ['pending_plan'],
        unique=False,
        postgresql_where=sa.text('pending_plan IS NOT NULL'),
    )


def downgrade() -> None:
    op.drop_index('ix_users_pending_plan', table_name='users', postgresql_where=sa.text('pending_plan IS NOT NULL'))
    op.drop_column('users', 'pending_interval')
    op.drop_column('users', 'pending_plan')
