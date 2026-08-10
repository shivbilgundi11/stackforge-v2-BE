"""projects and project items

Revision ID: c9c0f504c1c4
Revises: fd9b9bad9cd5
Create Date: 2026-08-10 07:19:39.121463+00:00
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql
revision: str = 'c9c0f504c1c4'
down_revision: str | None = 'fd9b9bad9cd5'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table('projects',
    sa.Column('id', sa.String(length=64), nullable=False),
    sa.Column('user_id', sa.String(length=64), nullable=False),
    sa.Column('name', sa.String(length=160), nullable=False),
    sa.Column('description', sa.Text(), nullable=True),
    sa.Column('use_case', sa.String(length=60), nullable=True),
    sa.Column('session_state', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('archived_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], name=op.f('fk_projects_user_id_users'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_projects'))
    )
    op.create_index('ix_projects_user_id_updated_at', 'projects', ['user_id', 'updated_at'], unique=False)
    # Full-text search over the fields a person would type. Declared here
    # rather than on the model: `to_tsvector('english', …)` needs a regconfig
    # literal that SQLAlchemy cannot render into an index definition.
    op.execute(
        "CREATE INDEX ix_projects_search ON projects USING gin ("
        "to_tsvector('english', coalesce(name, '') || ' ' || coalesce(description, '')))"
    )
    op.create_table('project_items',
    sa.Column('id', sa.String(length=64), nullable=False),
    sa.Column('project_id', sa.String(length=64), nullable=False),
    sa.Column('item_type', sa.Enum('run', 'stack', 'artifact', 'template', name='project_item_type'), nullable=False),
    sa.Column('item_id', sa.String(length=64), nullable=False),
    sa.Column('position', sa.Integer(), nullable=False),
    sa.Column('pinned', sa.Boolean(), nullable=False),
    sa.Column('note', sa.Text(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['project_id'], ['projects.id'], name=op.f('fk_project_items_project_id_projects'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_project_items')),
    sa.UniqueConstraint('project_id', 'item_type', 'item_id', name='uq_project_items_project_item')
    )
    op.create_index('ix_project_items_project_id_position', 'project_items', ['project_id', 'position'], unique=False)


def downgrade() -> None:
    op.drop_index('ix_project_items_project_id_position', table_name='project_items')
    op.drop_table('project_items')
    op.execute("DROP INDEX IF EXISTS ix_projects_search")
    op.drop_index('ix_projects_user_id_updated_at', table_name='projects')
    op.drop_table('projects')
    # Dropping the table leaves the enum type behind, and the next upgrade
    # then fails on a type that already exists.
    op.execute("DROP TYPE IF EXISTS project_item_type")
