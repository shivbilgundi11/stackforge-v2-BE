"""exports and share links

Revision ID: b41c7ae90f22
Revises: c9c0f504c1c4
Create Date: 2026-08-11 09:00:00.000000+00:00
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = 'b41c7ae90f22'
down_revision: str | None = 'c9c0f504c1c4'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    source_type = sa.Enum('run', 'stack', name='export_source_type')

    op.create_table('exports',
    sa.Column('id', sa.String(length=64), nullable=False),
    sa.Column('user_id', sa.String(length=64), nullable=True),
    sa.Column('anonymous_session_id', sa.String(length=64), nullable=True),
    sa.Column('source_type', source_type, nullable=False),
    sa.Column('source_id', sa.String(length=64), nullable=False),
    sa.Column('artifact_type', sa.String(length=60), nullable=True),
    sa.Column('format', sa.Enum('markdown', 'json', 'yaml', 'csv', 'pdf', 'zip', name='export_format'), nullable=False),
    sa.Column('status', sa.Enum('pending', 'ready', 'failed', name='export_status'), nullable=False),
    sa.Column('filename', sa.String(length=200), nullable=False),
    sa.Column('content_type', sa.String(length=120), nullable=False),
    sa.Column('size_bytes', sa.Integer(), nullable=False),
    sa.Column('content', sa.LargeBinary(), nullable=True),
    sa.Column('error', sa.Text(), nullable=True),
    sa.Column('expires_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
    sa.CheckConstraint('num_nonnulls(user_id, anonymous_session_id) = 1', name=op.f('ck_exports_exactly_one_owner')),
    sa.ForeignKeyConstraint(['anonymous_session_id'], ['anonymous_sessions.id'], name=op.f('fk_exports_anonymous_session_id_anonymous_sessions'), ondelete='SET NULL'),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], name=op.f('fk_exports_user_id_users'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_exports'))
    )
    op.create_index('ix_exports_user_id_created_at', 'exports', ['user_id', sa.text('created_at DESC')], unique=False)
    op.create_index('ix_exports_expires_at', 'exports', ['expires_at'], unique=False)
    op.create_index('ix_exports_status_pending', 'exports', ['status'], unique=False, postgresql_where=sa.text("status = 'pending'"))

    op.create_table('share_links',
    sa.Column('id', sa.String(length=64), nullable=False),
    sa.Column('user_id', sa.String(length=64), nullable=False),
    sa.Column('token', sa.String(length=64), nullable=False),
    # The enum type was created by `exports` above. Referencing it with
    # create_type=False is what stops this second column emitting a duplicate
    # CREATE TYPE, which fails the whole migration.
    sa.Column('target_type', sa.Enum('run', 'stack', name='export_source_type', create_type=False), nullable=False),
    sa.Column('target_id', sa.String(length=64), nullable=False),
    sa.Column('artifact_type', sa.String(length=60), nullable=True),
    sa.Column('title', sa.String(length=200), nullable=False),
    sa.Column('expires_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('revoked_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('view_count', sa.Integer(), nullable=False),
    sa.Column('last_viewed_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], name=op.f('fk_share_links_user_id_users'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_share_links')),
    sa.UniqueConstraint('token', name=op.f('uq_share_links_token'))
    )
    op.create_index('ix_share_links_user_id_created_at', 'share_links', ['user_id', sa.text('created_at DESC')], unique=False)
    op.create_index('ix_share_links_target', 'share_links', ['target_type', 'target_id'], unique=False)


def downgrade() -> None:
    op.drop_index('ix_share_links_target', table_name='share_links')
    op.drop_index('ix_share_links_user_id_created_at', table_name='share_links')
    op.drop_table('share_links')
    op.drop_index('ix_exports_status_pending', table_name='exports', postgresql_where=sa.text("status = 'pending'"))
    op.drop_index('ix_exports_expires_at', table_name='exports')
    op.drop_index('ix_exports_user_id_created_at', table_name='exports')
    op.drop_table('exports')
    # Dropping a table leaves its enum types behind, and the next upgrade then
    # fails on a type that already exists.
    op.execute("DROP TYPE IF EXISTS export_status")
    op.execute("DROP TYPE IF EXISTS export_format")
    op.execute("DROP TYPE IF EXISTS export_source_type")
