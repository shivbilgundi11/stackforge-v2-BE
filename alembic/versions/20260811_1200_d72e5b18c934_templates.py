"""templates

Revision ID: d72e5b18c934
Revises: b41c7ae90f22
Create Date: 2026-08-11 12:00:00.000000+00:00
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = 'd72e5b18c934'
down_revision: str | None = 'b41c7ae90f22'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table('templates',
    sa.Column('id', sa.String(length=64), nullable=False),
    sa.Column('slug', sa.String(length=120), nullable=False),
    sa.Column('title', sa.String(length=200), nullable=False),
    sa.Column('category', sa.Enum('stack', 'blueprint', 'code-starter', 'prompt', 'config', 'checklist', 'business', name='template_category'), nullable=False),
    sa.Column('difficulty', sa.Enum('beginner', 'intermediate', 'advanced', name='template_difficulty'), nullable=False),
    sa.Column('summary', sa.Text(), nullable=False),
    sa.Column('content_markdown', sa.Text(), nullable=False),
    sa.Column('files', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('stack_input', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('use_cases', postgresql.ARRAY(sa.String(length=40)), nullable=False),
    sa.Column('tags', postgresql.ARRAY(sa.String(length=40)), nullable=False),
    sa.Column('related_tools', postgresql.ARRAY(sa.String(length=80)), nullable=False),
    sa.Column('is_premium', sa.Boolean(), nullable=False),
    sa.Column('organization_id', sa.String(length=64), nullable=True),
    sa.Column('view_count', sa.Integer(), nullable=False),
    sa.Column('copy_count', sa.Integer(), nullable=False),
    sa.Column('published_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_templates')),
    sa.UniqueConstraint('slug', name=op.f('uq_templates_slug'))
    )
    op.create_index('ix_templates_category_published_at', 'templates', ['category', 'published_at'], unique=False)
    op.create_index('ix_templates_tags', 'templates', ['tags'], unique=False, postgresql_using='gin')
    op.create_index('ix_templates_use_cases', 'templates', ['use_cases'], unique=False, postgresql_using='gin')
    op.create_index('ix_templates_organization_id', 'templates', ['organization_id'], unique=False)
    # Full text over the three fields a person would actually type into a
    # search box. Declared here rather than on the model: `to_tsvector` needs a
    # regconfig literal SQLAlchemy cannot render into an index definition.
    #
    # `title` is weighted above `summary` above the body, so a template whose
    # *title* is "RAG Chatbot" outranks one that mentions RAG in paragraph
    # eleven. Without weighting, the longest document wins every query, which
    # is the opposite of useful.
    op.execute(
        "CREATE INDEX ix_templates_search ON templates USING gin (("
        "setweight(to_tsvector('english', coalesce(title, '')), 'A') || "
        "setweight(to_tsvector('english', coalesce(summary, '')), 'B') || "
        "setweight(to_tsvector('english', coalesce(content_markdown, '')), 'C')))"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_templates_search")
    op.drop_index('ix_templates_organization_id', table_name='templates')
    op.drop_index('ix_templates_use_cases', table_name='templates', postgresql_using='gin')
    op.drop_index('ix_templates_tags', table_name='templates', postgresql_using='gin')
    op.drop_index('ix_templates_category_published_at', table_name='templates')
    op.drop_table('templates')
    # Dropping the table leaves the enum types behind, and the next upgrade
    # then fails on a type that already exists.
    op.execute("DROP TYPE IF EXISTS template_difficulty")
    op.execute("DROP TYPE IF EXISTS template_category")
