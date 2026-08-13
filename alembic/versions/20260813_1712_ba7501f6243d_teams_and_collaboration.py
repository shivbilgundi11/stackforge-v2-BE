"""teams and collaboration

Revision ID: ba7501f6243d
Revises: 8533def4649c
Create Date: 2026-08-13 17:12:00.000000+00:00
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = 'ba7501f6243d'
down_revision: str | None = '8533def4649c'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # `plan` predates this revision (identity migration); reuse, never create.
    plan = postgresql.ENUM('free', 'pro', 'team', 'enterprise', name='plan', create_type=False)

    op.create_table('organizations',
    sa.Column('id', sa.String(length=64), nullable=False),
    sa.Column('name', sa.String(length=160), nullable=False),
    sa.Column('slug', sa.String(length=80), nullable=False),
    # RESTRICT: an owner cannot be hard-deleted out from under the org —
    # ownership is transferred first, and the service enforces that order.
    sa.Column('owner_id', sa.String(length=64), nullable=False),
    sa.Column('plan', plan, server_default='free', nullable=False),
    sa.Column('seats_purchased', sa.Integer(), server_default=sa.text('1'), nullable=False),
    sa.Column('settings', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('deleted_at', sa.DateTime(timezone=True), nullable=True),
    sa.ForeignKeyConstraint(['owner_id'], ['users.id'], name=op.f('fk_organizations_owner_id_users'), ondelete='RESTRICT'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_organizations')),
    sa.UniqueConstraint('slug', name=op.f('uq_organizations_slug'))
    )
    op.create_index('ix_organizations_active', 'organizations', ['deleted_at'], unique=False, postgresql_where=sa.text('deleted_at IS NULL'))

    op.create_table('organization_members',
    sa.Column('id', sa.String(length=64), nullable=False),
    sa.Column('organization_id', sa.String(length=64), nullable=False),
    sa.Column('user_id', sa.String(length=64), nullable=False),
    # First use of `org_role`, so this column is the one that creates it.
    sa.Column('role', sa.Enum('owner', 'admin', 'member', 'viewer', name='org_role'), nullable=False),
    sa.Column('invited_by_user_id', sa.String(length=64), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['invited_by_user_id'], ['users.id'], name=op.f('fk_organization_members_invited_by_user_id_users'), ondelete='SET NULL'),
    sa.ForeignKeyConstraint(['organization_id'], ['organizations.id'], name=op.f('fk_organization_members_organization_id_organizations'), ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], name=op.f('fk_organization_members_user_id_users'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_organization_members')),
    sa.UniqueConstraint('organization_id', 'user_id', name='uq_organization_members_organization_id_user_id')
    )
    op.create_index('ix_organization_members_user_id', 'organization_members', ['user_id'], unique=False)
    # Exactly one owner per organization — the invariant the role model leans
    # on, held at the database level so no code path can create a second.
    op.create_index('uq_organization_members_one_owner', 'organization_members', ['organization_id'], unique=True, postgresql_where=sa.text("role = 'owner'"))

    op.create_table('invitations',
    sa.Column('id', sa.String(length=64), nullable=False),
    sa.Column('organization_id', sa.String(length=64), nullable=False),
    sa.Column('email', postgresql.CITEXT(), nullable=False),
    sa.Column('role', postgresql.ENUM('owner', 'admin', 'member', 'viewer', name='org_role', create_type=False), nullable=False),
    sa.Column('token_hash', sa.String(length=64), nullable=False),
    sa.Column('invited_by_user_id', sa.String(length=64), nullable=True),
    sa.Column('expires_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('accepted_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('accepted_by_user_id', sa.String(length=64), nullable=True),
    sa.Column('revoked_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    # Ownership is transferred, never granted by invite.
    sa.CheckConstraint("role <> 'owner'", name=op.f('ck_invitations_no_owner_invites')),
    sa.ForeignKeyConstraint(['accepted_by_user_id'], ['users.id'], name=op.f('fk_invitations_accepted_by_user_id_users'), ondelete='SET NULL'),
    sa.ForeignKeyConstraint(['invited_by_user_id'], ['users.id'], name=op.f('fk_invitations_invited_by_user_id_users'), ondelete='SET NULL'),
    sa.ForeignKeyConstraint(['organization_id'], ['organizations.id'], name=op.f('fk_invitations_organization_id_organizations'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_invitations')),
    sa.UniqueConstraint('token_hash', name=op.f('uq_invitations_token_hash'))
    )
    op.create_index('ix_invitations_organization_id', 'invitations', ['organization_id'], unique=False)
    # One *open* invite per (organization, email). Accepted and revoked rows
    # are history and must not block a re-invite.
    op.create_index('uq_invitations_organization_id_email_open', 'invitations', ['organization_id', 'email'], unique=True, postgresql_where=sa.text('accepted_at IS NULL AND revoked_at IS NULL'))

    op.create_table('comments',
    sa.Column('id', sa.String(length=64), nullable=False),
    # First use of `team_resource_type`.
    sa.Column('resource_type', sa.Enum('stack', 'run', 'project', name='team_resource_type'), nullable=False),
    sa.Column('resource_id', sa.String(length=64), nullable=False),
    sa.Column('organization_id', sa.String(length=64), nullable=False),
    sa.Column('author_id', sa.String(length=64), nullable=True),
    sa.Column('body', sa.Text(), nullable=False),
    sa.Column('parent_id', sa.String(length=64), nullable=True),
    sa.Column('resolved_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('deleted_at', sa.DateTime(timezone=True), nullable=True),
    sa.ForeignKeyConstraint(['author_id'], ['users.id'], name=op.f('fk_comments_author_id_users'), ondelete='SET NULL'),
    sa.ForeignKeyConstraint(['organization_id'], ['organizations.id'], name=op.f('fk_comments_organization_id_organizations'), ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['parent_id'], ['comments.id'], name=op.f('fk_comments_parent_id_comments'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_comments'))
    )
    op.create_index('ix_comments_resource_type_resource_id_created_at', 'comments', ['resource_type', 'resource_id', 'created_at'], unique=False)

    op.create_table('approvals',
    sa.Column('id', sa.String(length=64), nullable=False),
    sa.Column('resource_type', postgresql.ENUM('stack', 'run', 'project', name='team_resource_type', create_type=False), nullable=False),
    sa.Column('resource_id', sa.String(length=64), nullable=False),
    sa.Column('organization_id', sa.String(length=64), nullable=False),
    sa.Column('requested_by_user_id', sa.String(length=64), nullable=True),
    sa.Column('decided_by_user_id', sa.String(length=64), nullable=True),
    sa.Column('status', sa.Enum('pending', 'approved', 'rejected', name='approval_status'), server_default='pending', nullable=False),
    sa.Column('decision_note', sa.Text(), nullable=True),
    sa.Column('decided_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['decided_by_user_id'], ['users.id'], name=op.f('fk_approvals_decided_by_user_id_users'), ondelete='SET NULL'),
    sa.ForeignKeyConstraint(['organization_id'], ['organizations.id'], name=op.f('fk_approvals_organization_id_organizations'), ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['requested_by_user_id'], ['users.id'], name=op.f('fk_approvals_requested_by_user_id_users'), ondelete='SET NULL'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_approvals'))
    )
    op.create_index('ix_approvals_organization_id_status', 'approvals', ['organization_id', 'status'], unique=False)
    # One pending approval per resource; decided rows are history.
    op.create_index('uq_approvals_resource_pending', 'approvals', ['resource_type', 'resource_id'], unique=True, postgresql_where=sa.text("status = 'pending'"))

    # --- Team visibility on existing work ------------------------------------
    # `visibility` is a new type used by two ALTERed tables, so it is created
    # explicitly here rather than by a create_table.
    sa.Enum('private', 'team', 'public', name='visibility').create(op.get_bind())
    visibility = postgresql.ENUM('private', 'team', 'public', name='visibility', create_type=False)

    op.add_column('stacks', sa.Column('organization_id', sa.String(length=64), nullable=True))
    op.add_column('stacks', sa.Column('visibility', visibility, server_default='private', nullable=False))
    op.create_foreign_key(op.f('fk_stacks_organization_id_organizations'), 'stacks', 'organizations', ['organization_id'], ['id'], ondelete='SET NULL')
    op.create_index('ix_stacks_organization_id', 'stacks', ['organization_id'], unique=False, postgresql_where=sa.text('organization_id IS NOT NULL'))

    op.add_column('projects', sa.Column('organization_id', sa.String(length=64), nullable=True))
    op.add_column('projects', sa.Column('visibility', visibility, server_default='private', nullable=False))
    op.create_foreign_key(op.f('fk_projects_organization_id_organizations'), 'projects', 'organizations', ['organization_id'], ['id'], ondelete='SET NULL')
    op.create_index('ix_projects_organization_id', 'projects', ['organization_id'], unique=False, postgresql_where=sa.text('organization_id IS NOT NULL'))

    # --- The FK M20 deliberately deferred ------------------------------------
    op.create_foreign_key(op.f('fk_subscriptions_organization_id_organizations'), 'subscriptions', 'organizations', ['organization_id'], ['id'], ondelete='CASCADE')
    # One live subscription per organization — the mirror of the user index.
    op.create_index('uq_subscriptions_organization_live', 'subscriptions', ['organization_id'], unique=True, postgresql_where=sa.text("organization_id IS NOT NULL AND status <> 'canceled'"))

    # Autogenerate also proposed dropping `ix_projects_search` and
    # `ix_templates_search`. Both are raw-SQL expression indexes it cannot
    # reflect into a model, so it reads them as orphans on every revision.
    # Those drops are deleted here, and will need deleting again next time.


def downgrade() -> None:
    op.drop_index('uq_subscriptions_organization_live', table_name='subscriptions', postgresql_where=sa.text("organization_id IS NOT NULL AND status <> 'canceled'"))
    op.drop_constraint(op.f('fk_subscriptions_organization_id_organizations'), 'subscriptions', type_='foreignkey')

    op.drop_index('ix_projects_organization_id', table_name='projects', postgresql_where=sa.text('organization_id IS NOT NULL'))
    op.drop_constraint(op.f('fk_projects_organization_id_organizations'), 'projects', type_='foreignkey')
    op.drop_column('projects', 'visibility')
    op.drop_column('projects', 'organization_id')

    op.drop_index('ix_stacks_organization_id', table_name='stacks', postgresql_where=sa.text('organization_id IS NOT NULL'))
    op.drop_constraint(op.f('fk_stacks_organization_id_organizations'), 'stacks', type_='foreignkey')
    op.drop_column('stacks', 'visibility')
    op.drop_column('stacks', 'organization_id')

    op.drop_index('uq_approvals_resource_pending', table_name='approvals', postgresql_where=sa.text("status = 'pending'"))
    op.drop_index('ix_approvals_organization_id_status', table_name='approvals')
    op.drop_table('approvals')
    op.drop_index('ix_comments_resource_type_resource_id_created_at', table_name='comments')
    op.drop_table('comments')
    op.drop_index('uq_invitations_organization_id_email_open', table_name='invitations', postgresql_where=sa.text('accepted_at IS NULL AND revoked_at IS NULL'))
    op.drop_index('ix_invitations_organization_id', table_name='invitations')
    op.drop_table('invitations')
    op.drop_index('uq_organization_members_one_owner', table_name='organization_members', postgresql_where=sa.text("role = 'owner'"))
    op.drop_index('ix_organization_members_user_id', table_name='organization_members')
    op.drop_table('organization_members')
    op.drop_index('ix_organizations_active', table_name='organizations', postgresql_where=sa.text('deleted_at IS NULL'))
    op.drop_table('organizations')

    # Dropping tables leaves the enum types behind, and the next upgrade then
    # fails on a type that already exists. `plan` is not dropped — it belongs
    # to `users` and predates this revision.
    op.execute("DROP TYPE IF EXISTS visibility")
    op.execute("DROP TYPE IF EXISTS approval_status")
    op.execute("DROP TYPE IF EXISTS team_resource_type")
    op.execute("DROP TYPE IF EXISTS org_role")
