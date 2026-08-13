"""Saved stacks, and their version history.

Two tables. `stacks` is the current state; `stack_versions` is an append-only
log of every save. The current state is duplicated into a version row on every
write rather than reconstructed by replaying the log, because the question
this data answers is "what did v2 look like" — and replaying to answer it
turns a diff into a migration risk.

**Neither table stores a score.** The score is computed from the catalog at
read time (M15), so a component deprecated tomorrow lowers this stack's score
tomorrow. A stored score is a snapshot of a catalog that has since moved, and
it is exactly what makes a deprecation alert cosmetic instead of real.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base, TimestampMixin, new_id
from app.models.organization import Visibility


class Stack(Base, TimestampMixin):
    __tablename__ = "stacks"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("stk"))
    user_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    #: SET NULL on org deletion — the stack reverts to private personal work
    #: rather than vanishing with the organization (M21).
    organization_id: Mapped[str | None] = mapped_column(
        String(64), ForeignKey("organizations.id", ondelete="SET NULL")
    )
    visibility: Mapped[Visibility] = mapped_column(
        Enum(Visibility, name="visibility", values_callable=lambda e: [m.value for m in e]),
        default=Visibility.PRIVATE,
        server_default=Visibility.PRIVATE.value,
        nullable=False,
    )
    name: Mapped[str] = mapped_column(String(160), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)

    #: What the user asked for. Kept so a saved stack can be re-scored against
    #: the same requirements later, and so the form can be re-opened as it was.
    requirements: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    #: Catalog slugs, not names or ids. Slugs are stable across re-seeds;
    #: row ids are not, and a saved stack that points at nothing after a reseed
    #: is the failure this column shape prevents.
    component_slugs: Mapped[list[str]] = mapped_column(
        ARRAY(String(80)), nullable=False, default=list
    )

    current_version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    source_run_id: Mapped[str | None] = mapped_column(String(64))
    cloned_from_id: Mapped[str | None] = mapped_column(String(64))

    __table_args__ = (
        Index("ix_stacks_user_id_updated_at", "user_id", "updated_at"),
        Index("ix_stacks_component_slugs", "component_slugs", postgresql_using="gin"),
        Index(
            "ix_stacks_organization_id",
            "organization_id",
            postgresql_where=text("organization_id IS NOT NULL"),
        ),
    )


class StackVersion(Base):
    """One save. Append-only — a past version is never edited."""

    __tablename__ = "stack_versions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("stv"))
    stack_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("stacks.id", ondelete="CASCADE"), nullable=False
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False)

    name: Mapped[str] = mapped_column(String(160), nullable=False)
    requirements: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    component_slugs: Mapped[list[str]] = mapped_column(
        ARRAY(String(80)), nullable=False, default=list
    )
    change_summary: Mapped[str | None] = mapped_column(Text)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    __table_args__ = (
        UniqueConstraint("stack_id", "version", name="uq_stack_versions_stack_id_version"),
        Index("ix_stack_versions_stack_id", "stack_id"),
    )
