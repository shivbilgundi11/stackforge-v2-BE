"""Projects, and the heterogeneous things they hold.

A project is a container with a name and a use case — "Client X RAG rollout",
"Internal support bot". What makes it a workspace rather than a folder is
`session_state`: the values a user has chosen to carry between tools, held on
the server so the handoff survives a new device and a new browser.

`project_items` is a **polymorphic join** on purpose. A foreign key per item
type on the project would mean a migration every time a new savable thing
exists, and there will be more of them — exports, templates, shared links. The
cost is that the database cannot enforce the reference, so the service layer
resolves items by type and drops the ones that no longer exist rather than
returning a dangling id.
"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Any

from sqlalchemy import (
    Boolean,
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
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base, TimestampMixin, new_id
from app.models.organization import Visibility


class ProjectItemType(str, enum.Enum):
    RUN = "run"
    STACK = "stack"
    ARTIFACT = "artifact"
    TEMPLATE = "template"


class Project(Base, TimestampMixin):
    __tablename__ = "projects"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("prj"))
    user_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    #: SET NULL on org deletion — the project reverts to private personal work
    #: rather than vanishing with the organization (M21).
    organization_id: Mapped[str | None] = mapped_column(
        String(64), ForeignKey("organizations.id", ondelete="SET NULL")
    )
    visibility: Mapped[Visibility] = mapped_column(
        Enum(
            Visibility,
            name="visibility",
            values_callable=lambda e: [m.value for m in e],
            create_type=False,
        ),
        default=Visibility.PRIVATE,
        server_default=Visibility.PRIVATE.value,
        nullable=False,
    )

    name: Mapped[str] = mapped_column(String(160), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    use_case: Mapped[str | None] = mapped_column(String(60))

    #: The cross-workflow session (`PRD.md` §8.7) — the values a user carries
    #: between tools. Server-side so it survives a device change; the Zustand
    #: store mirrors it.
    session_state: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)

    #: Archived rather than deleted by default. A project is the container a
    #: user's work hangs off, and an accidental delete takes the organisation
    #: of it with them even though the runs survive.
    archived_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # The Postgres FTS index over `name` and `description` is created in the
    # migration rather than declared here. `to_tsvector('english', …)` needs a
    # `regconfig` literal, which SQLAlchemy cannot render into an index
    # definition — autogenerate raises on it. Declaring it in SQL once is
    # better than an index the model claims to have and no migration creates.
    __table_args__ = (
        Index("ix_projects_user_id_updated_at", "user_id", "updated_at"),
        Index(
            "ix_projects_organization_id",
            "organization_id",
            postgresql_where=text("organization_id IS NOT NULL"),
        ),
    )


class ProjectItem(Base):
    """One thing inside a project.

    `item_id` is not a foreign key — it points at a row in whichever table
    `item_type` names. That is the cost of the polymorphic shape, and it is
    paid deliberately (see the module docstring).
    """

    __tablename__ = "project_items"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("pri"))
    project_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False
    )

    item_type: Mapped[ProjectItemType] = mapped_column(
        Enum(
            ProjectItemType,
            name="project_item_type",
            values_callable=lambda e: [m.value for m in e],
        ),
        nullable=False,
    )
    item_id: Mapped[str] = mapped_column(String(64), nullable=False)

    #: Explicit ordering. A list a user has arranged is a list they expect to
    #: stay arranged, and "created_at desc" is not an arrangement.
    position: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    pinned: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    note: Mapped[str | None] = mapped_column(Text)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    __table_args__ = (
        UniqueConstraint(
            "project_id", "item_type", "item_id", name="uq_project_items_project_item"
        ),
        Index("ix_project_items_project_id_position", "project_id", "position"),
    )
