"""The single log of every calculation.

One table for all 28 tools. The inputs and outputs differ per tool so they are
JSONB; everything anyone queries or aggregates on is a real column.

This is the North Star metric's data source, the dashboard's recent-activity
feed, and the input to quota accounting — three consumers that would otherwise
each need their own instrumentation, each with its own way of being wrong.
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
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base, new_id


class RunSource(str, enum.Enum):
    RULE_BASED = "rule_based"
    AI_SYNTHESIS = "ai_synthesis"
    HYBRID = "hybrid"


class ToolRun(Base):
    __tablename__ = "tool_runs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("run"))
    tool_slug: Mapped[str] = mapped_column(String(80), nullable=False)
    workflow: Mapped[str] = mapped_column(String(40), nullable=False)

    user_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    organization_id: Mapped[str | None] = mapped_column(String(64))
    project_id: Mapped[str | None] = mapped_column(String(64))

    input: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    output: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)

    source: Mapped[RunSource] = mapped_column(
        Enum(RunSource, name="run_source", values_callable=lambda e: [m.value for m in e]),
        nullable=False,
        default=RunSource.RULE_BASED,
    )
    ai_call_id: Mapped[str | None] = mapped_column(String(64))
    duration_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    # Unsaved rows purge after 30 days. Keeping every calculation forever
    # would make this the largest table in the database within a year, for data
    # nobody asked to keep.
    saved: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    __table_args__ = (
        # `user_id` used to be one of two nullable owner columns held to
        # exactly one by a check constraint. With the anonymous tier gone it is
        # simply NOT NULL, which is the same guarantee with none of the
        # machinery.
        Index("ix_tool_runs_user_id_created_at", "user_id", text("created_at DESC")),
        Index("ix_tool_runs_tool_slug_created_at", "tool_slug", text("created_at DESC")),
        Index(
            "ix_tool_runs_created_at_unsaved",
            "created_at",
            postgresql_where=text("saved IS false"),
        ),
    )


__all__ = ["RunSource", "ToolRun"]
