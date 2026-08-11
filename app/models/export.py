"""Exports and share links (M18).

Two tables that look similar and are not. An **export** is a rendered byte
string with an expiry — a thing the user asked us to make once. A **share
link** is a capability URL with a revoke switch — a thing the user handed to
someone else and may want back.

That difference decides how the tokens are stored. `share_links.token` is
plaintext, because the owner has to be able to re-read and re-copy it from the
UI; its protections are revocability, optional expiry, `noindex`, and 404 on a
dead token. Auth credentials are hashed for the opposite reason — they are
shown once and never re-read (D-14).

**Export bytes live in this table, not in object storage.** A stack-plan zip
is tens of kilobytes and there is no bucket configured anywhere in this
program. Putting the bytes in Postgres makes "expired export objects are
deleted from storage" a `DELETE` inside the same transaction that drops the
row, rather than a two-system reconciliation whose failure mode is paying for
orphaned objects nobody can enumerate. If exports ever grow to the size where
that is wrong, `content` becomes a key and this comment becomes the reason to
change it.
"""

from __future__ import annotations

import enum
from datetime import datetime

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base, TimestampMixin, new_id


class ExportFormat(str, enum.Enum):
    MARKDOWN = "markdown"
    JSON = "json"
    YAML = "yaml"
    CSV = "csv"
    PDF = "pdf"
    ZIP = "zip"


class ExportStatus(str, enum.Enum):
    PENDING = "pending"
    READY = "ready"
    FAILED = "failed"


class SourceType(str, enum.Enum):
    """What an export or a share points at.

    Only two. `PRD.md` §8.6 names artifacts and comparisons as shareable
    things too, but an artifact is one named block *inside* a run or a stack
    and a comparison is a run of a compare tool — neither is a row of its own.
    Modelling them as `(source_type, source_id, artifact_type)` covers all
    four nouns without inventing two tables that would each have one column
    and no lifecycle.
    """

    RUN = "run"
    STACK = "stack"


class Export(Base):
    __tablename__ = "exports"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("exp"))

    # Anonymous callers export too — Markdown is free and reaches every tool,
    # and an export that required an account would put the paywall in front of
    # the answer rather than in front of keeping it.
    user_id: Mapped[str | None] = mapped_column(
        String(64), ForeignKey("users.id", ondelete="CASCADE")
    )
    anonymous_session_id: Mapped[str | None] = mapped_column(
        String(64), ForeignKey("anonymous_sessions.id", ondelete="SET NULL")
    )

    source_type: Mapped[SourceType] = mapped_column(
        Enum(SourceType, name="export_source_type", values_callable=lambda e: [m.value for m in e]),
        nullable=False,
    )
    source_id: Mapped[str] = mapped_column(String(64), nullable=False)
    #: Which artifact within the source, or NULL for the whole result.
    artifact_type: Mapped[str | None] = mapped_column(String(60))

    format: Mapped[ExportFormat] = mapped_column(
        Enum(ExportFormat, name="export_format", values_callable=lambda e: [m.value for m in e]),
        nullable=False,
    )
    status: Mapped[ExportStatus] = mapped_column(
        Enum(ExportStatus, name="export_status", values_callable=lambda e: [m.value for m in e]),
        nullable=False,
        default=ExportStatus.PENDING,
    )

    filename: Mapped[str] = mapped_column(String(200), nullable=False)
    content_type: Mapped[str] = mapped_column(String(120), nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    content: Mapped[bytes | None] = mapped_column(LargeBinary)
    error: Mapped[str | None] = mapped_column(Text)

    #: Rendered bytes are a cache, not a record. The run and the stack they
    #: came from are the durable things; regenerating is deterministic and
    #: cheap, so keeping the bytes past a week buys nothing and costs storage.
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        # Same rule as `tool_runs`: exactly one owner, enforced by the
        # database. An export with neither vanishes from every listing while
        # still occupying storage nobody can find.
        CheckConstraint(
            "num_nonnulls(user_id, anonymous_session_id) = 1",
            name="exactly_one_owner",
        ),
        Index("ix_exports_user_id_created_at", "user_id", text("created_at DESC")),
        Index("ix_exports_expires_at", "expires_at"),
        Index(
            "ix_exports_status_pending",
            "status",
            postgresql_where=text("status = 'pending'"),
        ),
    )


class ShareLink(Base, TimestampMixin):
    """A capability URL over something the owner made.

    Requires an account. An anonymous visitor can run every tool and export
    Markdown, but a link they cannot later revoke is a link they cannot take
    back — and the revoke switch is the only protection this design has.
    """

    __tablename__ = "share_links"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("shr"))
    user_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )

    #: 256 bits, URL-safe, stored as issued (D-14).
    token: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)

    target_type: Mapped[SourceType] = mapped_column(
        Enum(
            SourceType,
            name="export_source_type",
            values_callable=lambda e: [m.value for m in e],
            # The type is created by the `exports` table above; declaring it
            # again here would make the migration emit a duplicate CREATE TYPE.
            create_type=False,
        ),
        nullable=False,
    )
    target_id: Mapped[str] = mapped_column(String(64), nullable=False)
    artifact_type: Mapped[str | None] = mapped_column(String(60))

    #: Snapshotted at mint time. The public page shows this rather than
    #: re-deriving a title from the target, so renaming a stack cannot silently
    #: change what a link someone already holds claims to be.
    title: Mapped[str] = mapped_column(String(200), nullable=False)

    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    view_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_viewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        Index("ix_share_links_user_id_created_at", "user_id", text("created_at DESC")),
        Index("ix_share_links_target", "target_type", "target_id"),
    )

    @property
    def is_live(self) -> bool:
        """Never `is_expired` — the caller must ask the positive question.

        A negative predicate invites `if not link.is_expired and not
        link.revoked_at`, and the second clause is the one that gets forgotten.
        """
        from app.core.database import utcnow

        if self.revoked_at is not None:
            return False
        return self.expires_at is None or self.expires_at > utcnow()


__all__ = ["Export", "ExportFormat", "ExportStatus", "ShareLink", "SourceType"]
