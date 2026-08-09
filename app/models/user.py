from __future__ import annotations

import enum
from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Enum,
    Index,
    String,
    Text,
    text,
)
from sqlalchemy.dialects.postgresql import CITEXT
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base, SoftDeleteMixin, TimestampMixin, new_id


class UserRole(str, enum.Enum):
    """Platform role. Independent of organization role — conflating the two is
    how 'the org owner cannot be a platform admin' bugs happen."""

    USER = "user"
    ADMIN = "admin"


class Plan(str, enum.Enum):
    FREE = "free"
    PRO = "pro"
    TEAM = "team"
    ENTERPRISE = "enterprise"


class PlanSource(str, enum.Enum):
    PERSONAL = "personal"
    ORGANIZATION = "organization"


class User(Base, TimestampMixin, SoftDeleteMixin):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("usr"))

    # citext so case never creates a duplicate account.
    email: Mapped[str] = mapped_column(CITEXT, unique=True, nullable=False)
    email_verified_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # Null for OAuth-only accounts. Those users cannot use the password reset
    # flow into a password login until they set one.
    password_hash: Mapped[str | None] = mapped_column(Text)
    password_algo: Mapped[str] = mapped_column(String(32), default="argon2id", nullable=False)
    must_set_password: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    password_changed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    name: Mapped[str] = mapped_column(String(120), nullable=False)
    avatar_url: Mapped[str | None] = mapped_column(Text)
    timezone: Mapped[str] = mapped_column(String(64), default="UTC", nullable=False)

    role: Mapped[UserRole] = mapped_column(
        Enum(UserRole, name="user_role", values_callable=lambda e: [m.value for m in e]),
        default=UserRole.USER,
        server_default=UserRole.USER.value,
        nullable=False,
    )

    # Denormalised from `subscriptions` for cheap gating on the hot path. The
    # JWT claim mirrors this but is never authoritative — a 15-minute-stale
    # claim would let a downgraded user keep paid features.
    plan: Mapped[Plan] = mapped_column(
        Enum(Plan, name="plan", values_callable=lambda e: [m.value for m in e]),
        default=Plan.FREE,
        server_default=Plan.FREE.value,
        nullable=False,
    )
    plan_source: Mapped[PlanSource] = mapped_column(
        Enum(PlanSource, name="plan_source", values_callable=lambda e: [m.value for m in e]),
        default=PlanSource.PERSONAL,
        server_default=PlanSource.PERSONAL.value,
        nullable=False,
    )

    locked_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    failed_login_count: Mapped[int] = mapped_column(
        BigInteger, default=0, server_default=text("0"), nullable=False
    )
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        Index("ix_users_plan", "plan"),
        Index("ix_users_active", "deleted_at", postgresql_where=text("deleted_at IS NULL")),
    )

    @property
    def is_verified(self) -> bool:
        return self.email_verified_at is not None

    @property
    def is_active(self) -> bool:
        return self.deleted_at is None

    def __repr__(self) -> str:  # pragma: no cover
        return f"<User {self.id} {self.email}>"
