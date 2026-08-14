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

    # A paid plan chosen at registration and not yet paid for. Null is the
    # normal state; a value here means the account owes a checkout and is what
    # the payment wall reads.
    #
    # Deliberately separate from `plan`, which is what the user *has*. Writing
    # the chosen plan straight onto `plan` would hand out Pro to anyone who
    # picked it on the signup form and closed the tab, and there is no later
    # event that would take it back — Stripe never sends a webhook for a
    # checkout that did not happen.
    pending_plan: Mapped[Plan | None] = mapped_column(
        Enum(Plan, name="plan", values_callable=lambda e: [m.value for m in e], create_type=False),
    )
    #: "monthly" | "annual" — which price the wall should open checkout on.
    pending_interval: Mapped[str | None] = mapped_column(String(10))

    locked_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    failed_login_count: Mapped[int] = mapped_column(
        BigInteger, default=0, server_default=text("0"), nullable=False
    )
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        Index("ix_users_plan", "plan"),
        Index("ix_users_active", "deleted_at", postgresql_where=text("deleted_at IS NULL")),
        Index(
            "ix_users_pending_plan",
            "pending_plan",
            postgresql_where=text("pending_plan IS NOT NULL"),
        ),
    )

    @property
    def owes_checkout(self) -> bool:
        """A paid plan was chosen and has not been paid for.

        Read by the payment wall. Not a security boundary — `plan` is what
        every feature check reads, and it stays Free until a webhook says
        otherwise, so a user who dodges the wall gets the free tier rather
        than the plan they picked.
        """
        return self.pending_plan is not None

    @property
    def is_verified(self) -> bool:
        return self.email_verified_at is not None

    @property
    def is_active(self) -> bool:
        return self.deleted_at is None

    def __repr__(self) -> str:  # pragma: no cover
        return f"<User {self.id} {self.email}>"
