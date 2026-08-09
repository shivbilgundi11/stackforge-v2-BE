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
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import CITEXT, INET, JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base, TimestampMixin, new_id


class Session(Base, TimestampMixin):
    """A refresh token.

    Stored as a SHA-256 hash — the plaintext exists only in the cookie, so a
    database dump yields nothing usable.

    `family_id` is the rotation lineage. Presenting a token that is already
    marked used means two parties hold tokens from the same family: the real
    client and a thief. The server cannot tell which, so it revokes the whole
    family. Without that, rotation is just churn.
    """

    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("ses"))
    user_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    family_id: Mapped[str] = mapped_column(String(64), nullable=False)
    token_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)

    used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    revoked_reason: Mapped[str | None] = mapped_column(String(32))

    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    absolute_expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    ip: Mapped[str | None] = mapped_column(INET)
    user_agent: Mapped[str | None] = mapped_column(Text)
    device_label: Mapped[str | None] = mapped_column(String(120))
    last_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    __table_args__ = (
        Index("ix_sessions_user_id_revoked_at", "user_id", "revoked_at"),
        Index("ix_sessions_family_id", "family_id"),
        Index("ix_sessions_expires_at", "expires_at"),
    )

    def is_usable(self, now: datetime) -> bool:
        return (
            self.revoked_at is None
            and self.used_at is None
            and self.expires_at > now
            and self.absolute_expires_at > now
        )


class TokenPurpose(str, enum.Enum):
    EMAIL_VERIFY = "email_verify"
    PASSWORD_RESET = "password_reset"
    EMAIL_CHANGE = "email_change"


class AuthToken(Base, TimestampMixin):
    """Single-use, hashed, short-lived. Issuing a new token of a purpose
    invalidates the previous one for that user."""

    __tablename__ = "auth_tokens"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("atk"))
    user_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    purpose: Mapped[TokenPurpose] = mapped_column(
        Enum(TokenPurpose, name="token_purpose", values_callable=lambda e: [m.value for m in e]),
        nullable=False,
    )
    token_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    payload: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    consumed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (Index("ix_auth_tokens_user_id_purpose", "user_id", "purpose"),)


class OAuthProvider(str, enum.Enum):
    GOOGLE = "google"
    GITHUB = "github"


class OAuthAccount(Base, TimestampMixin):
    __tablename__ = "oauth_accounts"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("oau"))
    user_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    provider: Mapped[OAuthProvider] = mapped_column(
        Enum(OAuthProvider, name="oauth_provider", values_callable=lambda e: [m.value for m in e]),
        nullable=False,
    )
    provider_account_id: Mapped[str] = mapped_column(String(255), nullable=False)
    provider_email: Mapped[str | None] = mapped_column(CITEXT)
    # Load-bearing: auto-linking on a provider-unverified email is an
    # account-takeover primitive, so the flag is stored, not assumed.
    provider_email_verified: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        UniqueConstraint("provider", "provider_account_id", name="uq_oauth_provider_account"),
        Index("ix_oauth_accounts_user_id", "user_id"),
    )


class AnonymousSession(Base, TimestampMixin):
    """Anonymous identity.

    The product allows 5 tool runs a day without an account, and the funnel
    depends on it. That needs an identity: IP-keyed quota breaks behind
    corporate NAT and shared Wi-Fi, which is where much of this audience works.

    Claimed on signup — turning "I already did the work here" into a reason to
    create an account is the highest-intent conversion moment the product has.
    """

    __tablename__ = "anonymous_sessions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("anon"))
    ip: Mapped[str | None] = mapped_column(INET)
    user_agent: Mapped[str | None] = mapped_column(Text)
    last_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    claimed_by_user_id: Mapped[str | None] = mapped_column(
        String(64), ForeignKey("users.id", ondelete="SET NULL")
    )
    claimed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (Index("ix_anonymous_sessions_claimed_by_user_id", "claimed_by_user_id"),)


class AuthEventType(str, enum.Enum):
    REGISTER = "register"
    LOGIN = "login"
    LOGIN_FAILED = "login_failed"
    LOGOUT = "logout"
    LOGOUT_ALL = "logout_all"
    REFRESH = "refresh"
    REFRESH_REUSE_DETECTED = "refresh_reuse_detected"
    PASSWORD_CHANGED = "password_changed"
    PASSWORD_RESET = "password_reset"
    PASSWORD_RESET_REQUESTED = "password_reset_requested"
    EMAIL_VERIFIED = "email_verified"
    EMAIL_CHANGED = "email_changed"
    OAUTH_LINKED = "oauth_linked"
    SESSION_REVOKED = "session_revoked"
    ACCOUNT_LOCKED = "account_locked"
    ACCOUNT_DELETED = "account_deleted"
    ANONYMOUS_CLAIMED = "anonymous_claimed"


class AuthOutcome(str, enum.Enum):
    SUCCESS = "success"
    FAILURE = "failure"


class AuthEvent(Base):
    """The audit trail. What makes "why am I logged out?" and "was I breached?"
    answerable questions. Retained 180 days."""

    __tablename__ = "auth_events"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("aev"))
    user_id: Mapped[str | None] = mapped_column(
        String(64), ForeignKey("users.id", ondelete="SET NULL")
    )
    anonymous_session_id: Mapped[str | None] = mapped_column(String(64))
    event: Mapped[AuthEventType] = mapped_column(
        Enum(AuthEventType, name="auth_event_type", values_callable=lambda e: [m.value for m in e]),
        nullable=False,
    )
    outcome: Mapped[AuthOutcome] = mapped_column(
        Enum(AuthOutcome, name="auth_outcome", values_callable=lambda e: [m.value for m in e]),
        nullable=False,
    )
    ip: Mapped[str | None] = mapped_column(INET)
    user_agent: Mapped[str | None] = mapped_column(Text)
    request_id: Mapped[str | None] = mapped_column(String(64))
    event_metadata: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    __table_args__ = (Index("ix_auth_events_user_id_created_at", "user_id", "created_at"),)
