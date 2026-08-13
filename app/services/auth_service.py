"""Registration, login, verification, reset, and the audit trail.

Two properties are load-bearing throughout and easy to lose in a refactor:

  * **No enumeration.** `/register`, `/login`, and `/forgot-password` return the
    same shape, status, and — as far as is practical — timing, whether or not
    the address exists.
  * **No exception leaks which half failed.** Login always raises
    `InvalidCredentials`, never "no such user".
"""

from __future__ import annotations

import asyncio
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.context import get_context
from app.core.database import new_id, utcnow
from app.core.errors import AccountLocked, Conflict, InvalidCredentials, TokenInvalid
from app.core.logging import get_logger
from app.integrations import email as email_integration
from app.models.auth import (
    AnonymousSession,
    AuthEvent,
    AuthEventType,
    AuthOutcome,
    AuthToken,
    TokenPurpose,
)
from app.models.user import User
from app.services import email_templates, password_service, session_service, token_service

logger = get_logger("auth")


@dataclass(frozen=True)
class RequestMeta:
    ip: str | None
    user_agent: str | None


# ── Audit ───────────────────────────────────────────────────────────────────


async def record_event(
    db: AsyncSession,
    *,
    event: AuthEventType,
    outcome: AuthOutcome = AuthOutcome.SUCCESS,
    user_id: str | None = None,
    anonymous_session_id: str | None = None,
    meta: RequestMeta | None = None,
    metadata: dict[str, str] | None = None,
) -> None:
    db.add(
        AuthEvent(
            id=new_id("aev"),
            user_id=user_id,
            anonymous_session_id=anonymous_session_id,
            event=event,
            outcome=outcome,
            ip=meta.ip if meta else None,
            user_agent=meta.user_agent if meta else None,
            request_id=get_context().request_id or None,
            event_metadata=metadata,
            created_at=utcnow(),
        )
    )


# ── Lookup ──────────────────────────────────────────────────────────────────


async def get_user_by_email(db: AsyncSession, email: str) -> User | None:
    return (
        await db.execute(select(User).where(User.email == email, User.deleted_at.is_(None)))
    ).scalar_one_or_none()


async def get_user(db: AsyncSession, user_id: str) -> User | None:
    return (
        await db.execute(select(User).where(User.id == user_id, User.deleted_at.is_(None)))
    ).scalar_one_or_none()


# ── One-time tokens ─────────────────────────────────────────────────────────


async def issue_auth_token(
    db: AsyncSession,
    *,
    user_id: str,
    purpose: TokenPurpose,
    ttl: timedelta,
    payload: dict[str, str] | None = None,
) -> str:
    """Returns the plaintext, stores only the hash.

    Issuing invalidates any outstanding token of the same purpose, so a
    forwarded older email cannot be used after a newer request.
    """
    await db.execute(
        delete(AuthToken).where(
            AuthToken.user_id == user_id,
            AuthToken.purpose == purpose,
            AuthToken.consumed_at.is_(None),
        )
    )

    plaintext = token_service.generate_secret()
    db.add(
        AuthToken(
            id=new_id("atk"),
            user_id=user_id,
            purpose=purpose,
            token_hash=token_service.hash_secret(plaintext),
            payload=payload,
            expires_at=utcnow() + ttl,
        )
    )
    await db.flush()
    return plaintext


async def consume_auth_token(db: AsyncSession, *, token: str, purpose: TokenPurpose) -> AuthToken:
    digest = token_service.hash_secret(token)
    record = (
        await db.execute(
            select(AuthToken).where(AuthToken.token_hash == digest, AuthToken.purpose == purpose)
        )
    ).scalar_one_or_none()

    if record is None or record.consumed_at is not None or record.expires_at <= utcnow():
        raise TokenInvalid("This link is invalid or has expired.")

    record.consumed_at = utcnow()
    return record


# ── Registration ────────────────────────────────────────────────────────────


async def register(
    db: AsyncSession,
    *,
    email: str,
    password: str,
    name: str,
    meta: RequestMeta,
    email_verified: bool = False,
) -> User | None:
    """Returns the new user, or None when the address already exists.

    The caller responds identically either way. The existing account holder is
    emailed instead — the only pattern that prevents enumeration without
    breaking the signup experience for everyone else.
    """
    await password_service.validate_password(password, email=email)

    existing = await get_user_by_email(db, email)
    if existing is not None:
        await email_integration.send(email_templates.registration_attempt(to=email))
        await record_event(
            db,
            event=AuthEventType.REGISTER,
            outcome=AuthOutcome.FAILURE,
            user_id=existing.id,
            meta=meta,
            metadata={"reason": "duplicate_email"},
        )
        return None

    user = User(
        id=new_id("usr"),
        email=email,
        name=name.strip(),
        password_hash=password_service.hash_password(password),
        password_changed_at=utcnow(),
    )
    if email_verified:
        # Signup-from-invite (M21): possession of the invite link proves
        # control of the inbox it was sent to, so a second round trip through
        # a verification email would verify nothing new.
        user.email_verified_at = utcnow()
    db.add(user)
    await db.flush()

    if not email_verified:
        token = await issue_auth_token(
            db,
            user_id=user.id,
            purpose=TokenPurpose.EMAIL_VERIFY,
            ttl=timedelta(hours=settings.email_verify_ttl_hours),
        )
        await email_integration.send(
            email_templates.verify_email(to=user.email, name=user.name, token=token)
        )
    await record_event(db, event=AuthEventType.REGISTER, user_id=user.id, meta=meta)
    return user


# ── Login ───────────────────────────────────────────────────────────────────

# Verifying a hash that does not exist still has to take as long as verifying
# one that does, or response timing answers "is this address registered?".
_DUMMY_HASH = password_service.hash_password(secrets.token_urlsafe(24))


async def authenticate(db: AsyncSession, *, email: str, password: str, meta: RequestMeta) -> User:
    user = await get_user_by_email(db, email)
    now = utcnow()

    if user is not None and user.locked_until and user.locked_until > now:
        await record_event(
            db,
            event=AuthEventType.LOGIN_FAILED,
            outcome=AuthOutcome.FAILURE,
            user_id=user.id,
            meta=meta,
            metadata={"reason": "locked"},
        )
        raise AccountLocked(details={"locked_until": user.locked_until.isoformat()})

    stored_hash = user.password_hash if user and user.password_hash else _DUMMY_HASH
    correct = password_service.verify_password(stored_hash, password)

    if user is None or user.password_hash is None or not correct:
        if user is not None:
            await _register_failure(db, user, meta)
        raise InvalidCredentials()

    if password_service.needs_rehash(stored_hash):
        user.password_hash = password_service.hash_password(password)

    user.failed_login_count = 0
    user.locked_until = None
    user.last_login_at = now

    await record_event(db, event=AuthEventType.LOGIN, user_id=user.id, meta=meta)
    return user


async def _register_failure(db: AsyncSession, user: User, meta: RequestMeta) -> None:
    user.failed_login_count += 1
    metadata = {"attempts": str(user.failed_login_count)}

    if user.failed_login_count >= settings.max_failed_logins:
        # Exponential from the base window, capped at an hour. A permanent lock
        # would turn a stuffing attack into a denial of service against the
        # real account holder.
        overshoot = user.failed_login_count - settings.max_failed_logins
        seconds = min(settings.lockout_seconds * (2**overshoot), 3600)
        user.locked_until = utcnow() + timedelta(seconds=seconds)
        metadata["locked_for"] = str(seconds)
        await record_event(
            db,
            event=AuthEventType.ACCOUNT_LOCKED,
            user_id=user.id,
            meta=meta,
            metadata=metadata,
        )

    await record_event(
        db,
        event=AuthEventType.LOGIN_FAILED,
        outcome=AuthOutcome.FAILURE,
        user_id=user.id,
        meta=meta,
        metadata=metadata,
    )

    # Commit before the caller raises InvalidCredentials. get_session rolls
    # back on exception, which would otherwise discard the attempt counter and
    # the lockout — making brute-force protection silently inert.
    await db.commit()


# ── Verification ────────────────────────────────────────────────────────────


async def verify_email(db: AsyncSession, *, token: str, meta: RequestMeta) -> User:
    record = await consume_auth_token(db, token=token, purpose=TokenPurpose.EMAIL_VERIFY)
    user = await get_user(db, record.user_id)
    if user is None:
        raise TokenInvalid("This link is invalid or has expired.")

    if user.email_verified_at is None:
        user.email_verified_at = utcnow()
        await record_event(db, event=AuthEventType.EMAIL_VERIFIED, user_id=user.id, meta=meta)
    return user


async def resend_verification(db: AsyncSession, *, user: User) -> None:
    if user.email_verified_at is not None:
        return
    token = await issue_auth_token(
        db,
        user_id=user.id,
        purpose=TokenPurpose.EMAIL_VERIFY,
        ttl=timedelta(hours=settings.email_verify_ttl_hours),
    )
    await email_integration.send(
        email_templates.verify_email(to=user.email, name=user.name, token=token)
    )


# ── Password reset ──────────────────────────────────────────────────────────


async def request_password_reset(db: AsyncSession, *, email: str, meta: RequestMeta) -> None:
    """Always succeeds from the caller's perspective."""
    user = await get_user_by_email(db, email)
    if user is None:
        # Spend comparable time so the response does not time-leak existence.
        await asyncio.sleep(0.05)
        return

    if user.password_hash is None:
        # OAuth-only account. Telling them which provider is safe: they already
        # proved ownership of the address to that provider.
        await asyncio.sleep(0.05)
        return

    token = await issue_auth_token(
        db,
        user_id=user.id,
        purpose=TokenPurpose.PASSWORD_RESET,
        ttl=timedelta(minutes=settings.password_reset_ttl_minutes),
    )
    await email_integration.send(
        email_templates.password_reset(to=user.email, name=user.name, token=token)
    )
    await record_event(db, event=AuthEventType.PASSWORD_RESET_REQUESTED, user_id=user.id, meta=meta)


async def reset_password(
    db: AsyncSession, *, token: str, new_password: str, meta: RequestMeta
) -> User:
    record = await consume_auth_token(db, token=token, purpose=TokenPurpose.PASSWORD_RESET)
    user = await get_user(db, record.user_id)
    if user is None:
        raise TokenInvalid("This link is invalid or has expired.")

    await password_service.validate_password(new_password, email=user.email)

    user.password_hash = password_service.hash_password(new_password)
    user.password_changed_at = utcnow()
    user.must_set_password = False
    user.failed_login_count = 0
    user.locked_until = None

    # A reset is the remedy for a compromised account, so it must end every
    # session — including the current one.
    await session_service.revoke_all_for_user(
        db, user_id=user.id, reason=session_service.RevokeReason.PASSWORD_RESET
    )

    await email_integration.send(email_templates.password_changed(to=user.email, name=user.name))
    await record_event(db, event=AuthEventType.PASSWORD_RESET, user_id=user.id, meta=meta)
    return user


async def change_password(
    db: AsyncSession,
    *,
    user: User,
    current_password: str,
    new_password: str,
    keep_session_id: str | None,
    meta: RequestMeta,
) -> None:
    if user.password_hash is None or not password_service.verify_password(
        user.password_hash, current_password
    ):
        raise InvalidCredentials("Your current password is incorrect.")

    await password_service.validate_password(new_password, email=user.email)

    user.password_hash = password_service.hash_password(new_password)
    user.password_changed_at = utcnow()

    # Every session but this one. Changing a password while signed in should
    # not sign you out of the browser you are using.
    await session_service.revoke_all_for_user(
        db,
        user_id=user.id,
        reason=session_service.RevokeReason.PASSWORD_CHANGE,
        except_session_id=keep_session_id,
    )

    await email_integration.send(email_templates.password_changed(to=user.email, name=user.name))
    await record_event(db, event=AuthEventType.PASSWORD_CHANGED, user_id=user.id, meta=meta)


# ── Anonymous identity ──────────────────────────────────────────────────────


async def create_anonymous_session(db: AsyncSession, *, meta: RequestMeta) -> AnonymousSession:
    record = AnonymousSession(
        id=new_id("anon"),
        ip=meta.ip,
        user_agent=meta.user_agent,
        last_seen_at=utcnow(),
    )
    db.add(record)
    await db.flush()
    return record


async def get_anonymous_session(db: AsyncSession, anon_id: str) -> AnonymousSession | None:
    return (
        await db.execute(select(AnonymousSession).where(AnonymousSession.id == anon_id))
    ).scalar_one_or_none()


async def claim_anonymous_session(
    db: AsyncSession, *, anon_id: str, user: User, meta: RequestMeta
) -> int:
    """Attach anonymous work to a new account.

    Returns the number of records reassigned. Zero today — `tool_runs` arrives
    in M08 — but the hook exists so the claim call site does not have to change
    later, and so the flow is exercised by tests from the start.
    """
    record = await get_anonymous_session(db, anon_id)
    if record is None or record.claimed_by_user_id is not None:
        return 0

    record.claimed_by_user_id = user.id
    record.claimed_at = utcnow()

    reassigned = 0
    await record_event(
        db,
        event=AuthEventType.ANONYMOUS_CLAIMED,
        user_id=user.id,
        anonymous_session_id=anon_id,
        meta=meta,
        metadata={"reassigned": str(reassigned)},
    )
    return reassigned


# ── Account ─────────────────────────────────────────────────────────────────


async def soft_delete_account(db: AsyncSession, *, user: User, meta: RequestMeta) -> None:
    now = utcnow()
    user.deleted_at = now
    # Free the address immediately so the user can register again, while
    # keeping the row for the 30-day purge window.
    user.email = f"deleted+{user.id}@invalid.local"
    await session_service.revoke_all_for_user(
        db, user_id=user.id, reason=session_service.RevokeReason.USER
    )
    await record_event(db, event=AuthEventType.ACCOUNT_DELETED, user_id=user.id, meta=meta)


async def ensure_unique_email(db: AsyncSession, email: str) -> None:
    if await get_user_by_email(db, email) is not None:
        raise Conflict("That email address is already in use.")


def utc_now() -> datetime:
    return datetime.now(UTC)
