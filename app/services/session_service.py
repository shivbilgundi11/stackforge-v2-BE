"""Refresh-token sessions: issue, rotate, revoke, and detect reuse."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, cast

from sqlalchemy import select, update
from sqlalchemy.engine import CursorResult
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import new_id, utcnow
from app.core.logging import get_logger
from app.models.auth import Session

logger = get_logger("auth.session")


class RevokeReason:
    LOGOUT = "logout"
    LOGOUT_ALL = "logout_all"
    REUSE = "reuse"
    PASSWORD_CHANGE = "password_change"  # noqa: S105
    PASSWORD_RESET = "password_reset"  # noqa: S105
    ADMIN = "admin"
    USER = "user"


@dataclass(frozen=True)
class IssuedSession:
    session: Session
    refresh_token: str  # plaintext — returned once, only ever sent as a cookie


@dataclass(frozen=True)
class RotationResult:
    session: Session | None
    refresh_token: str | None
    reuse_detected: bool


def _device_label(user_agent: str | None) -> str | None:
    """A short, human-readable label for the session list.

    Deliberately crude. Full UA parsing is a dependency and a fingerprinting
    surface for a string the user reads once.
    """
    if not user_agent:
        return None
    ua = user_agent.lower()
    browser = next(
        (
            name
            for marker, name in (
                ("edg/", "Edge"),
                ("opr/", "Opera"),
                ("chrome/", "Chrome"),
                ("safari/", "Safari"),
                ("firefox/", "Firefox"),
            )
            if marker in ua
        ),
        "Browser",
    )
    platform = next(
        (
            name
            for marker, name in (
                ("windows", "Windows"),
                ("mac os", "macOS"),
                ("iphone", "iPhone"),
                ("ipad", "iPad"),
                ("android", "Android"),
                ("linux", "Linux"),
            )
            if marker in ua
        ),
        "Unknown",
    )
    return f"{browser} on {platform}"


async def issue(
    db: AsyncSession,
    *,
    user_id: str,
    ip: str | None,
    user_agent: str | None,
    family_id: str | None = None,
    now: datetime | None = None,
) -> IssuedSession:
    """Create a session. A new `family_id` starts a fresh rotation lineage;
    passing one continues an existing lineage through a rotation."""
    from app.services.token_service import generate_secret, hash_secret

    moment = now or utcnow()
    token = generate_secret()

    session = Session(
        id=new_id("ses"),
        user_id=user_id,
        family_id=family_id or new_id("fam"),
        token_hash=hash_secret(token),
        expires_at=moment + timedelta(days=settings.refresh_token_ttl_days),
        absolute_expires_at=moment + timedelta(days=settings.refresh_token_absolute_ttl_days),
        ip=ip,
        user_agent=user_agent,
        device_label=_device_label(user_agent),
        last_seen_at=moment,
    )
    db.add(session)
    await db.flush()
    return IssuedSession(session=session, refresh_token=token)


async def rotate(
    db: AsyncSession,
    *,
    presented_token: str,
    ip: str | None,
    user_agent: str | None,
    now: datetime | None = None,
) -> RotationResult:
    """Exchange a refresh token for a new one.

    The reuse branch is the reason rotation is worth its complexity. A token
    already marked `used` means two parties hold tokens from this family — the
    legitimate client and a thief. The server cannot tell them apart, so it
    revokes the entire family: both are logged out, the real user signs in
    again, the thief has nothing.
    """
    from app.services.token_service import hash_secret

    moment = now or utcnow()
    digest = hash_secret(presented_token)

    existing = (
        await db.execute(select(Session).where(Session.token_hash == digest))
    ).scalar_one_or_none()

    if existing is None:
        return RotationResult(session=None, refresh_token=None, reuse_detected=False)

    if existing.used_at is not None:
        await revoke_family(db, family_id=existing.family_id, reason=RevokeReason.REUSE, now=moment)
        logger.warning(
            "auth.refresh_reuse_detected",
            user_id=existing.user_id,
            family_id=existing.family_id,
        )
        return RotationResult(session=existing, refresh_token=None, reuse_detected=True)

    if not existing.is_usable(moment):
        return RotationResult(session=None, refresh_token=None, reuse_detected=False)

    existing.used_at = moment
    existing.last_seen_at = moment

    issued = await issue(
        db,
        user_id=existing.user_id,
        ip=ip,
        user_agent=user_agent,
        family_id=existing.family_id,  # same lineage
        now=moment,
    )
    # The absolute ceiling does not slide. Otherwise a session refreshed every
    # day would live forever, and "log out everywhere" would be the only way
    # to ever end it.
    issued.session.absolute_expires_at = existing.absolute_expires_at

    return RotationResult(
        session=issued.session, refresh_token=issued.refresh_token, reuse_detected=False
    )


async def get_by_token(db: AsyncSession, token: str) -> Session | None:
    from app.services.token_service import hash_secret

    return (
        await db.execute(select(Session).where(Session.token_hash == hash_secret(token)))
    ).scalar_one_or_none()


async def revoke(
    db: AsyncSession, *, session_id: str, reason: str, now: datetime | None = None
) -> None:
    await db.execute(
        update(Session)
        .where(Session.id == session_id, Session.revoked_at.is_(None))
        .values(revoked_at=now or utcnow(), revoked_reason=reason)
    )


async def revoke_family(
    db: AsyncSession, *, family_id: str, reason: str, now: datetime | None = None
) -> None:
    await db.execute(
        update(Session)
        .where(Session.family_id == family_id, Session.revoked_at.is_(None))
        .values(revoked_at=now or utcnow(), revoked_reason=reason)
    )


async def revoke_all_for_user(
    db: AsyncSession,
    *,
    user_id: str,
    reason: str,
    except_session_id: str | None = None,
    now: datetime | None = None,
) -> int:
    statement = update(Session).where(Session.user_id == user_id, Session.revoked_at.is_(None))
    if except_session_id:
        statement = statement.where(Session.id != except_session_id)

    result = await db.execute(statement.values(revoked_at=now or utcnow(), revoked_reason=reason))
    # execute() is typed as Result; an UPDATE returns a CursorResult, which is
    # where rowcount lives.
    return int(cast("CursorResult[Any]", result).rowcount or 0)


async def list_active(db: AsyncSession, *, user_id: str) -> list[Session]:
    now = utcnow()
    rows = await db.execute(
        select(Session)
        .where(
            Session.user_id == user_id,
            Session.revoked_at.is_(None),
            Session.expires_at > now,
        )
        .order_by(Session.last_seen_at.desc())
    )
    return list(rows.scalars().all())


async def touch(db: AsyncSession, *, session_id: str, now: datetime | None = None) -> None:
    await db.execute(
        update(Session).where(Session.id == session_id).values(last_seen_at=now or utcnow())
    )


async def is_session_live(db: AsyncSession, session_id: str) -> bool:
    """Checked on every authenticated request.

    An access token is stateless and valid until it expires, so without this a
    revoked session would keep working for up to 15 minutes — which makes
    "revoke this device" a lie.
    """
    row = (
        await db.execute(
            select(Session.revoked_at, Session.absolute_expires_at).where(Session.id == session_id)
        )
    ).one_or_none()
    if row is None:
        return False
    revoked_at, absolute_expires_at = row
    return revoked_at is None and absolute_expires_at > utcnow()
