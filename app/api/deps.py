"""Authorization dependencies.

Every gate is a dependency, never an `if` inside a route body. A dependency is
testable in isolation and cannot be forgotten when a new route is added — an
inline check can.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.context import bind
from app.core.database import get_session
from app.core.errors import (
    EmailNotVerified,
    Forbidden,
    PlanRequired,
    TokenInvalid,
    Unauthenticated,
)
from app.models.user import Plan, User, UserRole
from app.services import auth_service, session_service, token_service

Db = Annotated[AsyncSession, Depends(get_session)]

PLAN_RANK = {Plan.FREE: 0, Plan.PRO: 1, Plan.TEAM: 2, Plan.ENTERPRISE: 3}


@dataclass(frozen=True)
class Identity:
    """A caller. Either a signed-in user or an anonymous session, never both.

    Quota, rate limiting, and run logging key on `key`, so none of them has to
    branch on which kind of caller they have.
    """

    user: User | None
    anonymous_id: str | None
    session_id: str | None

    @property
    def is_authenticated(self) -> bool:
        return self.user is not None

    @property
    def key(self) -> str:
        if self.user:
            return self.user.id
        return self.anonymous_id or "unknown"

    @property
    def plan(self) -> Plan:
        return self.user.plan if self.user else Plan.FREE


def request_meta(request: Request) -> auth_service.RequestMeta:
    return auth_service.RequestMeta(
        ip=request.client.host if request.client else None,
        user_agent=request.headers.get("user-agent"),
    )


RequestMeta = Annotated[auth_service.RequestMeta, Depends(request_meta)]


def _bearer(request: Request) -> str | None:
    header = request.headers.get("Authorization", "")
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or not token:
        return None
    return token


async def get_current_user_optional(request: Request, db: Db) -> User | None:
    """Resolve a user from the access token, or None.

    Two checks that a stateless JWT alone would skip:
      * the session must still be live, so revoking a device takes effect on
        the next request rather than in up to 15 minutes;
      * the plan is read from the database, never from the token claim.
    """
    token = _bearer(request)
    if not token:
        return None

    claims = token_service.decode_access_token(token)  # raises TokenExpired/Invalid

    if not await session_service.is_session_live(db, claims.session_id):
        raise TokenInvalid("This session has been signed out.")

    user = await auth_service.get_user(db, claims.user_id)
    if user is None:
        raise TokenInvalid()

    request.state.session_id = claims.session_id
    bind(user_id=user.id, plan=user.plan.value)
    return user


async def get_identity(request: Request, db: Db) -> Identity:
    """User, else anonymous session, else anonymous-without-a-cookie."""
    user = await get_current_user_optional(request, db)
    if user is not None:
        return Identity(
            user=user,
            anonymous_id=None,
            session_id=getattr(request.state, "session_id", None),
        )

    anon_id = request.cookies.get(settings.anon_cookie_name)
    if anon_id:
        bind(anonymous_id=anon_id)
    return Identity(user=None, anonymous_id=anon_id, session_id=None)


async def get_current_user(request: Request, db: Db) -> User:
    user = await get_current_user_optional(request, db)
    if user is None:
        raise Unauthenticated()
    return user


CurrentUser = Annotated[User, Depends(get_current_user)]
OptionalUser = Annotated[User | None, Depends(get_current_user_optional)]
CallerIdentity = Annotated[Identity, Depends(get_identity)]


async def require_verified(user: CurrentUser) -> User:
    if not user.is_verified:
        raise EmailNotVerified()
    return user


VerifiedUser = Annotated[User, Depends(require_verified)]


async def require_admin(user: CurrentUser) -> User:
    if user.role is not UserRole.ADMIN:
        raise Forbidden()
    return user


AdminUser = Annotated[User, Depends(require_admin)]


def require_plan(minimum: Plan) -> object:
    """Plan gate.

    Compares against the database column, not the JWT claim — the claim can be
    up to one access-token lifetime stale, which is exactly long enough for a
    downgraded account to keep using a paid feature.
    """

    async def dependency(user: CurrentUser) -> User:
        if PLAN_RANK[user.plan] < PLAN_RANK[minimum]:
            raise PlanRequired(
                f"This feature requires the {minimum.value.title()} plan.",
                details={"required_plan": minimum.value, "current_plan": user.plan.value},
            )
        return user

    return Depends(dependency)


def current_session_id(request: Request) -> str | None:
    session_id: str | None = getattr(request.state, "session_id", None)
    return session_id
