"""Authorization dependencies.

Every gate is a dependency, never an `if` inside a route body. A dependency is
testable in isolation and cannot be forgotten when a new route is added — an
inline check can.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, Request, Response
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
from app.data.plans import Feature
from app.models.billing import Metric
from app.models.organization import Organization, OrganizationMember, OrgRole
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


def set_anon_cookie(response: Response, anon_id: str) -> None:
    """Path `/`, so every tool endpoint sees it — unlike the refresh cookie,
    which is deliberately scoped to `/api/v1/auth`."""
    response.set_cookie(
        settings.anon_cookie_name,
        anon_id,
        max_age=30 * 86_400,
        httponly=True,
        secure=settings.cookie_secure,
        samesite="lax",
        path="/",
        domain=settings.cookie_domain,
    )


async def get_run_identity(
    request: Request, response: Response, db: Db, meta: RequestMeta
) -> Identity:
    """Like `get_identity`, but guarantees an owner.

    Every `tool_runs` row must belong to exactly one user or anonymous session
    — a check constraint enforces it, because orphan rows vanish from
    per-user queries while still inflating the totals. A caller arriving with
    no cookie at all is normal (a shared link, a cold first request, a curl),
    so rather than reject them this mints an anonymous session and sets the
    cookie, and the work becomes claimable if they sign up later.

    Also revalidates the cookie against the database: a stale `anon_` id from
    a purged session would otherwise fail the foreign key at insert time,
    turning a routine request into a 409.
    """
    user = await get_current_user_optional(request, db)
    if user is not None:
        return Identity(
            user=user,
            anonymous_id=None,
            session_id=getattr(request.state, "session_id", None),
        )

    anon_id = request.cookies.get(settings.anon_cookie_name)
    if anon_id:
        existing = await auth_service.get_anonymous_session(db, anon_id)
        if existing is not None:
            bind(anonymous_id=existing.id)
            return Identity(user=None, anonymous_id=existing.id, session_id=None)

    record = await auth_service.create_anonymous_session(db, meta=meta)
    set_anon_cookie(response, record.id)
    bind(anonymous_id=record.id)
    return Identity(user=None, anonymous_id=record.id, session_id=None)


RunIdentity = Annotated[Identity, Depends(get_run_identity)]


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


def require_feature(feature: Feature) -> object:
    """Feature gate (M20).

    Prefer this over `require_plan`: it names the capability rather than the
    tier, so moving PDF export from Pro to Team is one edit in `data/plans.py`
    instead of a search for `require_plan(Plan.PRO)` across the routers.

    Unlike `require_plan` this accepts an anonymous caller, because some
    features are open to one — the verdict, including "you need an account
    rather than a bigger plan", comes from `FeatureService`.
    """

    async def dependency(identity: CallerIdentity) -> Identity:
        from app.services import feature_service

        feature_service.require(identity, feature)
        return identity

    return Depends(dependency)


def consume_quota(metric: Metric, amount: int = 1) -> object:
    """Meter gate (M20).

    Declared as a dependency rather than called inside the handler for the
    reason at the top of this file: a gate in a route body is a gate the next
    endpoint forgets. It consumes *before* the handler runs, which is the
    conservative order — a handler that then fails has cost the user one unit
    of allowance, and the alternative is a handler that succeeds without ever
    being counted.

    The tool engine is the exception and consumes explicitly, because it has a
    persist step that must be inside the same transaction as the count.
    """

    async def dependency(identity: CallerIdentity, db: Db) -> Identity:
        from app.services import feature_service

        await feature_service.consume(db, identity, metric, amount)
        return identity

    return Depends(dependency)


def current_session_id(request: Request) -> str | None:
    session_id: str | None = getattr(request.state, "session_id", None)
    return session_id


# ── Organization scope (M21) ────────────────────────────────────────────────


@dataclass(frozen=True)
class OrgContext:
    """A caller acting inside an organization they belong to.

    Existence of this object *is* the authorization: it can only be built
    through membership resolution, which returns the same `NotFound` for
    another organization's id as for one that does not exist.
    """

    org: Organization
    member: OrganizationMember
    user: User

    @property
    def role(self) -> OrgRole:
        return self.member.role


def require_org_role(minimum: OrgRole) -> object:
    """Org-role gate.

    Same doctrine as every other gate: a dependency, never an `if` in a route
    body. A non-member gets 404 (cross-org isolation — a 403 would confirm the
    org exists); a member below `minimum` gets 403, because they already know
    the org exists.

    The route must declare an `organization_id` path parameter.
    """

    async def dependency(organization_id: str, user: CurrentUser, db: Db) -> OrgContext:
        from app.services import organization_service

        org, member = await organization_service.get_membership(
            db, user=user, organization_id=organization_id
        )
        if not member.role.covers(minimum):
            raise Forbidden("Your role in this organization does not allow that.")
        bind(organization_id=org.id)
        return OrgContext(org=org, member=member, user=user)

    return Depends(dependency)


ORG_ID_HEADER = "X-Organization-Id"


async def get_current_membership(
    request: Request, user: OptionalUser, db: Db
) -> OrganizationMember | None:
    """The org switcher's scope, from the `X-Organization-Id` header.

    `None` — no header or no user — means personal scope, which is every
    request made before M21 existed. A header naming an org the caller is not
    a member of is the same 404 as one that does not exist.
    """
    organization_id = request.headers.get(ORG_ID_HEADER)
    if not organization_id or user is None:
        return None

    from app.services import organization_service

    org, member = await organization_service.get_membership(
        db, user=user, organization_id=organization_id
    )
    bind(organization_id=org.id)
    return member


CurrentMembership = Annotated[OrganizationMember | None, Depends(get_current_membership)]

#: The four gates, pre-built. A route takes the weakest one that suffices —
#: the names mirror the role matrix in `04-Authentication-Design`.
OrgViewer = Annotated[OrgContext, require_org_role(OrgRole.VIEWER)]
OrgEditor = Annotated[OrgContext, require_org_role(OrgRole.MEMBER)]
OrgAdmin = Annotated[OrgContext, require_org_role(OrgRole.ADMIN)]
OrgOwner = Annotated[OrgContext, require_org_role(OrgRole.OWNER)]
