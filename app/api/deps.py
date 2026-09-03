"""Authorization dependencies.

Every gate is a dependency, never an `if` inside a route body. A dependency is
testable in isolation and cannot be forgotten when a new route is added — an
inline check can.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, Request, Response
from sqlalchemy.ext.asyncio import AsyncSession

from app.core import rate_limit
from app.core.context import bind
from app.core.database import get_session
from app.core.errors import (
    EmailNotVerified,
    Forbidden,
    PlanRequired,
    RateLimited,
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
    """A caller. Always a signed-in user.

    There used to be a second kind — an anonymous session keyed on a cookie —
    and `Identity` existed so quota, rate limiting, and run logging did not
    each have to branch on which they had. The anonymous tier is gone: every
    surface requires an account, so the union has collapsed to one member. The
    wrapper stays because `key` and `plan` are read in a few dozen places, and
    because it carries `session_id`, which the user row does not.
    """

    user: User
    session_id: str | None

    @property
    def key(self) -> str:
        return self.user.id

    @property
    def plan(self) -> Plan:
        return self.user.plan


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
    """The caller, or 401.

    Everything that used to be reachable without an account now sits behind
    `AuthGuard` on the web side and behind this on the API side, so there is no
    longer a "no identity" branch to fall through to.
    """
    user = await get_current_user(request, db)
    return Identity(user=user, session_id=getattr(request.state, "session_id", None))


#: Was distinct from `get_identity` while a tool run could be owned by an
#: anonymous session that had to be minted on the spot. Both now mean "a
#: signed-in caller"; the alias survives so the ~50 route signatures reading
#: `RunIdentity` still say what kind of endpoint they are.
get_run_identity = get_identity

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

    The verdict itself comes from `FeatureService`, never from a plan
    comparison written here.
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

# ── Rate limiting (M23) ─────────────────────────────────────────────────────


def _rate_limit_key(user: User | None, request: Request) -> str:
    """`user_id`, else IP.

    Identity before IP because IP-keyed limits break behind corporate NAT and
    shared Wi-Fi, which is where a good share of this audience works — one
    office would share a single allowance (D-14 makes the same argument for
    quota). The middle rung, an anonymous session id, went with the tier.

    IP is not a leftover: `/s/{token}` and the identity probe are reachable
    without a session by design, and they are keyed the only way a caller with
    no account can be.
    """
    if user is not None:
        return f"u:{user.id}"
    return f"ip:{_client_ip(request)}"


def _client_ip(request: Request) -> str:
    """The left-most `X-Forwarded-For` entry, or the socket peer.

    Only trusted because the API sits behind a proxy that overwrites the
    header. Read straight from `request.client` instead and every caller
    behind that proxy shares one allowance; trust it in a deploy without a
    proxy and any caller can spoof a fresh allowance per request.
    """
    forwarded = request.headers.get("X-Forwarded-For", "")
    if forwarded:
        first = forwarded.split(",")[0].strip()
        if first:
            return first
    client = request.client
    return client.host if client else "unknown"


def rate_limited(klass: rate_limit.RateLimitClass) -> Callable[..., Awaitable[None]]:
    """Build the dependency enforcing one policy class on a route.

    A dependency rather than middleware so it composes per route: health and
    the payment webhooks must not be limited, and a middleware would need a
    path allow-list that drifts from the routes it names.

    Keyed on `OptionalUser`, never `CallerIdentity`: the latter is a 401 for a
    caller with no session, and applying it here would put the whole shares
    router — including the public `/s/{token}` — behind the door.
    """

    async def enforce(request: Request, response: Response, user: OptionalUser) -> None:
        decision = await rate_limit.check(
            klass,
            identity_key=_rate_limit_key(user, request),
            plan=user.plan if user else None,
        )
        if decision is None:
            return

        # On the success path too, not only the refusal. A client that learns
        # its budget only by being refused finds out when it is already too
        # late to slow down.
        response.headers.update(decision.headers())
        if not decision.allowed:
            raise RateLimited(decision.retry_after, budget=decision.headers())

    return enforce


#: The two classes the contract names. Applied at the router, not per route,
#: so a new endpoint inherits a limit instead of quietly having none.
ReadLimit = Depends(rate_limited(rate_limit.READ))
ToolRunLimit = Depends(rate_limited(rate_limit.TOOL_RUN))
