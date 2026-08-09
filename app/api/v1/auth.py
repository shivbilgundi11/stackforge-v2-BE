from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request, Response, status

from app.api.deps import (
    CallerIdentity,
    CurrentUser,
    Db,
    RequestMeta,
    current_session_id,
)
from app.core.config import settings
from app.core.database import utcnow
from app.core.errors import Unauthenticated
from app.core.logging import get_logger
from app.core.responses import Envelope, ok
from app.integrations import email as email_integration
from app.models.auth import AuthEventType
from app.schemas.auth import (
    AnonymousSessionOut,
    AuthResult,
    ChangePasswordRequest,
    ClaimAnonymousRequest,
    ClaimResult,
    ForgotPasswordRequest,
    IdentityOut,
    LoginRequest,
    RegisterRequest,
    RegisterResult,
    ResetPasswordRequest,
    SessionListOut,
    SessionOut,
    SessionTokens,
    SimpleMessage,
    UpdateProfileRequest,
    UserOut,
    VerifyEmailRequest,
)
from app.services import auth_service, email_templates, session_service, token_service

logger = get_logger("auth.api")
router = APIRouter(tags=["auth"])

REFRESH_PATH = "/api/v1/auth"


# ── Cookies ─────────────────────────────────────────────────────────────────


def _set_refresh_cookie(response: Response, token: str) -> None:
    """HttpOnly, Secure, SameSite=Lax, and path-scoped to /api/v1/auth.

    The path scope is the cheap win: the token is not sent on the other ~130
    endpoints, which is a large reduction in exposure for one line.
    """
    response.set_cookie(
        settings.refresh_cookie_name,
        token,
        max_age=settings.refresh_token_ttl_days * 86_400,
        httponly=True,
        secure=settings.cookie_secure,
        samesite="lax",
        path=REFRESH_PATH,
        domain=settings.cookie_domain,
    )


def _clear_refresh_cookie(response: Response) -> None:
    response.delete_cookie(
        settings.refresh_cookie_name,
        path=REFRESH_PATH,
        domain=settings.cookie_domain,
        httponly=True,
        secure=settings.cookie_secure,
        samesite="lax",
    )


def _set_anon_cookie(response: Response, anon_id: str) -> None:
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


def _tokens_for(user: Any, session_id: str) -> SessionTokens:
    access, expires_in = token_service.create_access_token(
        user_id=user.id,
        session_id=session_id,
        plan=user.plan.value,
        role=user.role.value,
        verified=user.is_verified,
    )
    return SessionTokens(access_token=access, expires_in=expires_in)


# ── Registration and login ──────────────────────────────────────────────────


@router.post(
    "/register",
    response_model=Envelope[RegisterResult],
    name="register",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Create an account",
    response_description="Always the same shape, whether or not the address existed.",
)
async def register(payload: RegisterRequest, db: Db, meta: RequestMeta) -> dict[str, Any]:
    await auth_service.register(
        db,
        email=payload.email,
        password=payload.password,
        name=payload.name,
        meta=meta,
    )
    # No branch on the result. A different status or body for an existing
    # address is an enumeration oracle.
    return ok(RegisterResult())


@router.post("/login", response_model=Envelope[AuthResult], name="login", summary="Sign in")
async def login(
    payload: LoginRequest, response: Response, db: Db, meta: RequestMeta
) -> dict[str, Any]:
    user = await auth_service.authenticate(
        db, email=payload.email, password=payload.password, meta=meta
    )
    issued = await session_service.issue(
        db, user_id=user.id, ip=meta.ip, user_agent=meta.user_agent
    )
    _set_refresh_cookie(response, issued.refresh_token)

    return ok(AuthResult(user=UserOut.of(user), tokens=_tokens_for(user, issued.session.id)))


@router.post(
    "/refresh",
    response_model=Envelope[AuthResult],
    name="refresh",
    summary="Rotate the refresh token",
)
async def refresh(
    request: Request, response: Response, db: Db, meta: RequestMeta
) -> dict[str, Any]:
    presented = request.cookies.get(settings.refresh_cookie_name)
    if not presented:
        raise Unauthenticated("No refresh token was presented.")

    result = await session_service.rotate(
        db, presented_token=presented, ip=meta.ip, user_agent=meta.user_agent
    )

    if result.reuse_detected:
        # Two parties held tokens from this family. Both are now signed out.
        user = await auth_service.get_user(db, result.session.user_id) if result.session else None
        if user:
            await auth_service.record_event(
                db,
                event=AuthEventType.REFRESH_REUSE_DETECTED,
                user_id=user.id,
                meta=meta,
            )
            await email_integration.send(
                email_templates.suspicious_activity(to=user.email, name=user.name)
            )
        # Same reason as the failed-login path: this 401 would roll back the
        # family revocation that is the entire point of detecting reuse.
        await db.commit()
        _clear_refresh_cookie(response)
        raise Unauthenticated("This session has been signed out. Please sign in again.")

    if result.session is None or result.refresh_token is None:
        _clear_refresh_cookie(response)
        raise Unauthenticated("This session has expired. Please sign in again.")

    user = await auth_service.get_user(db, result.session.user_id)
    if user is None:
        _clear_refresh_cookie(response)
        raise Unauthenticated()

    _set_refresh_cookie(response, result.refresh_token)
    await auth_service.record_event(db, event=AuthEventType.REFRESH, user_id=user.id, meta=meta)

    return ok(AuthResult(user=UserOut.of(user), tokens=_tokens_for(user, result.session.id)))


@router.post(
    "/logout",
    response_model=Envelope[SimpleMessage],
    name="logout",
    summary="Sign out of this device",
)
async def logout(request: Request, response: Response, db: Db, meta: RequestMeta) -> dict[str, Any]:
    presented = request.cookies.get(settings.refresh_cookie_name)
    if presented:
        existing = await session_service.get_by_token(db, presented)
        if existing:
            await session_service.revoke(
                db, session_id=existing.id, reason=session_service.RevokeReason.LOGOUT
            )
            await auth_service.record_event(
                db, event=AuthEventType.LOGOUT, user_id=existing.user_id, meta=meta
            )

    _clear_refresh_cookie(response)
    # Always 200. Signing out must succeed even from an already-dead session,
    # or the client is stuck holding a cookie it cannot clear.
    return ok(SimpleMessage(message="Signed out."))


@router.post(
    "/logout-all",
    response_model=Envelope[SimpleMessage],
    name="logout_all",
    summary="Sign out everywhere",
)
async def logout_all(
    response: Response, db: Db, user: CurrentUser, meta: RequestMeta
) -> dict[str, Any]:
    count = await session_service.revoke_all_for_user(
        db, user_id=user.id, reason=session_service.RevokeReason.LOGOUT_ALL
    )
    await auth_service.record_event(
        db,
        event=AuthEventType.LOGOUT_ALL,
        user_id=user.id,
        meta=meta,
        metadata={"revoked": str(count)},
    )
    _clear_refresh_cookie(response)
    return ok(SimpleMessage(message=f"Signed out of {count} sessions."))


# ── Current user ────────────────────────────────────────────────────────────


@router.get("/me", response_model=Envelope[UserOut], name="me", summary="The signed-in user")
async def me(user: CurrentUser) -> dict[str, Any]:
    return ok(UserOut.of(user))


@router.patch(
    "/profile", response_model=Envelope[UserOut], name="update_profile", summary="Update profile"
)
async def update_profile(
    payload: UpdateProfileRequest, db: Db, user: CurrentUser
) -> dict[str, Any]:
    if payload.name is not None:
        user.name = payload.name.strip()
    if payload.timezone is not None:
        user.timezone = payload.timezone
    if payload.avatar_url is not None:
        user.avatar_url = payload.avatar_url or None
    await db.flush()
    return ok(UserOut.of(user))


@router.patch(
    "/password",
    response_model=Envelope[SimpleMessage],
    name="change_password",
    summary="Change password",
)
async def change_password(
    payload: ChangePasswordRequest,
    request: Request,
    db: Db,
    user: CurrentUser,
    meta: RequestMeta,
) -> dict[str, Any]:
    await auth_service.change_password(
        db,
        user=user,
        current_password=payload.current_password,
        new_password=payload.new_password,
        keep_session_id=current_session_id(request),
        meta=meta,
    )
    return ok(SimpleMessage(message="Password changed. Other devices were signed out."))


@router.delete(
    "/account",
    response_model=Envelope[SimpleMessage],
    name="delete_account",
    summary="Delete this account",
)
async def delete_account(
    response: Response, db: Db, user: CurrentUser, meta: RequestMeta
) -> dict[str, Any]:
    await auth_service.soft_delete_account(db, user=user, meta=meta)
    _clear_refresh_cookie(response)
    return ok(SimpleMessage(message="Account scheduled for deletion."))


# ── Email verification ──────────────────────────────────────────────────────


@router.post(
    "/verify-email",
    response_model=Envelope[UserOut],
    name="verify_email",
    summary="Confirm an email address",
)
async def verify_email(payload: VerifyEmailRequest, db: Db, meta: RequestMeta) -> dict[str, Any]:
    user = await auth_service.verify_email(db, token=payload.token, meta=meta)
    return ok(UserOut.of(user))


@router.post(
    "/verify-email/resend",
    response_model=Envelope[SimpleMessage],
    name="resend_verification",
    summary="Send another verification email",
)
async def resend_verification(db: Db, user: CurrentUser) -> dict[str, Any]:
    await auth_service.resend_verification(db, user=user)
    return ok(SimpleMessage(message="Verification email sent."))


# ── Password reset ──────────────────────────────────────────────────────────


@router.post(
    "/forgot-password",
    response_model=Envelope[SimpleMessage],
    name="forgot_password",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Request a password reset",
)
async def forgot_password(
    payload: ForgotPasswordRequest, db: Db, meta: RequestMeta
) -> dict[str, Any]:
    await auth_service.request_password_reset(db, email=payload.email, meta=meta)
    # Same response regardless of whether the address exists.
    return ok(
        SimpleMessage(message="If an account exists for that address, a reset link is on its way.")
    )


@router.post(
    "/reset-password",
    response_model=Envelope[SimpleMessage],
    name="reset_password",
    summary="Set a new password",
)
async def reset_password(
    payload: ResetPasswordRequest, response: Response, db: Db, meta: RequestMeta
) -> dict[str, Any]:
    await auth_service.reset_password(
        db, token=payload.token, new_password=payload.password, meta=meta
    )
    _clear_refresh_cookie(response)
    return ok(SimpleMessage(message="Password reset. Sign in with your new password."))


# ── Sessions ────────────────────────────────────────────────────────────────


@router.get(
    "/sessions",
    response_model=Envelope[SessionListOut],
    name="list_sessions",
    summary="Active sessions",
)
async def list_sessions(request: Request, db: Db, user: CurrentUser) -> dict[str, Any]:
    active = await session_service.list_active(db, user_id=user.id)
    this_session = current_session_id(request)
    return ok(
        SessionListOut(
            sessions=[
                SessionOut(
                    id=item.id,
                    device_label=item.device_label,
                    ip=str(item.ip) if item.ip else None,
                    created_at=item.created_at,
                    last_seen_at=item.last_seen_at,
                    expires_at=item.expires_at,
                    current=item.id == this_session,
                )
                for item in active
            ]
        )
    )


@router.delete(
    "/sessions/{session_id}",
    response_model=Envelope[SimpleMessage],
    name="revoke_session",
    summary="Revoke a session",
)
async def revoke_session(
    session_id: str, db: Db, user: CurrentUser, meta: RequestMeta
) -> dict[str, Any]:
    owned = {item.id for item in await session_service.list_active(db, user_id=user.id)}
    if session_id not in owned:
        # 404, not 403. A 403 confirms the session exists.
        from app.core.errors import NotFound

        raise NotFound()

    await session_service.revoke(
        db, session_id=session_id, reason=session_service.RevokeReason.USER
    )
    await auth_service.record_event(
        db,
        event=AuthEventType.SESSION_REVOKED,
        user_id=user.id,
        meta=meta,
        metadata={"session_id": session_id},
    )
    return ok(SimpleMessage(message="Session revoked."))


# ── Anonymous identity ──────────────────────────────────────────────────────


@router.post(
    "/anonymous",
    response_model=Envelope[AnonymousSessionOut],
    name="create_anonymous",
    summary="Start an anonymous session",
)
async def create_anonymous(
    request: Request, response: Response, db: Db, meta: RequestMeta
) -> dict[str, Any]:
    existing = request.cookies.get(settings.anon_cookie_name)
    if existing and await auth_service.get_anonymous_session(db, existing):
        return ok(AnonymousSessionOut(anonymous_id=existing))

    record = await auth_service.create_anonymous_session(db, meta=meta)
    _set_anon_cookie(response, record.id)
    return ok(AnonymousSessionOut(anonymous_id=record.id))


@router.post(
    "/claim",
    response_model=Envelope[ClaimResult],
    name="claim_anonymous",
    summary="Attach anonymous work to this account",
)
async def claim_anonymous(
    payload: ClaimAnonymousRequest, db: Db, user: CurrentUser, meta: RequestMeta
) -> dict[str, Any]:
    reassigned = await auth_service.claim_anonymous_session(
        db, anon_id=payload.anonymous_id, user=user, meta=meta
    )
    return ok(ClaimResult(claimed=True, reassigned=reassigned))


# ── Identity probe ──────────────────────────────────────────────────────────


@router.get(
    "/identity", response_model=Envelope[IdentityOut], name="identity", summary="Who is calling"
)
async def identity(caller: CallerIdentity) -> dict[str, Any]:
    """Unauthenticated-safe. Lets the client decide between the signed-in and
    anonymous experience without a 401 round trip on first load."""
    return ok(
        IdentityOut(
            authenticated=caller.is_authenticated,
            user=UserOut.of(caller.user) if caller.user else None,
            anonymous_id=caller.anonymous_id,
            plan=caller.plan,
            server_time=utcnow(),
        )
    )
