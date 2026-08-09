"""Error taxonomy.

`code` is stable and machine-readable. The frontend switches on `code`, never
on `message`. Adding a code is a minor version; changing what one means is
breaking.

Services raise these. Routers never build an error response by hand.
"""

from __future__ import annotations

from typing import Any


class AppError(Exception):
    code: str = "INTERNAL_ERROR"
    http_status: int = 500
    default_message: str = "Something went wrong."

    def __init__(
        self,
        message: str | None = None,
        *,
        details: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.message = message or self.default_message
        self.details = details
        self.headers = headers
        super().__init__(self.message)


# ── 4xx: the caller ─────────────────────────────────────────────────────────


class ValidationFailed(AppError):
    code = "VALIDATION_ERROR"
    http_status = 422
    default_message = "The request could not be validated."


class Unauthenticated(AppError):
    code = "UNAUTHENTICATED"
    http_status = 401
    default_message = "Authentication is required."


class TokenExpired(AppError):
    """Distinct from TOKEN_INVALID so the client knows to refresh and replay
    rather than send the user back to the login page."""

    code = "TOKEN_EXPIRED"
    http_status = 401
    default_message = "The access token has expired."


class TokenInvalid(AppError):
    code = "TOKEN_INVALID"
    http_status = 401
    default_message = "The credential is not valid."


class InvalidCredentials(AppError):
    """Never distinguishes 'no such user' from 'wrong password'. Doing so is
    an account-enumeration oracle."""

    code = "INVALID_CREDENTIALS"
    http_status = 401
    default_message = "Email or password is incorrect."


class EmailNotVerified(AppError):
    code = "EMAIL_NOT_VERIFIED"
    http_status = 403
    default_message = "Verify your email address to continue."


class AccountLocked(AppError):
    code = "ACCOUNT_LOCKED"
    http_status = 423
    default_message = "Too many failed attempts. Try again later."


class Forbidden(AppError):
    code = "FORBIDDEN"
    http_status = 403
    default_message = "You do not have access to this resource."


class PlanRequired(AppError):
    code = "PLAN_REQUIRED"
    http_status = 402
    default_message = "This feature requires a higher plan."


class QuotaExceeded(AppError):
    code = "QUOTA_EXCEEDED"
    http_status = 402
    default_message = "You have reached your usage limit."


class SeatsExceeded(AppError):
    code = "SEATS_EXCEEDED"
    http_status = 402
    default_message = "No seats remaining on this plan."


class NotFound(AppError):
    """Also returned for resources the caller may not see, and for revoked
    share links. A 403 would confirm the resource exists."""

    code = "NOT_FOUND"
    http_status = 404
    default_message = "Not found."


class Conflict(AppError):
    code = "CONFLICT"
    http_status = 409
    default_message = "That conflicts with something that already exists."


class RateLimited(AppError):
    code = "RATE_LIMITED"
    http_status = 429
    default_message = "Too many requests."

    def __init__(self, retry_after: int, message: str | None = None) -> None:
        super().__init__(
            message,
            details={"retry_after": retry_after},
            headers={"Retry-After": str(retry_after)},
        )


class PayloadTooLarge(AppError):
    code = "PAYLOAD_TOO_LARGE"
    http_status = 413
    default_message = "The uploaded file is too large."


# ── 5xx: us, or someone upstream ────────────────────────────────────────────


class UpstreamError(AppError):
    code = "UPSTREAM_ERROR"
    http_status = 502
    default_message = "An upstream service failed."


class InternalError(AppError):
    code = "INTERNAL_ERROR"
    http_status = 500
    default_message = "Something went wrong."
