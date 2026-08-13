from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator

from app.models.user import Plan, UserRole


class _Base(BaseModel):
    model_config = ConfigDict(from_attributes=True, extra="forbid")


# ── Requests ────────────────────────────────────────────────────────────────


class RegisterRequest(_Base):
    email: EmailStr
    password: str = Field(min_length=12, max_length=256)
    name: str = Field(min_length=1, max_length=120)
    #: Signup-from-invite (M21). When present, the email must be the invited
    #: address, and the address is verified implicitly — possession of the
    #: invite link proves control of the inbox it was sent to.
    invite_token: str | None = Field(default=None, min_length=16, max_length=256)

    @field_validator("name")
    @classmethod
    def _strip_name(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("Name cannot be blank.")
        return stripped


class LoginRequest(_Base):
    email: EmailStr
    password: str = Field(min_length=1, max_length=256)


class VerifyEmailRequest(_Base):
    token: str = Field(min_length=16, max_length=256)


class ForgotPasswordRequest(_Base):
    email: EmailStr


class ResetPasswordRequest(_Base):
    token: str = Field(min_length=16, max_length=256)
    password: str = Field(min_length=12, max_length=256)


class ChangePasswordRequest(_Base):
    current_password: str = Field(min_length=1, max_length=256)
    new_password: str = Field(min_length=12, max_length=256)


class UpdateProfileRequest(_Base):
    name: str | None = Field(default=None, min_length=1, max_length=120)
    timezone: str | None = Field(default=None, max_length=64)
    avatar_url: str | None = Field(default=None, max_length=2048)


class ClaimAnonymousRequest(_Base):
    anonymous_id: str = Field(min_length=8, max_length=64)


# ── Responses ───────────────────────────────────────────────────────────────


class UserOut(_Base):
    id: str
    email: EmailStr
    name: str
    avatar_url: str | None
    timezone: str
    role: UserRole
    plan: Plan
    email_verified: bool
    must_set_password: bool
    created_at: datetime

    @classmethod
    def of(cls, user: object) -> UserOut:
        from app.models.user import User

        assert isinstance(user, User)
        return cls(
            id=user.id,
            email=user.email,
            name=user.name,
            avatar_url=user.avatar_url,
            timezone=user.timezone,
            role=user.role,
            plan=user.plan,
            email_verified=user.is_verified,
            must_set_password=user.must_set_password,
            created_at=user.created_at,
        )


class SessionTokens(_Base):
    """The refresh token is deliberately absent.

    It is delivered only as an HttpOnly cookie scoped to /api/v1/auth. Putting
    it in a JSON body would make it readable by any script on the page, which
    is the whole thing the cookie exists to prevent.
    """

    access_token: str
    token_type: str = "Bearer"  # noqa: S105
    expires_in: int


class AuthResult(_Base):
    user: UserOut
    tokens: SessionTokens


class RegisterResult(_Base):
    """Identical for a new address and an existing one.

    `verification_sent` is always true: an account was either created and sent
    a verification email, or already existed and was sent a notice. Neither
    the body nor the status code reveals which.
    """

    verification_sent: bool = True
    message: str = "Check your email to finish setting up your account."


class SimpleMessage(_Base):
    message: str


class SessionOut(_Base):
    id: str
    device_label: str | None
    ip: str | None
    created_at: datetime
    last_seen_at: datetime
    expires_at: datetime
    current: bool


class SessionListOut(_Base):
    sessions: list[SessionOut]


class AnonymousSessionOut(_Base):
    anonymous_id: str


class IdentityOut(_Base):
    """Unauthenticated-safe. Lets the client choose between the signed-in and
    anonymous experience on first load without a 401 round trip."""

    authenticated: bool
    user: UserOut | None
    anonymous_id: str | None
    plan: Plan
    server_time: datetime


class ClaimResult(_Base):
    claimed: bool
    reassigned: int
