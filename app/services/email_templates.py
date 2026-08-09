"""Auth email bodies.

Plain text is the source of truth; the HTML is a light wrapper. Deliverability
improves with a text part, and these are short enough that HTML adds little.
"""

from __future__ import annotations

from app.core.config import settings
from app.integrations.email import Email

BRAND = "StackForge"


def _wrap(title: str, body: str, action: tuple[str, str] | None = None) -> str:
    button = ""
    if action:
        label, url = action
        button = (
            f'<p style="margin:24px 0"><a href="{url}" '
            'style="background:#C4643C;color:#fff;text-decoration:none;'
            "padding:10px 18px;border-radius:6px;display:inline-block;"
            'font-weight:600">'
            f"{label}</a></p>"
        )
    return (
        '<div style="font-family:-apple-system,Segoe UI,Inter,sans-serif;'
        "max-width:520px;margin:0 auto;padding:32px 24px;color:#2B2724;"
        'background:#FAF9F5">'
        f'<h1 style="font-size:19px;margin:0 0 16px">{title}</h1>'
        f'<div style="font-size:14px;line-height:1.6;color:#4A443E">{body}</div>'
        f"{button}"
        '<hr style="border:none;border-top:1px solid #E8E4DC;margin:28px 0">'
        f'<p style="font-size:12px;color:#8A827A;margin:0">{BRAND} · '
        "If you did not expect this email you can safely ignore it.</p>"
        "</div>"
    )


def verify_email(*, to: str, name: str, token: str) -> Email:
    url = f"{settings.web_base_url}/verify-email?token={token}"
    hours = settings.email_verify_ttl_hours
    return Email(
        to=to,
        subject=f"Verify your {BRAND} email address",
        text=(
            f"Hi {name},\n\n"
            f"Confirm your email address to finish setting up your {BRAND} account:\n\n"
            f"{url}\n\n"
            f"This link expires in {hours} hours.\n"
        ),
        html=_wrap(
            "Confirm your email address",
            f"<p>Hi {name}, confirm your address to finish setting up your account. "
            f"This link expires in {hours} hours.</p>",
            ("Verify email", url),
        ),
    )


def password_reset(*, to: str, name: str, token: str) -> Email:
    url = f"{settings.web_base_url}/reset-password?token={token}"
    minutes = settings.password_reset_ttl_minutes
    return Email(
        to=to,
        subject=f"Reset your {BRAND} password",
        text=(
            f"Hi {name},\n\n"
            f"Use this link to choose a new password:\n\n{url}\n\n"
            f"It expires in {minutes} minutes and can be used once. "
            "Resetting will sign you out everywhere.\n"
        ),
        html=_wrap(
            "Reset your password",
            f"<p>Hi {name}, choose a new password. This link expires in {minutes} minutes "
            "and can be used once. Resetting will sign you out on every device.</p>",
            ("Choose a new password", url),
        ),
    )


def password_changed(*, to: str, name: str) -> Email:
    return Email(
        to=to,
        subject=f"Your {BRAND} password was changed",
        text=(
            f"Hi {name},\n\n"
            "Your password was just changed and every other session was signed out.\n\n"
            "If this was not you, reset your password immediately at "
            f"{settings.web_base_url}/forgot-password\n"
        ),
        html=_wrap(
            "Your password was changed",
            "<p>Your password was just changed and every other session was signed out. "
            "If this was not you, reset it immediately.</p>",
            ("Reset password", f"{settings.web_base_url}/forgot-password"),
        ),
    )


def suspicious_activity(*, to: str, name: str) -> Email:
    """Sent on refresh-token reuse.

    Two parties held tokens from one session. Both have been signed out; the
    user needs to know why and to consider the password compromised.
    """
    return Email(
        to=to,
        subject=f"Unusual activity on your {BRAND} account",
        text=(
            f"Hi {name},\n\n"
            "We detected a sign-in token being reused, which can mean it was copied "
            "from your device. As a precaution we signed out every session on your "
            "account.\n\n"
            "Sign in again, and change your password if you do not recognise this:\n"
            f"{settings.web_base_url}/login\n"
        ),
        html=_wrap(
            "Unusual activity on your account",
            "<p>We detected a sign-in token being reused, which can mean it was copied "
            "from your device. As a precaution we signed out every session. Sign in "
            "again, and change your password if you do not recognise this.</p>",
            ("Sign in", f"{settings.web_base_url}/login"),
        ),
    )


def registration_attempt(*, to: str) -> Email:
    """Sent when someone tries to register with an address that already exists.

    This is what lets `/register` return an identical response for new and
    existing addresses — the account holder is told, the caller learns nothing.
    """
    return Email(
        to=to,
        subject=f"Someone tried to create a {BRAND} account with your email",
        text=(
            "Someone just tried to sign up using this address, but an account "
            "already exists.\n\n"
            f"If that was you, sign in instead: {settings.web_base_url}/login\n"
            f"Forgot your password? {settings.web_base_url}/forgot-password\n"
        ),
        html=_wrap(
            "An account already exists",
            "<p>Someone just tried to sign up using this address, but an account "
            "already exists. If that was you, sign in instead.</p>",
            ("Sign in", f"{settings.web_base_url}/login"),
        ),
    )
