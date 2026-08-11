"""Auth and billing email bodies.

Plain text is the source of truth; the HTML is a light wrapper. Deliverability
improves with a text part, and these are short enough that HTML adds little.

The billing half at the bottom is not transactional plumbing — the price-change
and deprecation alerts are the product's retention mechanism (`PRD.md` §24).
The data moves on its own, which gives a user a reason to come back that does
not depend on remembering to.
"""

from __future__ import annotations

from datetime import datetime

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


# ── Billing (M20) ───────────────────────────────────────────────────────────

BILLING_URL = f"{settings.web_base_url}/settings/billing"


def trial_ending(*, to: str, name: str, plan: str, ends_at: datetime | None) -> Email:
    """Three days before a trial expires.

    Says what happens if they do nothing, because that is the decision being
    made. A reminder that only says "your trial is ending" leaves the reader to
    guess whether their saved work is at risk — and the answer, that nothing is
    deleted, is the reassuring half.
    """
    when = ends_at.strftime("%d %B") if ends_at else "in three days"
    return Email(
        to=to,
        subject=f"Your {BRAND} {plan} trial ends {when}",
        text=(
            f"Hi {name},\n\n"
            f"Your {plan} trial ends on {when}. Add a card to stay on {plan}:\n\n"
            f"{BILLING_URL}\n\n"
            "If you do nothing, your account moves to the Free plan. Nothing is "
            "deleted — your projects and saved stacks stay where they are, and you "
            "can still read and export them.\n"
        ),
        html=_wrap(
            f"Your {plan} trial ends {when}",
            f"<p>Hi {name}, your {plan} trial ends on {when}. Add a card to stay on "
            f"{plan}.</p><p>If you do nothing, your account moves to the Free plan. "
            "Nothing is deleted — your projects and saved stacks stay where they are, "
            "and you can still read and export them.</p>",
            ("Add a card", BILLING_URL),
        ),
    )


def payment_failed(*, to: str, name: str, grace_days: int) -> Email:
    return Email(
        to=to,
        subject=f"Your {BRAND} payment did not go through",
        text=(
            f"Hi {name},\n\n"
            "We could not charge your card. Your plan is unchanged for now and we "
            f"will keep retrying for {grace_days} days.\n\n"
            f"Update your payment method here:\n{BILLING_URL}\n\n"
            "If it still has not gone through after that, the account moves to the "
            "Free plan. Nothing is deleted either way.\n"
        ),
        html=_wrap(
            "Your payment did not go through",
            f"<p>Hi {name}, we could not charge your card. Your plan is unchanged for "
            f"now and we will keep retrying for {grace_days} days.</p>"
            "<p>If it still has not gone through after that, the account moves to the "
            "Free plan. Nothing is deleted either way.</p>",
            ("Update payment method", BILLING_URL),
        ),
    )


def price_change_alert(*, to: str, name: str, changes: list[dict[str, str]]) -> Email:
    """FR-18: a saved estimate is affected by more than 10 % drift.

    The figures are in the body rather than behind a link. The point of the
    alert is that the reader can tell in five seconds whether it matters, and a
    mail that only says "something changed, come and look" converts to a click
    from the people who least needed it.
    """
    lines = "\n".join(
        f"  · {change['name']}: {change['from']} → {change['to']} ({change['direction']})"
        for change in changes
    )
    rows = "".join(
        f"<li><strong>{change['name']}</strong>: {change['from']} → {change['to']} "
        f"({change['direction']})</li>"
        for change in changes
    )
    count = len(changes)
    noun = "price" if count == 1 else "prices"
    return Email(
        to=to,
        subject=f"{count} {noun} in your saved work changed",
        text=(
            f"Hi {name},\n\n"
            f"{count} {noun} you have saved an estimate against moved by more than "
            f"10 %:\n\n{lines}\n\n"
            f"Re-run the affected estimates here:\n{settings.web_base_url}/dashboard\n"
        ),
        html=_wrap(
            f"{count} {noun} in your saved work changed",
            f"<p>Hi {name}, {count} {noun} you have saved an estimate against moved by "
            f"more than 10 %:</p><ul>{rows}</ul>",
            ("Re-run the estimates", f"{settings.web_base_url}/dashboard"),
        ),
    )


def deprecation_alert(*, to: str, name: str, tools: list[dict[str, str]]) -> Email:
    """A tool in a saved stack has been flagged in the Graveyard.

    Names the replacement where the catalog has one. "X is deprecated" is a
    problem; "X is deprecated, and the catalog's replacement is Y" is a task.
    """
    lines = "\n".join(
        f"  · {tool['name']} — {tool['status']}"
        + (f", consider {tool['replacement']}" if tool.get("replacement") else "")
        for tool in tools
    )
    rows = "".join(
        f"<li><strong>{tool['name']}</strong> — {tool['status']}"
        + (f", consider {tool['replacement']}" if tool.get("replacement") else "")
        + "</li>"
        for tool in tools
    )
    count = len(tools)
    noun = "tool" if count == 1 else "tools"
    return Email(
        to=to,
        subject=f"{count} {noun} in your saved stacks {'is' if count == 1 else 'are'} deprecated",
        text=(
            f"Hi {name},\n\n"
            f"{count} {noun} in your saved stacks changed status:\n\n{lines}\n\n"
            f"Review the stacks here:\n{settings.web_base_url}/dashboard\n"
        ),
        html=_wrap(
            f"{count} {noun} in your saved stacks changed status",
            f"<p>Hi {name}, {count} {noun} in your saved stacks changed status:</p><ul>{rows}</ul>",
            ("Review your stacks", f"{settings.web_base_url}/dashboard"),
        ),
    )
