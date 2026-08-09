"""Transactional email.

Three providers behind one interface: `console` (prints, for tests), `smtp`
(Mailpit locally), `resend` (production). The interface is narrow so the rest
of the app never learns which is in use.
"""

from __future__ import annotations

import re
import smtplib
from dataclasses import dataclass
from email.message import EmailMessage
from typing import Protocol

import httpx

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger("email")


@dataclass(frozen=True)
class Email:
    to: str
    subject: str
    text: str
    html: str | None = None


class EmailSender(Protocol):
    async def send(self, message: Email) -> None: ...


class ConsoleSender:
    """Test and CI default. Records what would have been sent so assertions can
    read the link out of it without a mail server.

    In local development it also prints any action link. Without that, an
    EMAIL_PROVIDER=console setup has no way to complete verification or reset —
    the mail goes nowhere and the token is only in the database as a hash.
    """

    def __init__(self) -> None:
        self.outbox: list[Email] = []

    async def send(self, message: Email) -> None:
        self.outbox.append(message)

        link: str | None = None
        if not settings.is_production:
            found = re.search(r"https?://\S+", message.text)
            link = found.group(0) if found else None

        logger.info(
            "email.console",
            to=message.to,
            subject=message.subject,
            **({"link": link} if link else {}),
        )


class SmtpSender:
    async def send(self, message: Email) -> None:
        payload = EmailMessage()
        payload["From"] = settings.email_from
        payload["To"] = message.to
        payload["Subject"] = message.subject
        payload.set_content(message.text)
        if message.html:
            payload.add_alternative(message.html, subtype="html")

        # smtplib is synchronous. Acceptable because this runs off the request
        # path in the worker; if it ever moves onto the request path, swap for
        # aiosmtplib rather than leaving a blocking call in the event loop.
        with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=10) as client:
            if settings.smtp_tls:
                client.starttls()
            if settings.smtp_user:
                client.login(settings.smtp_user, settings.smtp_password)
            client.send_message(payload)

        logger.info("email.sent", provider="smtp", to=message.to, subject=message.subject)


class ResendSender:
    async def send(self, message: Email) -> None:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.post(
                "https://api.resend.com/emails",
                headers={"Authorization": f"Bearer {settings.resend_api_key}"},
                json={
                    "from": settings.email_from,
                    "to": [message.to],
                    "subject": message.subject,
                    "text": message.text,
                    **({"html": message.html} if message.html else {}),
                },
            )
            response.raise_for_status()
        logger.info("email.sent", provider="resend", to=message.to, subject=message.subject)


_sender: EmailSender | None = None


def get_sender() -> EmailSender:
    global _sender
    if _sender is None:
        match settings.email_provider:
            case "smtp":
                _sender = SmtpSender()
            case "resend":
                _sender = ResendSender()
            case _:
                _sender = ConsoleSender()
    return _sender


def set_sender(sender: EmailSender | None) -> None:
    """Test seam."""
    global _sender
    _sender = sender


async def send(message: Email) -> None:
    """Never raises.

    A mail provider outage must not fail a registration — the user is created,
    and they can request another verification email. The failure is logged
    loudly because silent non-delivery is worse than a visible error.
    """
    try:
        await get_sender().send(message)
    except Exception as exc:
        logger.error("email.failed", to=message.to, subject=message.subject, error=str(exc))
