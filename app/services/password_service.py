"""Password hashing and policy.

Argon2id, not bcrypt: bcrypt silently truncates input at 72 bytes and is
memory-cheap, so GPU attacks scale well against it. Argon2id is the current
OWASP first choice.
"""

from __future__ import annotations

import hashlib

import httpx
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerifyMismatchError

from app.core.config import settings
from app.core.errors import ValidationFailed
from app.core.logging import get_logger

logger = get_logger("auth.password")

MIN_LENGTH = 12
MAX_LENGTH = 256  # Argon2 will hash anything; a 1MB password is a DoS vector.

_hasher = PasswordHasher(
    time_cost=settings.argon2_time_cost,
    memory_cost=settings.argon2_memory_cost_kib,
    parallelism=settings.argon2_parallelism,
    hash_len=32,
    salt_len=16,
)

# A short deny-list of the passwords that dominate credential-stuffing lists.
# The real defence is the HIBP check below; this catches the worst offenders
# with no network call.
_COMMON = frozenset(
    {
        "password",
        "password1",
        "password123",
        "passw0rd",
        "12345678",
        "123456789",
        "1234567890",
        "qwertyuiop",
        "letmein123",
        "iloveyou123",
        "admin12345",
        "welcome123",
        "changeme123",
        "stackforge",
        "stackforge123",
    }
)


def hash_password(password: str) -> str:
    return _hasher.hash(password)


def verify_password(password_hash: str, password: str) -> bool:
    try:
        return _hasher.verify(password_hash, password)
    except (VerifyMismatchError, InvalidHashError):
        return False


def needs_rehash(password_hash: str) -> bool:
    """True when the stored hash is below current policy.

    Called after a successful verify so parameters can be raised over time
    without forcing a reset.
    """
    try:
        return _hasher.check_needs_rehash(password_hash)
    except InvalidHashError:
        return True


async def validate_password(password: str, *, email: str | None = None) -> None:
    """Raises ValidationFailed with field-level detail, or returns.

    No composition rules — they push users toward `Password1!` and buy nothing.
    Length, a deny-list, an email-similarity check, and a breach check.
    """
    problems: list[str] = []

    if len(password) < MIN_LENGTH:
        problems.append(f"Use at least {MIN_LENGTH} characters.")
    if len(password) > MAX_LENGTH:
        problems.append(f"Use at most {MAX_LENGTH} characters.")

    lowered = password.lower()
    if lowered in _COMMON:
        problems.append("That password is too common.")

    if email:
        local_part = email.split("@", 1)[0].lower()
        if len(local_part) >= 4 and local_part in lowered:
            problems.append("Do not include your email address in your password.")

    if not problems and settings.hibp_enabled and await _is_breached(password):
        problems.append(
            "That password has appeared in a known data breach. Choose a different one."
        )

    if problems:
        raise ValidationFailed(
            problems[0],
            details={"fields": [{"path": "password", "message": message} for message in problems]},
        )


async def _is_breached(password: str) -> bool:
    """Have I Been Pwned, k-anonymity.

    Only the first five characters of the SHA-1 hash leave the process — the
    password itself never does, and the service cannot determine which suffix
    was being checked.

    Fails open: an outage at HIBP must not block signups.
    """
    digest = hashlib.sha1(password.encode(), usedforsecurity=False).hexdigest().upper()
    prefix, suffix = digest[:5], digest[5:]

    try:
        async with httpx.AsyncClient(timeout=2.5) as client:
            response = await client.get(
                f"https://api.pwnedpasswords.com/range/{prefix}",
                headers={"Add-Padding": "true"},
            )
            response.raise_for_status()
    except Exception as exc:
        logger.warning("hibp.unavailable", error=type(exc).__name__)
        return False

    for line in response.text.splitlines():
        candidate, _, count = line.partition(":")
        if candidate == suffix and count.strip() not in ("", "0"):
            return True
    return False
