"""Access tokens (JWT, EdDSA) and opaque secrets.

EdDSA over HS256 so the public key can be distributed if a second service ever
needs to verify without holding a signing secret. `kid` supports rotation.
"""

from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import jwt

from app.core.config import settings
from app.core.errors import TokenExpired, TokenInvalid

ALGORITHM = "EdDSA"
ISSUER = "stackforge"
AUDIENCE = "stackforge-api"


@dataclass(frozen=True)
class AccessClaims:
    user_id: str
    session_id: str
    plan: str
    role: str
    verified: bool


def create_access_token(
    *, user_id: str, session_id: str, plan: str, role: str, verified: bool
) -> tuple[str, int]:
    """Returns (token, expires_in_seconds).

    `plan` and `role` are hints for rendering only. Every gate is re-checked
    server-side against the database — a 15-minute-stale plan claim would
    otherwise let a downgraded user keep paid features for a quarter of an hour.
    """
    now = datetime.now(UTC)
    ttl = settings.access_token_ttl_seconds
    payload: dict[str, Any] = {
        "sub": user_id,
        "sid": session_id,
        "typ": "access",
        "plan": plan,
        "rol": role,
        "ver": verified,
        "iss": ISSUER,
        "aud": AUDIENCE,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(seconds=ttl)).timestamp()),
        "jti": secrets.token_urlsafe(16),
    }
    token = jwt.encode(
        payload,
        settings.auth_private_key,
        algorithm=ALGORITHM,
        headers={"kid": settings.auth_key_id},
    )
    return token, ttl


def decode_access_token(token: str) -> AccessClaims:
    try:
        payload = jwt.decode(
            token,
            settings.auth_public_key,
            algorithms=[ALGORITHM],
            audience=AUDIENCE,
            issuer=ISSUER,
            options={"require": ["exp", "iat", "sub", "sid"]},
        )
    except jwt.ExpiredSignatureError as exc:
        raise TokenExpired() from exc
    except jwt.InvalidTokenError as exc:
        raise TokenInvalid() from exc

    if payload.get("typ") != "access":
        raise TokenInvalid()

    return AccessClaims(
        user_id=payload["sub"],
        session_id=payload["sid"],
        plan=payload.get("plan", "free"),
        role=payload.get("rol", "user"),
        verified=bool(payload.get("ver", False)),
    )


# ── Opaque secrets ──────────────────────────────────────────────────────────


def generate_secret(bytes_length: int = 32) -> str:
    """256 bits by default. Used for refresh tokens, verification tokens, and
    share links — anything where there is nothing to read and a JWT would only
    leak claims."""
    return secrets.token_urlsafe(bytes_length)


def hash_secret(secret: str) -> str:
    """SHA-256, not Argon2.

    These are high-entropy random values, not user-chosen passwords: there is
    no dictionary to attack, so a slow KDF buys nothing and would make every
    refresh cost 250ms.
    """
    return hashlib.sha256(secret.encode()).hexdigest()


def constant_time_equals(left: str, right: str) -> bool:
    return secrets.compare_digest(left, right)


# ── Key generation (used by the CLI) ────────────────────────────────────────


def generate_keypair() -> tuple[str, str]:
    """Ed25519 keypair as PEM strings."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ed25519

    private = ed25519.Ed25519PrivateKey.generate()
    private_pem = private.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode()
    public_pem = (
        private.public_key()
        .public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        .decode()
    )
    return private_pem, public_pem
