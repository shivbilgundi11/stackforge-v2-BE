"""Unit tests for the security primitives.

Auth is the module where a missing test is a hole rather than an
inconvenience, so these assert behaviour rather than that a function returns.
"""

from __future__ import annotations

import time
from datetime import timedelta

import jwt
import pytest

from app.core.errors import TokenExpired, TokenInvalid, ValidationFailed
from app.services import password_service, token_service


class TestPasswordHashing:
    def test_hash_verifies_and_is_salted(self) -> None:
        digest_a = password_service.hash_password("correct-horse-battery-staple")
        digest_b = password_service.hash_password("correct-horse-battery-staple")

        assert digest_a != digest_b, "identical passwords must not share a hash"
        assert password_service.verify_password(digest_a, "correct-horse-battery-staple")
        assert password_service.verify_password(digest_b, "correct-horse-battery-staple")

    def test_wrong_password_fails(self) -> None:
        digest = password_service.hash_password("correct-horse-battery-staple")
        assert not password_service.verify_password(digest, "wrong-horse-battery-staple")

    def test_argon2id_is_the_algorithm(self) -> None:
        assert password_service.hash_password("x" * 20).startswith("$argon2id$")

    def test_long_password_is_not_truncated(self) -> None:
        """bcrypt caps at 72 bytes; Argon2 must not. Two passwords sharing a
        72-byte prefix have to be distinguishable."""
        base = "a" * 72
        digest = password_service.hash_password(base + "TAIL-ONE")
        assert not password_service.verify_password(digest, base + "TAIL-TWO")

    def test_malformed_hash_does_not_raise(self) -> None:
        assert not password_service.verify_password("not-a-hash", "whatever")


class TestPasswordPolicy:
    async def test_accepts_a_reasonable_password(self) -> None:
        await password_service.validate_password("correct-horse-battery-staple")

    async def test_rejects_short(self) -> None:
        with pytest.raises(ValidationFailed) as exc:
            await password_service.validate_password("short")
        assert exc.value.details is not None
        assert exc.value.details["fields"][0]["path"] == "password"

    async def test_rejects_common(self) -> None:
        with pytest.raises(ValidationFailed):
            await password_service.validate_password("password123")

    async def test_rejects_password_containing_email_local_part(self) -> None:
        with pytest.raises(ValidationFailed):
            await password_service.validate_password(
                "adalovelace-is-me", email="adalovelace@example.com"
            )

    async def test_short_local_part_does_not_trigger(self) -> None:
        """A two-character local part appears in almost any string; matching on
        it would reject valid passwords."""
        await password_service.validate_password("correct-horse-battery", email="ad@example.com")


class TestAccessTokens:
    def test_round_trip(self) -> None:
        token, expires_in = token_service.create_access_token(
            user_id="usr_1", session_id="ses_1", plan="pro", role="user", verified=True
        )
        claims = token_service.decode_access_token(token)

        assert claims.user_id == "usr_1"
        assert claims.session_id == "ses_1"
        assert claims.plan == "pro"
        assert claims.verified is True
        assert expires_in > 0

    def test_tampered_signature_is_rejected(self) -> None:
        token, _ = token_service.create_access_token(
            user_id="usr_1", session_id="ses_1", plan="free", role="user", verified=False
        )
        header, payload, signature = token.split(".")
        forged = f"{header}.{payload}.{signature[:-4]}AAAA"

        with pytest.raises(TokenInvalid):
            token_service.decode_access_token(forged)

    def test_expired_token_raises_token_expired_not_invalid(self) -> None:
        """The distinction is load-bearing: the client refreshes on
        TOKEN_EXPIRED and sends the user to login on TOKEN_INVALID."""
        from app.core.config import settings

        payload = {
            "sub": "usr_1",
            "sid": "ses_1",
            "typ": "access",
            "iss": token_service.ISSUER,
            "aud": token_service.AUDIENCE,
            "iat": int(time.time()) - 100,
            "exp": int(time.time()) - 10,
        }
        expired = jwt.encode(payload, settings.auth_private_key, algorithm="EdDSA")

        with pytest.raises(TokenExpired):
            token_service.decode_access_token(expired)

    def test_wrong_audience_is_rejected(self) -> None:
        from app.core.config import settings

        payload = {
            "sub": "usr_1",
            "sid": "ses_1",
            "typ": "access",
            "iss": token_service.ISSUER,
            "aud": "some-other-service",
            "iat": int(time.time()),
            "exp": int(time.time()) + 300,
        }
        token = jwt.encode(payload, settings.auth_private_key, algorithm="EdDSA")

        with pytest.raises(TokenInvalid):
            token_service.decode_access_token(token)

    def test_non_access_token_type_is_rejected(self) -> None:
        from app.core.config import settings

        payload = {
            "sub": "usr_1",
            "sid": "ses_1",
            "typ": "refresh",
            "iss": token_service.ISSUER,
            "aud": token_service.AUDIENCE,
            "iat": int(time.time()),
            "exp": int(time.time()) + 300,
        }
        token = jwt.encode(payload, settings.auth_private_key, algorithm="EdDSA")

        with pytest.raises(TokenInvalid):
            token_service.decode_access_token(token)

    def test_none_algorithm_is_rejected(self) -> None:
        """The classic JWT attack: strip the signature and set alg=none."""
        payload = {
            "sub": "usr_1",
            "sid": "ses_1",
            "typ": "access",
            "iss": token_service.ISSUER,
            "aud": token_service.AUDIENCE,
            "iat": int(time.time()),
            "exp": int(time.time()) + 300,
        }
        forged = jwt.encode(payload, key="", algorithm="none")

        with pytest.raises(TokenInvalid):
            token_service.decode_access_token(forged)


class TestSecrets:
    def test_generated_secrets_are_unique_and_long(self) -> None:
        secrets = {token_service.generate_secret() for _ in range(200)}
        assert len(secrets) == 200
        assert all(len(value) >= 32 for value in secrets)

    def test_hash_is_stable_and_one_way(self) -> None:
        secret = token_service.generate_secret()
        digest = token_service.hash_secret(secret)

        assert digest == token_service.hash_secret(secret)
        assert secret not in digest
        assert len(digest) == 64  # sha256 hex


class TestKeypair:
    def test_generates_a_usable_pair(self) -> None:
        private_pem, public_pem = token_service.generate_keypair()
        assert "BEGIN PRIVATE KEY" in private_pem
        assert "BEGIN PUBLIC KEY" in public_pem

        token = jwt.encode({"a": 1}, private_pem, algorithm="EdDSA")
        assert jwt.decode(token, public_pem, algorithms=["EdDSA"]) == {"a": 1}


class TestCursorPagination:
    def test_round_trip(self) -> None:
        from app.core.database import utcnow
        from app.core.pagination import Cursor

        now = utcnow()
        encoded = Cursor(now, "usr_1").encode()
        decoded = Cursor.decode(encoded)

        assert decoded.id == "usr_1"
        assert decoded.created_at == now

    def test_malformed_cursor_is_a_validation_error(self) -> None:
        from app.core.pagination import Cursor

        with pytest.raises(ValidationFailed):
            Cursor.decode("!!!not-base64!!!")


class TestRedaction:
    def test_secrets_never_reach_the_log_output(self) -> None:
        """Enforced by a test rather than by discipline."""
        from app.core.logging import redact_processor

        event = {
            "event": "login",
            "password": "hunter2",
            "authorization": "Bearer abc.def.ghi",
            "nested": {"api_key": "sk-live-123", "safe": "keep me"},
            "items": [{"refresh_token": "secret"}],
            "user_id": "usr_1",
        }
        result = redact_processor(None, "", event)  # type: ignore[arg-type]
        serialized = str(result)

        assert "hunter2" not in serialized
        assert "abc.def.ghi" not in serialized
        assert "sk-live-123" not in serialized
        assert "secret" not in serialized
        assert result["user_id"] == "usr_1"
        assert result["nested"]["safe"] == "keep me"


class TestDeviceLabel:
    @pytest.mark.parametrize(
        ("user_agent", "expected"),
        [
            (
                "Mozilla/5.0 (Windows NT 10.0) AppleWebKit/537.36 Chrome/120.0 Safari/537.36",
                "Chrome on Windows",
            ),
            ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) Firefox/121.0", "Firefox on macOS"),
            (None, None),
        ],
    )
    def test_labels(self, user_agent: str | None, expected: str | None) -> None:
        from app.services.session_service import _device_label

        assert _device_label(user_agent) == expected


class TestSessionExpiry:
    def test_absolute_expiry_wins_over_sliding(self) -> None:
        from app.core.database import utcnow
        from app.models.auth import Session

        now = utcnow()
        session = Session(
            id="ses_1",
            user_id="usr_1",
            family_id="fam_1",
            token_hash="x",
            expires_at=now + timedelta(days=30),
            absolute_expires_at=now - timedelta(seconds=1),  # ceiling already passed
            last_seen_at=now,
        )
        assert not session.is_usable(now)

    def test_used_token_is_not_usable(self) -> None:
        from app.core.database import utcnow
        from app.models.auth import Session

        now = utcnow()
        session = Session(
            id="ses_1",
            user_id="usr_1",
            family_id="fam_1",
            token_hash="x",
            used_at=now,
            expires_at=now + timedelta(days=30),
            absolute_expires_at=now + timedelta(days=90),
            last_seen_at=now,
        )
        assert not session.is_usable(now)
