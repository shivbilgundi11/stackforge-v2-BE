"""End-to-end auth flows through the app.

Every test here asserts behaviour that would be a security hole if it changed,
not that an endpoint returns 200.
"""

from __future__ import annotations

import re

import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.integrations.email import ConsoleSender
from app.models.auth import AuthEvent, AuthEventType, Session
from app.models.user import User
from tests.conftest import sign_in

PASSWORD = "Correct-horse-battery-staple-42!"
BASE = "/api/v1/auth"


def token_from(sender: ConsoleSender, needle: str) -> str:
    """Pull the one-time token out of the most recent matching email."""
    for message in reversed(sender.outbox):
        if needle in message.subject.lower():
            match = re.search(r"token=([A-Za-z0-9_\-]+)", message.text)
            if match:
                return match.group(1)
    raise AssertionError(f"no email matching {needle!r} in {[m.subject for m in sender.outbox]}")


async def register(anon_client: AsyncClient, email: str = "ada@example.com") -> None:
    response = await anon_client.post(
        f"{BASE}/register",
        json={"email": email, "password": PASSWORD, "name": "Ada Lovelace"},
    )
    assert response.status_code == 202


class TestRegistration:
    async def test_creates_a_user_and_sends_verification(
        self, anon_client: AsyncClient, db: AsyncSession, outbox: ConsoleSender
    ) -> None:
        await register(anon_client)

        user = (await db.execute(select(User).where(User.email == "ada@example.com"))).scalar_one()
        assert user.name == "Ada Lovelace"
        assert user.email_verified_at is None
        assert user.password_hash is not None
        assert PASSWORD not in (user.password_hash or "")

        assert any("verify" in m.subject.lower() for m in outbox.outbox)

    async def test_duplicate_email_is_indistinguishable(
        self, anon_client: AsyncClient, outbox: ConsoleSender
    ) -> None:
        """The single most important enumeration test.

        Status, body, and shape must be identical for a new and an existing
        address. The existing account holder is emailed instead.
        """
        first = await anon_client.post(
            f"{BASE}/register",
            json={"email": "ada@example.com", "password": PASSWORD, "name": "Ada"},
        )
        second = await anon_client.post(
            f"{BASE}/register",
            json={"email": "ada@example.com", "password": PASSWORD, "name": "Someone Else"},
        )

        assert first.status_code == second.status_code == 202
        assert first.json()["data"] == second.json()["data"]

        assert any("someone tried" in m.subject.lower() for m in outbox.outbox), (
            "the real account holder must be told someone tried to register"
        )

    async def test_duplicate_registration_does_not_overwrite_the_account(
        self, anon_client: AsyncClient, db: AsyncSession
    ) -> None:
        await register(anon_client)
        await anon_client.post(
            f"{BASE}/register",
            json={"email": "ada@example.com", "password": "a-totally-different-one", "name": "Mal"},
        )

        user = (await db.execute(select(User).where(User.email == "ada@example.com"))).scalar_one()
        assert user.name == "Ada Lovelace"

    async def test_weak_password_is_rejected_with_a_field_error(
        self, anon_client: AsyncClient
    ) -> None:
        response = await anon_client.post(
            f"{BASE}/register",
            json={"email": "ada@example.com", "password": "password123", "name": "Ada"},
        )
        assert response.status_code == 422
        body = response.json()
        assert body["error"]["code"] == "VALIDATION_ERROR"
        assert body["error"]["details"]["fields"][0]["path"] == "password"


class TestLogin:
    async def test_success_returns_a_token_and_sets_the_cookie(
        self, anon_client: AsyncClient
    ) -> None:
        await register(anon_client)
        response = await anon_client.post(
            f"{BASE}/login", json={"email": "ada@example.com", "password": PASSWORD}
        )

        assert response.status_code == 200
        data = response.json()["data"]
        assert data["tokens"]["access_token"]
        assert data["tokens"]["expires_in"] == settings.access_token_ttl_seconds
        assert data["user"]["email"] == "ada@example.com"

        cookie = response.cookies.get(settings.refresh_cookie_name)
        assert cookie, "the refresh token must be delivered as a cookie"

        raw = response.headers.get("set-cookie", "")
        assert "HttpOnly" in raw
        assert "Path=/api/v1/auth" in raw, "the cookie must not be sent to every endpoint"
        assert "samesite=lax" in raw.lower()

    async def test_refresh_token_is_never_in_the_body(self, anon_client: AsyncClient) -> None:
        await register(anon_client)
        response = await anon_client.post(
            f"{BASE}/login", json={"email": "ada@example.com", "password": PASSWORD}
        )
        cookie = response.cookies.get(settings.refresh_cookie_name)
        assert cookie and cookie not in response.text

    @pytest.mark.parametrize(
        ("email", "password"),
        [
            ("ada@example.com", "wrong-password-entirely"),
            ("nobody@example.com", PASSWORD),
        ],
    )
    async def test_unknown_user_and_wrong_password_are_indistinguishable(
        self, anon_client: AsyncClient, email: str, password: str
    ) -> None:
        await register(anon_client)
        response = await anon_client.post(
            f"{BASE}/login", json={"email": email, "password": password}
        )

        assert response.status_code == 401
        assert response.json()["error"]["code"] == "INVALID_CREDENTIALS"

    async def test_lockout_after_repeated_failures(
        self, anon_client: AsyncClient, db: AsyncSession
    ) -> None:
        await register(anon_client)
        for _ in range(settings.max_failed_logins):
            await anon_client.post(
                f"{BASE}/login", json={"email": "ada@example.com", "password": "nope-nope-nope"}
            )

        locked = await anon_client.post(
            f"{BASE}/login", json={"email": "ada@example.com", "password": PASSWORD}
        )
        assert locked.status_code == 423
        assert locked.json()["error"]["code"] == "ACCOUNT_LOCKED"
        assert locked.json()["error"]["details"]["locked_until"]

    async def test_successful_login_clears_the_failure_counter(
        self, anon_client: AsyncClient, db: AsyncSession
    ) -> None:
        await register(anon_client)
        await anon_client.post(
            f"{BASE}/login", json={"email": "ada@example.com", "password": "nope-nope-nope"}
        )
        await anon_client.post(
            f"{BASE}/login", json={"email": "ada@example.com", "password": PASSWORD}
        )

        user = (await db.execute(select(User).where(User.email == "ada@example.com"))).scalar_one()
        assert user.failed_login_count == 0
        assert user.last_login_at is not None


class TestRefreshRotation:
    async def test_rotation_issues_a_new_token_and_kills_the_old(
        self, anon_client: AsyncClient
    ) -> None:
        await register(anon_client)
        await anon_client.post(
            f"{BASE}/login", json={"email": "ada@example.com", "password": PASSWORD}
        )
        first = anon_client.cookies.get(settings.refresh_cookie_name)

        response = await anon_client.post(f"{BASE}/refresh")
        assert response.status_code == 200

        second = anon_client.cookies.get(settings.refresh_cookie_name)
        assert second and second != first, "the refresh token must rotate"

    async def test_reuse_revokes_the_whole_family_and_warns_the_user(
        self, anon_client: AsyncClient, db: AsyncSession, outbox: ConsoleSender
    ) -> None:
        """The property that makes rotation worth its complexity.

        A token already marked used means two parties hold tokens from this
        family. The server cannot tell which is the thief, so both lose access.
        """
        await register(anon_client)
        await anon_client.post(
            f"{BASE}/login", json={"email": "ada@example.com", "password": PASSWORD}
        )
        stolen = anon_client.cookies.get(settings.refresh_cookie_name)
        assert stolen

        # The legitimate anon_client rotates.
        await anon_client.post(f"{BASE}/refresh")

        # The thief presents the copy they took earlier.
        anon_client.cookies.set(settings.refresh_cookie_name, stolen, path="/api/v1/auth")
        replay = await anon_client.post(f"{BASE}/refresh")
        assert replay.status_code == 401

        sessions = (await db.execute(select(Session))).scalars().all()
        assert sessions, "sanity: sessions exist"
        assert all(s.revoked_at is not None for s in sessions), (
            "every session in the family must be revoked, not just the replayed one"
        )
        assert all(s.revoked_reason == "reuse" for s in sessions)

        events = (
            (
                await db.execute(
                    select(AuthEvent).where(AuthEvent.event == AuthEventType.REFRESH_REUSE_DETECTED)
                )
            )
            .scalars()
            .all()
        )
        assert events, "reuse must be recorded in the audit trail"
        assert any("unusual activity" in m.subject.lower() for m in outbox.outbox)

    async def test_refresh_without_a_cookie_is_401(self, anon_client: AsyncClient) -> None:
        response = await anon_client.post(f"{BASE}/refresh")
        assert response.status_code == 401
        assert response.json()["error"]["code"] == "UNAUTHENTICATED"


class TestSessionLifecycle:
    async def test_logout_revokes_and_clears(
        self, anon_client: AsyncClient, db: AsyncSession
    ) -> None:
        await register(anon_client)
        await anon_client.post(
            f"{BASE}/login", json={"email": "ada@example.com", "password": PASSWORD}
        )

        response = await anon_client.post(f"{BASE}/logout")
        assert response.status_code == 200

        session = (await db.execute(select(Session))).scalars().first()
        assert session is not None and session.revoked_at is not None

        assert (await anon_client.post(f"{BASE}/refresh")).status_code == 401

    async def test_logout_succeeds_without_a_session(self, anon_client: AsyncClient) -> None:
        """Otherwise a anon_client holding a dead cookie can never clear it."""
        assert (await anon_client.post(f"{BASE}/logout")).status_code == 200

    async def test_revoked_session_kills_the_access_token_immediately(
        self, anon_client: AsyncClient, db: AsyncSession
    ) -> None:
        """An access token is stateless and valid for 15 minutes. Without the
        session-liveness check, 'revoke this device' would be a lie for that
        long."""
        await register(anon_client)
        login = await anon_client.post(
            f"{BASE}/login", json={"email": "ada@example.com", "password": PASSWORD}
        )
        token = login.json()["data"]["tokens"]["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        assert (await anon_client.get(f"{BASE}/me", headers=headers)).status_code == 200

        await anon_client.post(f"{BASE}/logout")

        after = await anon_client.get(f"{BASE}/me", headers=headers)
        assert after.status_code == 401
        assert after.json()["error"]["code"] == "TOKEN_INVALID"

    async def test_sessions_can_be_listed_and_revoked(self, anon_client: AsyncClient) -> None:
        await register(anon_client)
        login = await anon_client.post(
            f"{BASE}/login", json={"email": "ada@example.com", "password": PASSWORD}
        )
        headers = {"Authorization": f"Bearer {login.json()['data']['tokens']['access_token']}"}

        listing = await anon_client.get(f"{BASE}/sessions", headers=headers)
        assert listing.status_code == 200
        sessions = listing.json()["data"]["sessions"]
        assert len(sessions) == 1
        assert sessions[0]["current"] is True

        revoked = await anon_client.delete(f"{BASE}/sessions/{sessions[0]['id']}", headers=headers)
        assert revoked.status_code == 200

    async def test_revoking_another_users_session_is_404_not_403(
        self, anon_client: AsyncClient
    ) -> None:
        """A 403 would confirm the session exists."""
        await register(anon_client, "ada@example.com")
        await register(anon_client, "grace@example.com")

        login = await anon_client.post(
            f"{BASE}/login", json={"email": "grace@example.com", "password": PASSWORD}
        )
        headers = {"Authorization": f"Bearer {login.json()['data']['tokens']['access_token']}"}

        response = await anon_client.delete(f"{BASE}/sessions/ses_someone_else", headers=headers)
        assert response.status_code == 404


class TestVerification:
    async def test_verify_marks_the_account(
        self, anon_client: AsyncClient, db: AsyncSession, outbox: ConsoleSender
    ) -> None:
        await register(anon_client)
        token = token_from(outbox, "verify")

        response = await anon_client.post(f"{BASE}/verify-email", json={"token": token})
        assert response.status_code == 200
        assert response.json()["data"]["email_verified"] is True

    async def test_token_is_single_use(
        self, anon_client: AsyncClient, outbox: ConsoleSender
    ) -> None:
        await register(anon_client)
        token = token_from(outbox, "verify")

        assert (
            await anon_client.post(f"{BASE}/verify-email", json={"token": token})
        ).status_code == 200
        second = await anon_client.post(f"{BASE}/verify-email", json={"token": token})
        assert second.status_code == 401
        assert second.json()["error"]["code"] == "TOKEN_INVALID"

    async def test_garbage_token_is_rejected(self, anon_client: AsyncClient) -> None:
        response = await anon_client.post(f"{BASE}/verify-email", json={"token": "x" * 40})
        assert response.status_code == 401


class TestPasswordReset:
    async def test_forgot_password_is_the_same_for_unknown_addresses(
        self, anon_client: AsyncClient
    ) -> None:
        await register(anon_client)

        known = await anon_client.post(f"{BASE}/forgot-password", json={"email": "ada@example.com"})
        unknown = await anon_client.post(
            f"{BASE}/forgot-password", json={"email": "nope@example.com"}
        )

        assert known.status_code == unknown.status_code == 202
        assert known.json()["data"] == unknown.json()["data"]

    async def test_reset_changes_the_password_and_kills_every_session(
        self, anon_client: AsyncClient, db: AsyncSession, outbox: ConsoleSender
    ) -> None:
        await register(anon_client)
        await anon_client.post(
            f"{BASE}/login", json={"email": "ada@example.com", "password": PASSWORD}
        )
        await anon_client.post(f"{BASE}/forgot-password", json={"email": "ada@example.com"})

        token = token_from(outbox, "reset")
        new_password = "An-entirely-different-passphrase1!"

        response = await anon_client.post(
            f"{BASE}/reset-password", json={"token": token, "password": new_password}
        )
        assert response.status_code == 200

        sessions = (await db.execute(select(Session))).scalars().all()
        assert all(s.revoked_at is not None for s in sessions), (
            "a reset is the remedy for a compromise, so it must end every session"
        )

        assert (
            await anon_client.post(
                f"{BASE}/login", json={"email": "ada@example.com", "password": PASSWORD}
            )
        ).status_code == 401
        assert (
            await anon_client.post(
                f"{BASE}/login", json={"email": "ada@example.com", "password": new_password}
            )
        ).status_code == 200

    async def test_reset_token_is_single_use(
        self, anon_client: AsyncClient, outbox: ConsoleSender
    ) -> None:
        await register(anon_client)
        await anon_client.post(f"{BASE}/forgot-password", json={"email": "ada@example.com"})
        token = token_from(outbox, "reset")

        await anon_client.post(
            f"{BASE}/reset-password", json={"token": token, "password": "First-new-passphrase1!"}
        )
        second = await anon_client.post(
            f"{BASE}/reset-password", json={"token": token, "password": "Second-new-passphrase1!"}
        )
        assert second.status_code == 401


class TestPasswordChange:
    async def test_change_keeps_this_session_and_ends_the_others(
        self, anon_client: AsyncClient, db: AsyncSession
    ) -> None:
        await register(anon_client)
        # Two devices.
        first = await anon_client.post(
            f"{BASE}/login", json={"email": "ada@example.com", "password": PASSWORD}
        )
        second = await anon_client.post(
            f"{BASE}/login", json={"email": "ada@example.com", "password": PASSWORD}
        )
        headers = {"Authorization": f"Bearer {second.json()['data']['tokens']['access_token']}"}

        response = await anon_client.patch(
            f"{BASE}/password",
            json={"current_password": PASSWORD, "new_password": "A-brand-new-passphrase1!"},
            headers=headers,
        )
        assert response.status_code == 200

        # The device that made the change is still usable.
        assert (await anon_client.get(f"{BASE}/me", headers=headers)).status_code == 200

        other = {"Authorization": f"Bearer {first.json()['data']['tokens']['access_token']}"}
        assert (await anon_client.get(f"{BASE}/me", headers=other)).status_code == 401

    async def test_wrong_current_password_is_rejected(self, anon_client: AsyncClient) -> None:
        await register(anon_client)
        login = await anon_client.post(
            f"{BASE}/login", json={"email": "ada@example.com", "password": PASSWORD}
        )
        headers = {"Authorization": f"Bearer {login.json()['data']['tokens']['access_token']}"}

        response = await anon_client.patch(
            f"{BASE}/password",
            json={"current_password": "not-it-at-all", "new_password": "a-brand-new-passphrase"},
            headers=headers,
        )
        assert response.status_code == 401


class TestIdentityProbe:
    """The one route left that answers without a session.

    `POST /anonymous` and `POST /claim` used to live here too — a visitor was
    issued a session cookie, and the work they did with it moved onto the
    account they later created. Both endpoints are gone with the tier.
    """

    async def test_identity_works_without_credentials(self, anon_client: AsyncClient) -> None:
        response = await anon_client.get(f"{BASE}/identity")
        assert response.status_code == 200
        assert response.json()["data"]["authenticated"] is False
        assert response.json()["data"]["plan"] is None

    async def test_identity_names_the_caller_once_there_is_one(
        self, anon_client: AsyncClient, db: AsyncSession
    ) -> None:
        await sign_in(anon_client, db)

        data = (await anon_client.get(f"{BASE}/identity")).json()["data"]
        assert data["authenticated"] is True
        assert data["user"]["email"] == "ada@example.com"
        assert data["plan"] == "free"


class TestAuditTrail:
    async def test_events_are_recorded_for_the_whole_lifecycle(
        self, anon_client: AsyncClient, db: AsyncSession, outbox: ConsoleSender
    ) -> None:
        await register(anon_client)
        token = token_from(outbox, "verify")
        await anon_client.post(f"{BASE}/verify-email", json={"token": token})
        await anon_client.post(
            f"{BASE}/login", json={"email": "ada@example.com", "password": "wrong-one-here"}
        )
        await anon_client.post(
            f"{BASE}/login", json={"email": "ada@example.com", "password": PASSWORD}
        )
        await anon_client.post(f"{BASE}/refresh")
        await anon_client.post(f"{BASE}/logout")

        recorded = {event for (event,) in await db.execute(select(AuthEvent.event))}
        for expected in (
            AuthEventType.REGISTER,
            AuthEventType.EMAIL_VERIFIED,
            AuthEventType.LOGIN_FAILED,
            AuthEventType.LOGIN,
            AuthEventType.REFRESH,
            AuthEventType.LOGOUT,
        ):
            assert expected in recorded, f"{expected.value} was not recorded"


class TestEnvelopeAndErrors:
    async def test_every_response_carries_the_envelope_and_a_request_id(
        self, anon_client: AsyncClient
    ) -> None:
        response = await anon_client.get("/health")
        body = response.json()

        assert body["success"] is True
        assert body["meta"]["request_id"].startswith("req_")
        assert response.headers["X-Request-ID"] == body["meta"]["request_id"]

    async def test_unknown_route_is_a_shaped_404(self, anon_client: AsyncClient) -> None:
        response = await anon_client.get("/api/v1/nope")
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "NOT_FOUND"

    async def test_protected_route_without_a_token_is_401(self, anon_client: AsyncClient) -> None:
        response = await anon_client.get(f"{BASE}/me")
        assert response.status_code == 401
        assert response.json()["error"]["code"] == "UNAUTHENTICATED"

    async def test_security_headers_are_present(self, anon_client: AsyncClient) -> None:
        response = await anon_client.get("/health")
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        assert response.headers["X-Frame-Options"] == "DENY"
        assert "Content-Security-Policy" in response.headers

    async def test_client_supplied_request_id_is_sanitised(self, anon_client: AsyncClient) -> None:
        """Echoing an arbitrary anon_client string into logs and headers is a
        log-injection vector."""
        response = await anon_client.get("/health", headers={"X-Request-ID": "bad\ninjected: yes"})
        assert "\n" not in response.headers["X-Request-ID"]
        assert response.headers["X-Request-ID"].startswith("req_")
