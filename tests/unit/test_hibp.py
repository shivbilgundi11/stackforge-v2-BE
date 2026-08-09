"""The Have I Been Pwned check.

Two properties are deliberate and both would be easy to lose in a refactor:
the password never leaves the process, and an outage must not block signups.
"""

from __future__ import annotations

import hashlib
from typing import Any, ClassVar

import httpx
import pytest

from app.core.errors import ValidationFailed
from app.services import password_service


class _MockResponse:
    def __init__(self, text: str) -> None:
        self.text = text

    def raise_for_status(self) -> None:
        return None


class _MockClient:
    """Records the URL so the test can assert what was actually sent."""

    requested: ClassVar[list[str]] = []

    def __init__(self, *_: Any, **__: Any) -> None:
        pass

    async def __aenter__(self) -> _MockClient:
        return self

    async def __aexit__(self, *_: Any) -> None:
        return None

    async def get(self, url: str, **_: Any) -> _MockResponse:
        type(self).requested.append(url)
        return _MockResponse(type(self).body)

    body: str = ""


@pytest.fixture(autouse=True)
def _enable_hibp(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(password_service.settings, "hibp_enabled", True)
    _MockClient.requested = []


async def test_only_the_hash_prefix_leaves_the_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    password = "correct-horse-battery-staple"
    digest = hashlib.sha1(password.encode(), usedforsecurity=False).hexdigest().upper()

    _MockClient.body = ""
    monkeypatch.setattr(httpx, "AsyncClient", _MockClient)
    await password_service._is_breached(password)

    sent = _MockClient.requested[0]
    assert sent.endswith(digest[:5]), "only the five-character prefix may be sent"
    assert password not in sent
    assert digest[5:] not in sent, "the suffix must never leave the process"


async def test_breached_password_is_detected(monkeypatch: pytest.MonkeyPatch) -> None:
    password = "correct-horse-battery-staple"
    digest = hashlib.sha1(password.encode(), usedforsecurity=False).hexdigest().upper()

    _MockClient.body = f"AAAAAAAAAA:1\r\n{digest[5:]}:42\r\nBBBBBBBBBB:7"
    monkeypatch.setattr(httpx, "AsyncClient", _MockClient)

    assert await password_service._is_breached(password) is True

    with pytest.raises(ValidationFailed) as exc:
        await password_service.validate_password(password)
    assert "breach" in str(exc.value).lower()


async def test_unbreached_password_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    _MockClient.body = "AAAAAAAAAA:1\r\nBBBBBBBBBB:7"
    monkeypatch.setattr(httpx, "AsyncClient", _MockClient)

    assert await password_service._is_breached("correct-horse-battery-staple") is False
    await password_service.validate_password("correct-horse-battery-staple")


async def test_zero_count_is_not_a_breach(monkeypatch: pytest.MonkeyPatch) -> None:
    """The padded responses HIBP returns include decoy suffixes with count 0."""
    password = "correct-horse-battery-staple"
    digest = hashlib.sha1(password.encode(), usedforsecurity=False).hexdigest().upper()

    _MockClient.body = f"{digest[5:]}:0"
    monkeypatch.setattr(httpx, "AsyncClient", _MockClient)

    assert await password_service._is_breached(password) is False


async def test_outage_fails_open(monkeypatch: pytest.MonkeyPatch) -> None:
    """An HIBP outage must not stop people creating accounts."""

    class _Broken(_MockClient):
        async def get(self, url: str, **_: Any) -> _MockResponse:
            raise httpx.ConnectError("boom")

    monkeypatch.setattr(httpx, "AsyncClient", _Broken)

    assert await password_service._is_breached("correct-horse-battery-staple") is False
    await password_service.validate_password("correct-horse-battery-staple")


async def test_disabled_skips_the_network_entirely(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(password_service.settings, "hibp_enabled", False)

    class _Exploding(_MockClient):
        async def get(self, url: str, **_: Any) -> _MockResponse:
            raise AssertionError("no request should be made when HIBP is disabled")

    monkeypatch.setattr(httpx, "AsyncClient", _Exploding)
    await password_service.validate_password("correct-horse-battery-staple")
