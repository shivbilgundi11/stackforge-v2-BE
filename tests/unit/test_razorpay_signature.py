"""`verify_signature`, for real.

This is the one place in the suite that does *not* bypass verification.
Everywhere else patches it out on the sound argument that re-implementing the
HMAC to hand it back to our own verifier tests `hmac` rather than the
application — but the consequence, under the previous provider, was that the
function itself had no coverage at all and shipped broken for months. So it
gets its own file, and the file signs bodies exactly the way Razorpay does.

Razorpay's scheme is simpler than Stripe's: a bare hex HMAC-SHA256 of the raw
body, with no timestamp. That means there is no replay window — a captured
delivery stays valid forever — so idempotency, not the signature, is what makes
a replay harmless. `billing_events.id` is the thing standing there, which is
why `test_billing.py` spends so much of its length on it.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from typing import Any

import pytest

from app.core.config import settings
from app.core.errors import UpstreamError
from app.integrations import razorpay as razorpay_integration

SECRET = "whsec_test_secret_for_signature_verification"


@pytest.fixture(autouse=True)
def _configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "razorpay_webhook_secret", SECRET)


def _sign(body: bytes, *, secret: str = SECRET) -> str:
    return hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def _event() -> dict[str, Any]:
    return {
        "entity": "event",
        "event": "subscription.activated",
        "contains": ["subscription"],
        "payload": {
            "subscription": {
                "entity": {
                    "id": "sub_1",
                    "plan_id": "plan_1",
                    "customer_id": "cust_1",
                    "status": "active",
                    "quantity": 3,
                    "notes": {"user_id": "usr_1"},
                }
            }
        },
    }


def test_a_signed_body_comes_back_as_a_plain_nested_dict() -> None:
    """Every nested level has to be a real dict: the payload is written to a
    JSONB column and read back by key."""
    body = json.dumps(_event()).encode()

    parsed = razorpay_integration.verify_signature(body, _sign(body))

    assert parsed == _event()
    assert type(parsed) is dict
    assert type(parsed["payload"]["subscription"]["entity"]) is dict
    # Serialisable as-is — this is what goes into `billing_events.payload`.
    assert json.loads(json.dumps(parsed)) == _event()


def test_the_handler_can_read_what_comes_back() -> None:
    """The parsed shape reaches the code that consumes it, not just a type
    assertion. A dict that is plain but reshaped would pass the test above."""
    from app.services import billing_service

    body = json.dumps(_event()).encode()
    parsed = razorpay_integration.verify_signature(body, _sign(body))

    entity = billing_service._event_object(parsed)
    assert entity["status"] == "active"
    assert entity["quantity"] == 3
    assert entity["notes"] == {"user_id": "usr_1"}


def test_a_forged_signature_is_refused() -> None:
    """The endpoint's entire authentication. Without this it is an
    unauthenticated "make me Pro" API."""
    body = json.dumps(_event()).encode()

    with pytest.raises(UpstreamError):
        razorpay_integration.verify_signature(body, _sign(body, secret="whsec_wrong"))


def test_a_body_changed_after_signing_is_refused() -> None:
    """Signed one thing, sent another — the attack the signature exists for."""
    signature = _sign(json.dumps(_event()).encode())
    tampered = json.dumps({**_event(), "event": "subscription.cancelled"}).encode()

    with pytest.raises(UpstreamError):
        razorpay_integration.verify_signature(tampered, signature)


def test_a_missing_signature_is_refused() -> None:
    with pytest.raises(UpstreamError):
        razorpay_integration.verify_signature(b"{}", None)


def test_an_unset_secret_rejects_everything(monkeypatch: pytest.MonkeyPatch) -> None:
    """Deliberately noisier than accepting. A silently-unverified endpoint in
    production is the worst outcome available here."""
    monkeypatch.setattr(settings, "razorpay_webhook_secret", "")
    body = json.dumps(_event()).encode()

    with pytest.raises(UpstreamError):
        razorpay_integration.verify_signature(body, _sign(body))


def test_a_malformed_body_is_refused() -> None:
    """Correctly signed garbage is still garbage."""
    body = b"{not json"

    with pytest.raises(UpstreamError):
        razorpay_integration.verify_signature(body, _sign(body))


def test_a_signed_non_object_is_refused() -> None:
    """Valid JSON that is not an event. `[1, 2]` parses fine and would reach
    the handler as something with no `.get`."""
    body = b"[1, 2]"

    with pytest.raises(UpstreamError):
        razorpay_integration.verify_signature(body, _sign(body))


def test_the_comparison_is_constant_time() -> None:
    """Not observable from a test, so this asserts the call rather than the
    timing: a byte-by-byte compare leaks how much of a forged signature was
    right, which is enough to construct one."""
    import inspect

    source = inspect.getsource(razorpay_integration.verify_signature)
    assert "compare_digest" in source
