"""`verify_signature`, against the real SDK.

This is the one place in the suite that does *not* bypass verification.
Everywhere else patches it out on the sound argument that re-implementing
Stripe's HMAC to hand it back to Stripe's own verifier tests the SDK rather
than the application — but the consequence was that the function itself had no
coverage at all, and it shipped broken.

The bug it shipped with: `stripe.Event` subclassed `dict` in stripe-python 13
and stopped in 15, so `dict(event)` quietly changed from "copy the mapping" to
"iterate as pairs of pairs" and raised `KeyError: 0` on *every* real delivery.
Nothing caught it, because no test ever passed a genuinely signed body and no
environment had a webhook secret configured.

So this file signs for real. The secret is a local constant, the payload is
built here, and the assertion is that what comes back out is a plain nested
dict — the shape `stripe_events.payload` has to store and the handlers have to
read.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from typing import Any

import pytest

from app.core.config import settings
from app.core.errors import UpstreamError
from app.integrations import stripe as stripe_integration

SECRET = "whsec_test_secret_for_signature_verification"


@pytest.fixture(autouse=True)
def _configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "stripe_webhook_secret", SECRET)


def _sign(body: bytes, *, secret: str = SECRET, timestamp: int | None = None) -> str:
    """Stripe's scheme: `t=<ts>,v1=<hex hmac-sha256 of "<ts>.<body>">`."""
    ts = timestamp if timestamp is not None else int(time.time())
    mac = hmac.new(secret.encode(), f"{ts}.".encode() + body, hashlib.sha256)
    return f"t={ts},v1={mac.hexdigest()}"


def _event() -> dict[str, Any]:
    return {
        "id": "evt_signature_1",
        "type": "customer.subscription.updated",
        "data": {
            "object": {
                "id": "sub_1",
                "customer": "cus_1",
                "status": "active",
                "items": {"data": [{"price": {"id": "price_1"}, "quantity": 3}]},
            }
        },
    }


def test_a_signed_body_comes_back_as_a_plain_nested_dict() -> None:
    """The regression. Every nested level has to be a real dict, because the
    payload is written to a JSONB column and read back by key."""
    body = json.dumps(_event()).encode()

    parsed = stripe_integration.verify_signature(body, _sign(body))

    assert parsed == _event()
    assert type(parsed) is dict
    assert type(parsed["data"]) is dict
    assert type(parsed["data"]["object"]) is dict
    assert type(parsed["data"]["object"]["items"]["data"][0]) is dict
    # Serialisable as-is — this is what goes into `stripe_events.payload`, and
    # a StripeObject in there raises at the driver rather than here.
    assert json.loads(json.dumps(parsed)) == _event()


def test_the_handlers_can_read_what_comes_back() -> None:
    """The parsed shape reaches the code that consumes it, not just a type
    assertion. A dict that is plain but reshaped would pass the test above."""
    from app.services import billing_service

    body = json.dumps(_event()).encode()
    parsed = stripe_integration.verify_signature(body, _sign(body))

    obj = parsed["data"]["object"]
    item = obj["items"]["data"][0]
    assert obj["status"] == "active"
    assert item["quantity"] == 3
    assert billing_service.plan_for_price(item["price"]["id"]) is None


def test_a_forged_signature_is_refused() -> None:
    """The endpoint's entire authentication. Without this it is an
    unauthenticated "make me Pro" API."""
    body = json.dumps(_event()).encode()

    with pytest.raises(UpstreamError):
        stripe_integration.verify_signature(body, _sign(body, secret="whsec_wrong"))


def test_a_body_changed_after_signing_is_refused() -> None:
    """Signed one thing, sent another — the attack the signature exists for."""
    signature = _sign(json.dumps(_event()).encode())
    tampered = json.dumps({**_event(), "type": "invoice.paid"}).encode()

    with pytest.raises(UpstreamError):
        stripe_integration.verify_signature(tampered, signature)


def test_an_old_timestamp_is_refused() -> None:
    """Replay protection. Stripe's default tolerance is five minutes."""
    body = json.dumps(_event()).encode()
    stale = _sign(body, timestamp=int(time.time()) - 3600)

    with pytest.raises(UpstreamError):
        stripe_integration.verify_signature(body, stale)


def test_a_missing_signature_is_refused() -> None:
    with pytest.raises(UpstreamError):
        stripe_integration.verify_signature(b"{}", None)


def test_an_unset_secret_rejects_everything(monkeypatch: pytest.MonkeyPatch) -> None:
    """Deliberately noisier than accepting. A silently-unverified endpoint in
    production is the worst outcome available here."""
    monkeypatch.setattr(settings, "stripe_webhook_secret", "")
    body = json.dumps(_event()).encode()

    with pytest.raises(UpstreamError):
        stripe_integration.verify_signature(body, _sign(body))


def test_a_malformed_body_is_refused() -> None:
    """Correctly signed garbage is still garbage."""
    body = b"{not json"

    with pytest.raises(UpstreamError):
        stripe_integration.verify_signature(body, _sign(body))
