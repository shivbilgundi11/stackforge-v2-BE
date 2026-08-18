"""The request bodies handed to Razorpay.

The integration tests replace the whole client with a fake, so they assert
what *we* would call and never what goes on the wire. That gap let a real bug
ship: `fail_existing` was sent as the integer `0`, Razorpay silently ignored
it, applied its failing default, and every returning customer got
`Customer already exists for the merchant` — a 502 on the checkout button.

These tests exercise the real client with only the SDK's transport swapped
out, so the payload itself is the thing under test.
"""

from __future__ import annotations

from typing import Any

import pytest

from app.integrations.razorpay import LiveRazorpay


class _Recorder:
    """Stands in for one SDK resource, keeping the body it was handed."""

    def __init__(self, result: dict[str, Any]) -> None:
        self.result = result
        self.calls: list[dict[str, Any]] = []

    def create(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(payload)
        return self.result


@pytest.fixture
def client() -> LiveRazorpay:
    return LiveRazorpay("rzp_test_key", "secret")


async def test_fail_existing_is_the_string_zero(client: LiveRazorpay) -> None:
    """Razorpay only honours the flag as `"0"`.

    An integer `0` is ignored and the request falls back to failing on a
    duplicate — verified against the live API, where the same call differs only
    in that quote pair. The flag exists so a customer who cancelled and came
    back lands on their original record instead of splitting their invoice
    history in two, so getting the type wrong silently removed the feature.
    """
    recorder = _Recorder({"id": "cust_test"})
    client._client.customer = recorder  # type: ignore[assignment]

    customer_id = await client.create_customer(
        email="returning@example.com", name="Returning Customer", user_id="usr_1"
    )

    assert customer_id == "cust_test"
    (payload,) = recorder.calls
    assert payload["fail_existing"] == "0"
    assert isinstance(payload["fail_existing"], str), (
        "Razorpay ignores an integer here and 400s on the duplicate instead"
    )


async def test_the_customer_carries_the_account_it_belongs_to(client: LiveRazorpay) -> None:
    """Without the note, a customer found in the Razorpay dashboard cannot be
    traced back to an account without a database query nobody in support can
    run."""
    recorder = _Recorder({"id": "cust_test"})
    client._client.customer = recorder  # type: ignore[assignment]

    await client.create_customer(email="a@example.com", name="A", user_id="usr_42")

    (payload,) = recorder.calls
    assert payload["notes"] == {"user_id": "usr_42"}
    assert payload["email"] == "a@example.com"
    assert payload["name"] == "A"


async def test_the_subscription_carries_the_account_through_the_redirect(
    client: LiveRazorpay,
) -> None:
    """`notes.user_id` is what lets a webhook attribute a subscription without
    racing the browser back to the return page."""
    recorder = _Recorder({"id": "sub_test"})
    client._client.subscription = recorder  # type: ignore[assignment]

    subscription_id = await client.create_subscription(
        plan_id="plan_x",
        customer_id="cust_x",
        quantity=1,
        total_count=120,
        start_at=None,
        notes={"user_id": "usr_7"},
    )

    assert subscription_id == "sub_test"
    (payload,) = recorder.calls
    assert payload["notes"]["user_id"] == "usr_7"
    assert payload["customer_id"] == "cust_x"
    # Numeric on purpose, unlike `fail_existing`: these are counts and a unix
    # timestamp, and Razorpay takes them as numbers.
    assert isinstance(payload["quantity"], int)
    assert isinstance(payload["total_count"], int)
    # A trial is a delayed start; with none configured the field must be absent
    # rather than null, or Checkout collects its authorization amount instead
    # of the plan price.
    assert "start_at" not in payload or payload["start_at"] is not None
