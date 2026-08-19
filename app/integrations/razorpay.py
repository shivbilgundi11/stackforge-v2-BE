"""The Razorpay boundary.

Every call to Razorpay goes through this module, for the same reason every
email goes through `integrations.email`: the rest of the app should not learn a
vendor's object model, and a test should not need a network.

Four things it provides:

  * a client that is `None` when the keys are unset, so `billing_service` can
    answer "checkout unavailable" rather than raise in local development and CI;
  * `verify_signature`, the only thing standing between the webhook endpoint
    and an unauthenticated "make me Pro" API;
  * a `set_client` seam, so the test suite substitutes a fake and asserts on
    the arguments a subscription was created with;
  * `to_thread` wrapping, because the Razorpay SDK is synchronous `requests`
    and calling it directly would block the event loop for the duration of a
    network round trip.

## How this differs from the Stripe boundary it replaced

**There is no checkout session.** A Razorpay subscription is created server
side, and the browser authorizes a mandate against its *id* using Razorpay's
Checkout script. So the subscription row exists before the customer has paid,
in `created` state, and nothing here returns a URL to redirect to.

**The hosted page is not used, and cannot be.** Subscriptions do carry a
`short_url`, and it looked like the obvious redirect target. It is not one: it
does not exist for a subscription created against a `customer_id`, and the
Create Subscription API takes no `callback_url`, so a customer who authorizes
there is simply stranded on Razorpay with the app none the wiser. Checkout
accepts the callback; the hosted page never can (D-52).

**There is no billing portal.** Razorpay has no equivalent, so invoices and
cancellation are served in-app from the API calls below.

**A trial is a delayed start, not a trial flag.** `start_at` in the future
means the first charge happens then; the mandate is still authorized up front.
Razorpay has no card-free trial, which is why D-42's no-card property does not
survive this migration (see D-50).

**Amounts are in paise.** Razorpay's minor unit for INR, the same shape as
Stripe's cents — `plans.py` stores minor units and nothing here converts.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
from typing import Any, Protocol

import razorpay
from razorpay.errors import BadRequestError, GatewayError, ServerError, SignatureVerificationError

from app.core.config import settings
from app.core.errors import UpstreamError
from app.core.logging import get_logger

logger = get_logger("razorpay")

#: Every SDK exception that means "the provider said no" rather than "this
#: process is broken". Caught narrowly so a genuine bug here still raises.
_SDK_ERRORS = (BadRequestError, GatewayError, ServerError)


class RazorpayClient(Protocol):
    """Only what this app calls.

    Narrow on purpose: a Protocol the size of the real SDK would make the test
    fake a maintenance burden, and the surface below is the whole of what
    billing needs.
    """

    async def create_customer(self, *, email: str, name: str, user_id: str) -> str: ...

    async def create_subscription(
        self,
        *,
        plan_id: str,
        customer_id: str,
        quantity: int,
        total_count: int,
        start_at: int | None,
        notes: dict[str, str],
    ) -> str: ...

    async def fetch_subscription(self, *, subscription_id: str) -> dict[str, Any]: ...

    async def cancel_subscription(
        self, *, subscription_id: str, at_cycle_end: bool
    ) -> dict[str, Any]: ...

    async def cancel_scheduled_changes(self, *, subscription_id: str) -> dict[str, Any]: ...

    async def update_subscription_quantity(
        self, *, subscription_id: str, quantity: int
    ) -> dict[str, Any]: ...

    async def list_invoices(self, *, subscription_id: str, limit: int) -> list[dict[str, Any]]: ...


class LiveRazorpay:
    """The real thing.

    Every method translates an SDK error into `UpstreamError`. A payment
    processor being down is a 502, not a 500 — the distinction matters because
    one is our bug and the other is not, and the alert routing differs.

    Every method also hops to a worker thread. The SDK is `requests` under the
    hood, and a synchronous HTTP call on the event loop stalls every other
    request in the process for as long as Razorpay takes to answer.
    """

    def __init__(self, key_id: str, key_secret: str) -> None:
        self._client = razorpay.Client(auth=(key_id, key_secret))
        self._client.set_app_details({"title": "StackForge", "version": "1.0"})

    async def _call(self, action: str, fn: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            return await asyncio.to_thread(fn, *args, **kwargs)
        except _SDK_ERRORS as exc:
            raise _upstream(action, exc) from exc

    async def create_customer(self, *, email: str, name: str, user_id: str) -> str:
        # `fail_existing="0"`: Razorpay 400s on a duplicate email by default,
        # and a customer who cancelled and came back six months later must land
        # on the same record or their invoice history splits in two.
        #
        # **The string, not the integer.** Razorpay only honours the flag when
        # it arrives as `"0"`; an integer `0` is ignored and the request falls
        # back to the failing default, which comes back as
        # `Customer already exists for the merchant`. Verified against the live
        # API — the same call differs only in that quote pair.
        #
        # The bug hid because `_ensure_customer` short-circuits whenever the
        # row already holds a customer id, so this path only runs for an
        # account that has never had one. That is exactly the returning
        # customer this flag exists to serve, so the feature had never worked.
        customer = await self._call(
            "customer creation",
            self._client.customer.create,
            {
                "name": name,
                "email": email,
                "fail_existing": "0",
                # The link back. Without it, a customer found in the Razorpay
                # dashboard cannot be traced to an account here without a
                # database query nobody in support can run.
                "notes": {"user_id": user_id},
            },
        )
        return str(customer["id"])

    async def create_subscription(
        self,
        *,
        plan_id: str,
        customer_id: str,
        quantity: int,
        total_count: int,
        start_at: int | None,
        notes: dict[str, str],
    ) -> str:
        """Create a subscription and return its id.

        The id is what Checkout opens against in the browser. The subscription
        exists before anyone pays, which is why `billing_service` treats
        `subscription.activated` rather than the return redirect as the moment
        a plan is granted.

        **The `short_url` on the response is deliberately ignored.** It is
        Razorpay's hosted authorization page, and it is unusable to us twice
        over: it does not exist at all for a subscription created against a
        `customer_id`, and even without one it takes the mandate and then has
        nowhere to send the customer — the Create Subscription API accepts no
        `callback_url`, so the hosted page is a terminal state. Checkout takes
        the callback instead, which is why authorization happens in the browser
        (D-52).
        """
        params: dict[str, Any] = {
            "plan_id": plan_id,
            "customer_id": customer_id,
            "quantity": quantity,
            # Razorpay requires a finite number of billing cycles. This is the
            # ceiling, not a commitment — cancelling ends it earlier, and the
            # count is high enough (10 years monthly) that nobody reaches it.
            "total_count": total_count,
            "customer_notify": 1,
            "notes": notes,
        }
        if start_at is not None:
            # A trial: the mandate is authorized now, the first charge waits.
            params["start_at"] = start_at

        subscription = await self._call(
            "subscription creation", self._client.subscription.create, params
        )
        return str(subscription["id"])

    async def fetch_subscription(self, *, subscription_id: str) -> dict[str, Any]:
        result = await self._call(
            "subscription fetch", self._client.subscription.fetch, subscription_id
        )
        return dict(result)

    async def cancel_subscription(
        self, *, subscription_id: str, at_cycle_end: bool
    ) -> dict[str, Any]:
        """Cancel now or at the end of the paid period.

        Never an immediate cutoff by default: the cycle is paid for and stays.
        """
        result = await self._call(
            "subscription cancellation",
            self._client.subscription.cancel,
            subscription_id,
            {"cancel_at_cycle_end": 1 if at_cycle_end else 0},
        )
        return dict(result)

    async def cancel_scheduled_changes(self, *, subscription_id: str) -> dict[str, Any]:
        """Undo a pending change — a scheduled cancellation or a seat change.

        Razorpay has no "un-cancel" flag. Both a cancel-at-cycle-end and a
        seat change scheduled for cycle end are *pending updates* on the
        subscription, and this is the single call that drops them. It is why
        undo is a separate method rather than the cancel call with the flag
        inverted.
        """
        result = await self._call(
            "scheduled change cancellation",
            self._client.subscription.cancel_scheduled_changes,
            subscription_id,
        )
        return dict(result)

    async def update_subscription_quantity(
        self, *, subscription_id: str, quantity: int
    ) -> dict[str, Any]:
        """Seat change (M21).

        `schedule_change_at: now` would reprice mid-cycle, but Razorpay does
        not prorate — the customer would be charged a whole cycle again. The
        change lands at the next cycle instead, and the UI says so (D-51).
        """
        result = await self._call(
            "seat update",
            self._client.subscription.edit,
            subscription_id,
            {"quantity": quantity, "schedule_change_at": "cycle_end"},
        )
        return dict(result)

    async def list_invoices(self, *, subscription_id: str, limit: int) -> list[dict[str, Any]]:
        """Live from Razorpay. A subscription with no invoices yet is an empty
        list rather than an error — every account is in that state until the
        first charge."""
        page = await self._call(
            "invoice list",
            self._client.invoice.all,
            {"subscription_id": subscription_id, "count": limit},
        )
        return [dict(item) for item in page.get("items", [])]


def _upstream(action: str, exc: Exception) -> UpstreamError:
    logger.error("razorpay.call_failed", action=action, error=str(exc))
    return UpstreamError(f"The payment provider rejected the {action} request.")


_client: RazorpayClient | None = None
_resolved = False


def get_client() -> RazorpayClient | None:
    """The client, or `None` when Razorpay is not configured.

    `None` is a supported state, not a failure. Local development, CI, and
    every test run happen without keys, and billing has to degrade to
    "checkout unavailable" rather than break the app that surrounds it.
    """
    global _client, _resolved
    if not _resolved:
        _client = (
            LiveRazorpay(settings.razorpay_key_id, settings.razorpay_key_secret)
            if settings.razorpay_enabled and settings.razorpay_key_id
            else None
        )
        _resolved = True
    return _client


def set_client(client: RazorpayClient | None) -> None:
    """Test seam."""
    global _client, _resolved
    _client = client
    _resolved = True


def reset_client() -> None:
    """Forget the memoised client, so a settings change is picked up. Used by
    the CLI, which patches keys in before building one."""
    global _client, _resolved
    _client = None
    _resolved = False


def verify_signature(payload: bytes, signature: str | None) -> dict[str, Any]:
    """Parse a webhook body, or raise.

    Never parses an unverified body — not even to log what it claimed to be.
    The signature is the only thing distinguishing a Razorpay delivery from
    anyone on the internet POSTing `subscription.activated` with a plan of
    their choosing, and a handler that reads the payload before checking has
    already lost.

    An unset webhook secret rejects everything. That is deliberately noisier
    than accepting: a silently-unverified endpoint in production is the worst
    outcome available here.

    Razorpay's scheme is simpler than Stripe's — a bare hex HMAC-SHA256 of the
    raw body, with no timestamp — so there is no replay window to check. The
    event id is what makes a redelivery harmless, and that is enforced by the
    `billing_events` primary key rather than here.
    """
    if not settings.razorpay_webhook_secret:
        logger.error("razorpay.webhook_secret_missing")
        raise UpstreamError("Webhook verification is not configured.")

    if not signature:
        raise UpstreamError("Missing Razorpay signature.")

    secrets = (
        settings.razorpay_webhook_secret,
        settings.razorpay_webhook_previous_secret,
    )
    # Constant time: a byte-by-byte comparison leaks how much of a forged
    # signature was right, which is enough to construct one.
    matches = [
        hmac.compare_digest(
            hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest(), signature
        )
        for secret in secrets
        if secret
    ]
    if not any(matches):
        logger.warning("razorpay.webhook_bad_signature")
        raise UpstreamError("Signature verification failed.")

    try:
        event = json.loads(payload)
    except ValueError as exc:
        logger.warning("razorpay.webhook_bad_payload", error=str(exc))
        raise UpstreamError("Malformed webhook payload.") from exc

    if not isinstance(event, dict):
        raise UpstreamError("Malformed webhook payload.")
    return event


__all__ = [
    "LiveRazorpay",
    "RazorpayClient",
    "SignatureVerificationError",
    "get_client",
    "reset_client",
    "set_client",
    "verify_signature",
]
