"""The Stripe boundary.

Every call to Stripe goes through this module, for the same reason every email
goes through `integrations.email`: the rest of the app should not learn a
vendor's object model, and a test should not need a network.

Three things it provides:

  * a client that is `None` when the key is unset, so `billing_service` can
    answer "checkout unavailable" rather than raise `ImportError` in local
    development and CI;
  * `verify_signature`, which is the only thing standing between the webhook
    endpoint and an unauthenticated "make me Pro" API;
  * a `set_client` seam, so the test suite substitutes a fake and asserts on
    the arguments a checkout was created with.

The SDK's `v1` namespace is used throughout. The un-namespaced accessors are
deprecated in stripe-python 15, and this suite turns `DeprecationWarning` from
app modules into an error.
"""

from __future__ import annotations

from typing import Any, Protocol

import stripe

from app.core.config import settings
from app.core.errors import UpstreamError
from app.core.logging import get_logger

logger = get_logger("stripe")


class StripeClient(Protocol):
    """Only what this app calls.

    Narrow on purpose: a Protocol the size of the real SDK would make the test
    fake a maintenance burden, and the surface below is the whole of what
    billing needs.
    """

    async def create_customer(self, *, email: str, name: str, user_id: str) -> str: ...

    async def create_checkout_session(
        self,
        *,
        customer_id: str,
        price_id: str,
        client_reference_id: str,
        success_url: str,
        cancel_url: str,
        quantity: int,
        trial_days: int,
    ) -> tuple[str, str]: ...

    async def create_portal_session(self, *, customer_id: str, return_url: str) -> str: ...

    async def list_invoices(self, *, customer_id: str, limit: int) -> list[dict[str, Any]]: ...

    async def cancel_at_period_end(self, *, subscription_id: str, cancel: bool) -> None: ...


class LiveStripe:
    """The real thing.

    Every method translates a `StripeError` into `UpstreamError`. A card
    processor being down is a 502, not a 500 — the distinction matters because
    one is our bug and the other is not, and the alert routing differs.
    """

    def __init__(self, api_key: str) -> None:
        self._client = stripe.StripeClient(api_key)

    async def create_customer(self, *, email: str, name: str, user_id: str) -> str:
        try:
            customer = await self._client.v1.customers.create_async(
                params={
                    "email": email,
                    "name": name,
                    # The link back. Without it, a customer found in the Stripe
                    # dashboard cannot be traced to an account here without a
                    # database query nobody in support can run.
                    "metadata": {"user_id": user_id},
                }
            )
        except stripe.StripeError as exc:
            raise _upstream("customer creation", exc) from exc
        return str(customer.id)

    async def create_checkout_session(
        self,
        *,
        customer_id: str,
        price_id: str,
        client_reference_id: str,
        success_url: str,
        cancel_url: str,
        quantity: int,
        trial_days: int,
    ) -> tuple[str, str]:
        # Typed `Any` rather than `dict[str, Any]`. The SDK annotates this
        # argument with a TypedDict whose only importable path is a private
        # module, and the trial fields below are set conditionally — which a
        # TypedDict literal cannot express without building two of them.
        params: Any = {
            "mode": "subscription",
            "customer": customer_id,
            # Carries the user or organization id through the redirect, so the
            # webhook attributes the subscription without racing the browser
            # back to the success page.
            "client_reference_id": client_reference_id,
            "line_items": [{"price": price_id, "quantity": quantity}],
            "success_url": success_url,
            "cancel_url": cancel_url,
            "allow_promotion_codes": True,
            "billing_address_collection": "auto",
            "subscription_data": {"metadata": {"client_reference_id": client_reference_id}},
        }
        if trial_days > 0:
            # `payment_method_collection: if_required` is what makes this a
            # no-card trial (Q-04). Without it Stripe asks for a card up front
            # and the trial stops being the low-friction thing it is for.
            params["subscription_data"]["trial_period_days"] = trial_days
            params["payment_method_collection"] = "if_required"

        try:
            session = await self._client.v1.checkout.sessions.create_async(params=params)
        except stripe.StripeError as exc:
            raise _upstream("checkout", exc) from exc

        if not session.url:  # pragma: no cover — Stripe always returns one
            raise UpstreamError("Stripe did not return a checkout URL.")
        return str(session.id), str(session.url)

    async def create_portal_session(self, *, customer_id: str, return_url: str) -> str:
        try:
            session = await self._client.v1.billing_portal.sessions.create_async(
                params={"customer": customer_id, "return_url": return_url}
            )
        except stripe.StripeError as exc:
            raise _upstream("billing portal", exc) from exc
        return str(session.url)

    async def list_invoices(self, *, customer_id: str, limit: int) -> list[dict[str, Any]]:
        """Invoices are read live rather than mirrored into a table.

        They are Stripe's record, they are immutable once finalised, and a
        local copy would be one more thing to reconcile for a list most users
        open twice a year.
        """
        try:
            page = await self._client.v1.invoices.list_async(
                params={"customer": customer_id, "limit": limit}
            )
        except stripe.StripeError as exc:
            raise _upstream("invoice list", exc) from exc

        return [
            {
                "id": invoice.id,
                "number": invoice.number,
                "status": invoice.status,
                "amount_due": invoice.amount_due,
                "amount_paid": invoice.amount_paid,
                "currency": invoice.currency,
                "created": invoice.created,
                "hosted_invoice_url": invoice.hosted_invoice_url,
                "invoice_pdf": invoice.invoice_pdf,
            }
            for invoice in page.data
        ]

    async def cancel_at_period_end(self, *, subscription_id: str, cancel: bool) -> None:
        try:
            await self._client.v1.subscriptions.update_async(
                subscription_id, params={"cancel_at_period_end": cancel}
            )
        except stripe.StripeError as exc:
            raise _upstream("subscription update", exc) from exc


def _upstream(action: str, exc: stripe.StripeError) -> UpstreamError:
    logger.error("stripe.call_failed", action=action, error=str(exc))
    return UpstreamError(f"The payment provider rejected the {action} request.")


_client: StripeClient | None = None
_resolved = False


def get_client() -> StripeClient | None:
    """The client, or `None` when Stripe is not configured.

    `None` is a supported state, not a failure. Local development, CI, and
    every test run happen without a key, and billing has to degrade to
    "checkout unavailable" rather than break the app that surrounds it.
    """
    global _client, _resolved
    if not _resolved:
        _client = LiveStripe(settings.stripe_secret_key) if settings.stripe_secret_key else None
        _resolved = True
    return _client


def set_client(client: StripeClient | None) -> None:
    """Test seam."""
    global _client, _resolved
    _client = client
    _resolved = True


def verify_signature(payload: bytes, signature: str | None) -> dict[str, Any]:
    """Parse a webhook body, or raise.

    Never parses an unverified body — not even to log what it claimed to be.
    The signature is the only thing distinguishing a Stripe delivery from
    anyone on the internet POSTing `customer.subscription.updated` with a plan
    of their choosing, and a handler that reads the payload before checking has
    already lost.

    An unset webhook secret rejects everything. That is deliberately noisier
    than accepting: a silently-unverified endpoint in production is the worst
    outcome available here.
    """
    if not settings.stripe_webhook_secret:
        logger.error("stripe.webhook_secret_missing")
        raise UpstreamError("Webhook verification is not configured.")

    if not signature:
        raise UpstreamError("Missing Stripe signature.")

    try:
        event = stripe.Webhook.construct_event(payload, signature, settings.stripe_webhook_secret)
    except stripe.SignatureVerificationError as exc:
        logger.warning("stripe.webhook_bad_signature", error=str(exc))
        raise UpstreamError("Signature verification failed.") from exc
    except ValueError as exc:
        logger.warning("stripe.webhook_bad_payload", error=str(exc))
        raise UpstreamError("Malformed webhook payload.") from exc

    return dict(event)


__all__ = [
    "LiveStripe",
    "StripeClient",
    "get_client",
    "set_client",
    "verify_signature",
]
