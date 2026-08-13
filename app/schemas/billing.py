"""Billing wire shapes (M20).

The pricing page is generated from `GET /billing/plans`, which is why
`PricingPlanOut` carries the limits as well as the copy: a marketing page with its own
hardcoded table drifts from the enforcement the first time a limit moves, and
the drift is invisible until a user hits a wall the page said was not there.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from app.schemas.tools import QuotaOut


class PlanFeatureOut(BaseModel):
    key: str
    label: str
    included: bool
    #: What this unlocks, in the words the upgrade dialog uses.
    pitch: str


class PlanLimitOut(BaseModel):
    metric: str
    label: str
    #: `null` is unlimited. The client renders the word, not a number.
    limit: int | None


class PricingPlanOut(BaseModel):
    key: str
    label: str
    tagline: str
    #: Minor units. `null` on Enterprise, which has no self-serve price — a
    #: page that invented one would be lying.
    monthly_cents: int | None
    annual_cents: int | None
    #: What annual billing saves against twelve monthly payments.
    annual_saving_cents: int
    currency: str
    per_seat: bool
    trial_days: int
    highlights: list[str]
    cta: str
    #: False for Free and Enterprise: one is the default, the other is a
    #: conversation.
    checkout: bool
    #: True for the caller's own plan, so the page can mark it rather than
    #: offering to sell it again.
    current: bool
    features: list[PlanFeatureOut]
    limits: list[PlanLimitOut]


class SubscriptionOut(BaseModel):
    plan: str
    status: str | None
    #: False when this environment has no payment provider configured. The UI
    #: hides checkout rather than showing a button that returns a 402.
    checkout_available: bool
    seats: int
    cancel_at_period_end: bool
    current_period_end: datetime | None
    trial_ends_at: datetime | None
    past_due_since: datetime | None
    #: Days before a failed payment downgrades the account, or null.
    grace_days_left: int | None


class UsageSummaryOut(BaseModel):
    """Every meter at once. One call rather than six from the client."""

    plan: str
    quotas: list[QuotaOut]


class CheckoutIn(BaseModel):
    plan: str
    interval: str = Field(default="monthly", pattern="^(monthly|annual)$")
    seats: int = Field(default=1, ge=1, le=500)


class CheckoutOut(BaseModel):
    url: str


class PortalOut(BaseModel):
    url: str


class InvoiceOut(BaseModel):
    id: str
    number: str | None
    status: str | None
    amount_due: int
    amount_paid: int
    currency: str
    created: datetime
    hosted_invoice_url: str | None
    invoice_pdf: str | None


class CancellationIn(BaseModel):
    cancel: bool = True


class SeatChangeIn(BaseModel):
    seats: int = Field(ge=1, le=500)
    #: Which organization, when the caller belongs to more than one. Omitted
    #: means the one they own.
    organization_id: str | None = Field(default=None, max_length=64)


class SeatChangeOut(BaseModel):
    seats: int
    used: int


class WebhookOut(BaseModel):
    """Deliberately uninformative to the caller.

    A webhook response is read by Stripe's retry logic and by anyone probing
    the endpoint. `received` is all either needs; whether an event applied,
    was a duplicate, or failed is in the logs and the `stripe_events` table.
    """

    received: bool


__all__ = [
    "CancellationIn",
    "CheckoutIn",
    "CheckoutOut",
    "InvoiceOut",
    "PlanFeatureOut",
    "PlanLimitOut",
    "PortalOut",
    "PricingPlanOut",
    "SubscriptionOut",
    "UsageSummaryOut",
    "WebhookOut",
]
