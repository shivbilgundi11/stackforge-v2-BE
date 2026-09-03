"""The plan catalog, the feature map, and the default quota rows (M20).

Three things live here, and only one of them is authoritative at runtime.

**The plan catalog is code.** Prices, copy, and trial length are a marketing
surface that changes with a deploy anyway — the pricing page is generated from
this, so the page cannot claim a number the checkout will not charge.

**The feature map is code.** "PDF export is a Pro feature" is a product
definition, not a tunable. Moving a feature between plans changes what the
upgrade dialog says and what the pricing table lists, so it is a change to the
product, reviewed as one.

**The quota numbers here are only a seed.** `plan_quotas` is the authority once
the seeder has run, and `FeatureService` never reads `DEFAULT_QUOTAS`. These
values exist so a fresh database has working limits and so the numbers have one
written-down origin — but changing the free tier from 5 runs to 10 is an
`UPDATE` against the table, and a re-seed will not overwrite it (the seeder is
non-destructive, same rule as the catalog).

## One row per plan

There used to be a fifth bucket — `(plan=free, anonymous=true)`, the allowance
for a caller with no account. The product no longer has callers with no
account: every surface is behind sign-in, so free *is* the entry tier and the
`anonymous` discriminator that keyed these rows is gone.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Final

from app.models.billing import Metric
from app.models.export import ExportFormat
from app.models.user import Plan


class Feature(str, enum.Enum):
    """A capability a plan may or may not have.

    Deliberately *not* one per route. A feature is something a user would
    recognise as a thing they are buying — "PDF export", "team workspace" —
    because every one of these ends up as a line on the pricing page and a
    sentence in the upgrade dialog. `POST /projects/{id}/items` is not a
    feature; the projects quota is.
    """

    EXPORT_MARKDOWN = "export_markdown"
    EXPORT_JSON = "export_json"
    EXPORT_YAML = "export_yaml"
    EXPORT_CSV = "export_csv"
    EXPORT_PDF = "export_pdf"
    EXPORT_ZIP = "export_zip"

    SAVE_WORK = "save_work"
    SHARE_LINKS = "share_links"
    PROJECTS = "projects"

    PREMIUM_TEMPLATES = "premium_templates"
    P2_TOOLS = "p2_tools"
    TEAM_WORKSPACE = "team_workspace"
    ALERTS = "alerts"


@dataclass(frozen=True)
class FeatureSpec:
    key: Feature
    label: str
    minimum_plan: Plan
    #: What the locked state says. Naming the thing being sold rather than the
    #: wall in front of it — M18's UpgradeDialog reads this.
    pitch: str


FEATURES: Final[tuple[FeatureSpec, ...]] = (
    FeatureSpec(
        key=Feature.EXPORT_MARKDOWN,
        label="Markdown export",
        minimum_plan=Plan.FREE,
        pitch="The whole answer as Markdown — every figure, table, and source on the page.",
    ),
    FeatureSpec(
        key=Feature.EXPORT_JSON,
        label="JSON export",
        minimum_plan=Plan.PRO,
        pitch=(
            "The full result as structured data, with a versioned envelope — "
            "for a pipeline, a diff, or a script."
        ),
    ),
    FeatureSpec(
        key=Feature.EXPORT_YAML,
        label="YAML export",
        minimum_plan=Plan.PRO,
        pitch="The same structured data as JSON, in the format your config tooling already reads.",
    ),
    FeatureSpec(
        key=Feature.EXPORT_CSV,
        label="CSV export",
        minimum_plan=Plan.PRO,
        pitch="One table, ready for a spreadsheet.",
    ),
    FeatureSpec(
        key=Feature.EXPORT_PDF,
        label="PDF export",
        minimum_plan=Plan.PRO,
        pitch=(
            "A laid-out, paginated document with a cover page and your share link "
            "in the footer — the version you send to a client."
        ),
    ),
    FeatureSpec(
        key=Feature.EXPORT_ZIP,
        label="Bundle export",
        minimum_plan=Plan.PRO,
        pitch=(
            "Every file in the plan as one download: the architecture document, "
            "the diagram, the roadmap, a starter Compose file, and .cursorrules."
        ),
    ),
    FeatureSpec(
        key=Feature.SAVE_WORK,
        label="Saved work",
        minimum_plan=Plan.FREE,
        pitch="Keep a run past the 30-day purge and reopen it whenever you like.",
    ),
    FeatureSpec(
        key=Feature.SHARE_LINKS,
        label="Share links",
        minimum_plan=Plan.FREE,
        pitch="A public link to a result that opens logged out — and that you can revoke.",
    ),
    FeatureSpec(
        key=Feature.PROJECTS,
        label="Projects",
        minimum_plan=Plan.PRO,
        pitch=(
            "Group runs, stacks, and exports into a project, and carry one session between tools."
        ),
    ),
    FeatureSpec(
        key=Feature.PREMIUM_TEMPLATES,
        label="Premium templates",
        minimum_plan=Plan.PRO,
        pitch="The full template library, including the production blueprints and code starters.",
    ),
    FeatureSpec(
        key=Feature.P2_TOOLS,
        label="Advanced tools",
        minimum_plan=Plan.PRO,
        pitch="The advanced calculators beyond the twenty-eight open ones.",
    ),
    FeatureSpec(
        key=Feature.TEAM_WORKSPACE,
        label="Team workspace",
        minimum_plan=Plan.TEAM,
        pitch="A shared workspace with roles, shared stacks, comments, and approvals.",
    ),
    FeatureSpec(
        key=Feature.ALERTS,
        label="Price and deprecation alerts",
        minimum_plan=Plan.PRO,
        pitch=(
            "An email when a price in one of your saved estimates moves more than 10 %, "
            "or when a tool you depend on is deprecated."
        ),
    ),
)

BY_FEATURE: Final[dict[Feature, FeatureSpec]] = {spec.key: spec for spec in FEATURES}

#: Export format → the feature that unlocks it. `export_service` used to hold
#: this as its own plan map; the mapping stays, but the *verdict* now comes
#: from one place.
FORMAT_FEATURES: Final[dict[ExportFormat, Feature]] = {
    ExportFormat.MARKDOWN: Feature.EXPORT_MARKDOWN,
    ExportFormat.JSON: Feature.EXPORT_JSON,
    ExportFormat.YAML: Feature.EXPORT_YAML,
    ExportFormat.CSV: Feature.EXPORT_CSV,
    ExportFormat.PDF: Feature.EXPORT_PDF,
    ExportFormat.ZIP: Feature.EXPORT_ZIP,
}


# ── Plans ───────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Price:
    """One plan's amount in one currency, in that currency's minor units.

    Only INR is ever charged. Razorpay settles in rupees on a standard Indian
    account, so every other currency here is a *reading* of the price: the
    dollar figure is what a page may show, the rupee figure is what the card
    is debited. Every surface that quotes an amount someone is about to pay
    quotes the INR one.

    Hand-set rather than converted at render time. A live rate moves the shelf
    price of the product between two page loads, and a hardcoded rate is a lie
    with a delay on it — so the dollar price is chosen, reviewed, and shipped
    like the rupee one.
    """

    currency: str
    #: Minor units, per month. `None` means "talk to us" — Enterprise has no
    #: self-serve price, and a page that invents one would be lying.
    monthly_minor: int | None
    #: Annual billing at two months free, so twelve months costs ten.
    annual_minor: int | None

    @property
    def annual_saving_minor(self) -> int:
        """What the annual price saves against twelve monthly payments."""
        if self.monthly_minor is None or self.annual_minor is None:
            return 0
        return max(0, self.monthly_minor * 12 - self.annual_minor)


@dataclass(frozen=True)
class PlanSpec:
    plan: Plan
    label: str
    tagline: str
    #: Minor units (paise), per month. `None` means "talk to us" — Enterprise
    #: has no self-serve price, and a page that invents one would be lying.
    #:
    #: This is the charged amount: `razorpay-sync` builds the provider's plans
    #: from it, and it is what the checkout debits.
    monthly_minor: int | None
    #: Annual billing at two months free, so twelve months costs ten.
    annual_minor: int | None
    #: The same plan in the currencies it can be *read* in, INR excluded — it
    #: is already above. Display only; see `Price`.
    display_prices: tuple[Price, ...]
    per_seat: bool
    #: Zero on every plan: the product does not sell a trial.
    #:
    #: Kept as a field rather than deleted because it is not only copy. A
    #: non-zero value makes `billing_service` set `start_at` in the future,
    #: and Razorpay reads a future start as "authorize the mandate now, charge
    #: later" — so Checkout collects its token authorization amount (₹5)
    #: instead of the plan price, and the customer sees ₹5 where ₹499 was
    #: advertised. The trial machinery downstream (`trial_ends_at`,
    #: `_has_trialed`, the expiry reminder worker) still exists to serve any
    #: subscription created while this was 7; nothing new enters it.
    trial_days: int
    #: The pricing table's bullet list, in order.
    highlights: tuple[str, ...]
    cta: str
    #: Enterprise is listed but not purchasable; free is the default, not a
    #: thing to buy.
    checkout: bool


#: Razorpay settles in INR on a standard Indian account; charging USD needs
#: International Payments enabled and approved. Every `monthly_minor` and
#: `annual_minor` below is paise, and it is the amount the card is debited.
CURRENCY: Final = "inr"

#: Every currency a price may be *shown* in, charged one first. The settings
#: toggle offers exactly these, and `display_prices` carries the rest of them
#: per plan. Adding one here without adding it to every priced plan leaves
#: that plan quoting rupees while the rest of the page quotes dollars — which
#: is what `test_plans.py` asserts against.
DISPLAY_CURRENCIES: Final[tuple[str, ...]] = ("inr", "usd")

PLANS: Final[tuple[PlanSpec, ...]] = (
    PlanSpec(
        plan=Plan.FREE,
        label="Free",
        tagline="Every tool, every catalog page, no card.",
        monthly_minor=0,
        annual_minor=0,
        display_prices=(Price(currency="usd", monthly_minor=0, annual_minor=0),),
        per_seat=False,
        trial_days=0,
        highlights=(
            # Was "All 28 tools and both workflows". There are five workflows
            # plus Stack Architect and Compare Center, and a hardcoded tool
            # count drifts the moment a spec file is added — this string is
            # rendered on the public pricing page (M22), where a wrong claim
            # is a copy-accuracy defect rather than a typo.
            "Every tool and every workflow",
            "25 tool runs a day",
            "Markdown export on every result",
            "One saved stack",
        ),
        cta="Get started",
        checkout=False,
    ),
    PlanSpec(
        plan=Plan.PRO,
        label="Pro",
        tagline="For the person who has to defend the number.",
        monthly_minor=49_900,
        annual_minor=499_900,
        display_prices=(Price(currency="usd", monthly_minor=599, annual_minor=5_999),),
        per_seat=False,
        trial_days=0,
        highlights=(
            "Unlimited tool runs",
            "100 AI-assisted runs a day",
            "PDF, JSON, YAML, CSV, and bundle export",
            "20 projects and unlimited saved stacks",
            "Price-change and deprecation alerts",
            "The full template library",
        ),
        cta="Upgrade to Pro",
        checkout=True,
    ),
    PlanSpec(
        plan=Plan.TEAM,
        label="Team",
        tagline="One shared view of what the team has decided.",
        monthly_minor=129_900,
        annual_minor=1_299_900,
        display_prices=(Price(currency="usd", monthly_minor=1_499, annual_minor=14_999),),
        per_seat=True,
        trial_days=0,
        highlights=(
            "Everything in Pro, per seat",
            "300 AI-assisted runs a day",
            "Shared workspace, roles, and approvals",
            "Unlimited projects",
            "500 exports a month",
        ),
        cta="Choose Team",
        checkout=True,
    ),
    PlanSpec(
        plan=Plan.ENTERPRISE,
        label="Enterprise",
        tagline="Your procurement process, our numbers.",
        monthly_minor=None,
        annual_minor=None,
        display_prices=(Price(currency="usd", monthly_minor=None, annual_minor=None),),
        per_seat=True,
        trial_days=0,
        highlights=(
            "Everything in Team, without the ceilings",
            "Custom AI allowance",
            "SSO and an audit trail",
            "A named contact",
        ),
        cta="Talk to us",
        checkout=False,
    ),
)

BY_PLAN: Final[dict[Plan, PlanSpec]] = {spec.plan: spec for spec in PLANS}

#: Ordering, for "is this plan at least that plan" questions. Every comparison
#: in the app goes through this — a string compare would put `enterprise`
#: below `free` alphabetically.
PLAN_RANK: Final[dict[Plan, int]] = {
    Plan.FREE: 0,
    Plan.PRO: 1,
    Plan.TEAM: 2,
    Plan.ENTERPRISE: 3,
}


# ── Default quotas ──────────────────────────────────────────────────────────
#
# `None` is unlimited. Not a large number — a sentinel like 999_999 renders as
# a real limit ("3 of 999999 used") and invites arithmetic that treats it as
# one.

#: `plan -> {metric: limit}`.
DEFAULT_QUOTAS: Final[dict[Plan, dict[Metric, int | None]]] = {
    Plan.FREE: {
        # The entry tier, and the one every new account lands on. Generous
        # enough to plan a real stack with; not so generous that a team never
        # needs Pro.
        Metric.TOOL_RUNS_PER_DAY: 25,
        Metric.AI_CALLS_PER_DAY: 3,
        Metric.PROJECTS: 0,
        Metric.SAVED_STACKS: 1,
        # Format-gated rather than count-gated — free is Markdown only, and
        # this ceiling is an abuse stop, not a product boundary.
        Metric.EXPORTS_PER_MONTH: 30,
        Metric.SEATS: 0,
    },
    Plan.PRO: {
        Metric.TOOL_RUNS_PER_DAY: None,
        Metric.AI_CALLS_PER_DAY: 100,
        Metric.PROJECTS: 20,
        Metric.SAVED_STACKS: None,
        Metric.EXPORTS_PER_MONTH: 200,
        Metric.SEATS: 1,
    },
    Plan.TEAM: {
        Metric.TOOL_RUNS_PER_DAY: None,
        Metric.AI_CALLS_PER_DAY: 300,
        Metric.PROJECTS: None,
        Metric.SAVED_STACKS: None,
        Metric.EXPORTS_PER_MONTH: 500,
        # The seat *count* is bought in Razorpay and written onto the
        # subscription; this is the floor a Team plan starts at.
        Metric.SEATS: 5,
    },
    Plan.ENTERPRISE: {
        Metric.TOOL_RUNS_PER_DAY: None,
        Metric.AI_CALLS_PER_DAY: None,
        Metric.PROJECTS: None,
        Metric.SAVED_STACKS: None,
        Metric.EXPORTS_PER_MONTH: None,
        Metric.SEATS: None,
    },
}


def spec_for(plan: Plan) -> PlanSpec:
    return BY_PLAN[plan]


def feature_spec(feature: Feature) -> FeatureSpec:
    return BY_FEATURE[feature]


def outranks(plan: Plan, minimum: Plan) -> bool:
    return PLAN_RANK[plan] >= PLAN_RANK[minimum]


def annual_saving_minor(spec: PlanSpec) -> int:
    """What the annual price saves against twelve monthly payments, in INR."""
    return charged_price(spec).annual_saving_minor


def charged_price(spec: PlanSpec) -> Price:
    """The amount that actually leaves a card. Always INR."""
    return Price(
        currency=CURRENCY,
        monthly_minor=spec.monthly_minor,
        annual_minor=spec.annual_minor,
    )


def prices_for(spec: PlanSpec) -> tuple[Price, ...]:
    """Every currency this plan can be read in, the charged one first.

    Order matters on the wire: a client that does not recognise the currency
    it was asked for falls back to the head of this list, which is the only
    one guaranteed to be both present and true.
    """
    return (charged_price(spec), *spec.display_prices)


__all__ = [
    "BY_FEATURE",
    "BY_PLAN",
    "CURRENCY",
    "DEFAULT_QUOTAS",
    "DISPLAY_CURRENCIES",
    "FEATURES",
    "FORMAT_FEATURES",
    "PLANS",
    "PLAN_RANK",
    "Feature",
    "FeatureSpec",
    "PlanSpec",
    "Price",
    "annual_saving_minor",
    "charged_price",
    "feature_spec",
    "outranks",
    "prices_for",
    "spec_for",
]
