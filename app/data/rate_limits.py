"""Published API rate limits, by provider and tier.

The same treatment as pricing (D-16): hand-verified from the provider's own
rate-limit page, dated, and never inferred from a neighbouring number. Nothing
in here is scraped, because none of these pages are machine-readable and a
half-parsed limits table is worse than a stale one — it fails silently in the
direction of "you have more headroom than you do".

Two facts about this data shape the tool that reads it:

**Token budgets are not one number everywhere.** OpenAI and Google publish a
single TPM covering input and output together. Anthropic publishes input and
output separately, and on agent workloads the output budget is routinely the
one that binds first — long tool-calling loops emit far more output per minute
than a chat product does, and a calculator that folds the two together cannot
show that. `combined_tokens` records which model a provider uses.

**These are defaults for the named model family.** Per-model limits differ,
negotiated limits differ, and a tier can be raised mid-month by spend. The
figures are a planning baseline; the account dashboard is authoritative, and
the tool says so on every run rather than in a footnote.
"""

from __future__ import annotations

from datetime import date
from typing import Final, NamedTuple


class TierLimits(NamedTuple):
    key: str
    label: str
    #: How an account arrives in this tier. Shown beside the upgrade suggestion,
    #: because "move to tier 3" is not actionable without it.
    qualifier: str
    requests_per_min: int
    input_tokens_per_min: int
    #: `None` when the provider bills input and output against one budget.
    output_tokens_per_min: int | None = None
    requests_per_day: int | None = None


class ProviderLimits(NamedTuple):
    key: str
    label: str
    #: Which family the numbers describe. Limits are per-model at every
    #: provider, so naming the family is the difference between a figure and a
    #: guess.
    model_family: str
    combined_tokens: bool
    source_name: str
    source_url: str
    verified_on: date
    tiers: tuple[TierLimits, ...]

    def tier(self, key: str) -> TierLimits | None:
        return next((tier for tier in self.tiers if tier.key == key), None)

    def next_tier(self, key: str) -> TierLimits | None:
        for index, tier in enumerate(self.tiers):
            if tier.key == key:
                return self.tiers[index + 1] if index + 1 < len(self.tiers) else None
        return None


VERIFIED_ON: Final = date(2026, 8, 10)

OPENAI: Final = ProviderLimits(
    key="openai",
    label="OpenAI",
    model_family="GPT-4o / GPT-4.1 class",
    combined_tokens=True,
    source_name="OpenAI — rate limits",
    source_url="https://platform.openai.com/docs/guides/rate-limits",
    verified_on=VERIFIED_ON,
    tiers=(
        TierLimits(
            key="free",
            label="Free",
            qualifier="No payment method on file. Not all models are available.",
            requests_per_min=3,
            input_tokens_per_min=40_000,
            requests_per_day=200,
        ),
        TierLimits(
            key="tier-1",
            label="Tier 1",
            qualifier="$5 paid",
            requests_per_min=500,
            input_tokens_per_min=30_000,
            requests_per_day=10_000,
        ),
        TierLimits(
            key="tier-2",
            label="Tier 2",
            qualifier="$50 paid and 7 days since first payment",
            requests_per_min=5_000,
            input_tokens_per_min=450_000,
        ),
        TierLimits(
            key="tier-3",
            label="Tier 3",
            qualifier="$100 paid and 7 days since first payment",
            requests_per_min=5_000,
            input_tokens_per_min=800_000,
        ),
        TierLimits(
            key="tier-4",
            label="Tier 4",
            qualifier="$250 paid and 14 days since first payment",
            requests_per_min=10_000,
            input_tokens_per_min=2_000_000,
        ),
        TierLimits(
            key="tier-5",
            label="Tier 5",
            qualifier="$1,000 paid and 30 days since first payment",
            requests_per_min=10_000,
            input_tokens_per_min=30_000_000,
        ),
    ),
)

ANTHROPIC: Final = ProviderLimits(
    key="anthropic",
    label="Anthropic",
    model_family="Claude Sonnet class",
    combined_tokens=False,
    source_name="Anthropic — rate limits",
    source_url="https://platform.claude.com/docs/en/api/rate-limits",
    verified_on=VERIFIED_ON,
    tiers=(
        TierLimits(
            key="tier-1",
            label="Tier 1",
            qualifier="$5 deposited",
            requests_per_min=50,
            input_tokens_per_min=30_000,
            output_tokens_per_min=8_000,
        ),
        TierLimits(
            key="tier-2",
            label="Tier 2",
            qualifier="$40 deposited and 7 days since first purchase",
            requests_per_min=1_000,
            input_tokens_per_min=450_000,
            output_tokens_per_min=90_000,
        ),
        TierLimits(
            key="tier-3",
            label="Tier 3",
            qualifier="$200 deposited and 7 days since first purchase",
            requests_per_min=2_000,
            input_tokens_per_min=800_000,
            output_tokens_per_min=160_000,
        ),
        TierLimits(
            key="tier-4",
            label="Tier 4",
            qualifier="$400 deposited and 14 days since first purchase",
            requests_per_min=4_000,
            input_tokens_per_min=2_000_000,
            output_tokens_per_min=400_000,
        ),
    ),
)

GOOGLE: Final = ProviderLimits(
    key="google",
    label="Google Gemini",
    model_family="Gemini Flash class",
    combined_tokens=True,
    source_name="Google — Gemini API rate limits",
    source_url="https://ai.google.dev/gemini-api/docs/rate-limits",
    verified_on=VERIFIED_ON,
    tiers=(
        TierLimits(
            key="free",
            label="Free",
            qualifier="No billing enabled",
            requests_per_min=15,
            input_tokens_per_min=250_000,
            requests_per_day=1_000,
        ),
        TierLimits(
            key="tier-1",
            label="Tier 1",
            qualifier="Billing enabled on the project",
            requests_per_min=2_000,
            input_tokens_per_min=4_000_000,
        ),
        TierLimits(
            key="tier-2",
            label="Tier 2",
            qualifier="$250 total spend and 30 days since first payment",
            requests_per_min=10_000,
            input_tokens_per_min=10_000_000,
        ),
        TierLimits(
            key="tier-3",
            label="Tier 3",
            qualifier="$1,000 total spend and 30 days since first payment",
            requests_per_min=30_000,
            input_tokens_per_min=30_000_000,
        ),
    ),
)

PROVIDERS: Final[tuple[ProviderLimits, ...]] = (ANTHROPIC, OPENAI, GOOGLE)

BY_KEY: Final[dict[str, ProviderLimits]] = {provider.key: provider for provider in PROVIDERS}
