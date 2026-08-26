"""What our own model calls cost us.

Deliberately **not** `model_pricing`. That table is user-facing product
content, edited by editorial staff on a whim and by design; billing internal
accounting off it would mean a content edit changes the books. These are the
rates we are charged, and they move on the provider's schedule, not ours.

Rates are per **million** tokens, matching how they are published. Everything
downstream works in dollars per token, so the division happens once, here.

Synthesis runs on Groq, so these are Groq's on-demand rates for the models in
`ai_prompts`. The multipliers below are Groq's too — they are not universal,
which is the reason they are named constants rather than literals at the
arithmetic.
"""

from __future__ import annotations

from datetime import date
from decimal import ROUND_HALF_UP, Decimal
from typing import Final, NamedTuple

MILLION: Final = Decimal(1_000_000)
MICRO: Final = Decimal("0.000001")

#: Both are multipliers on the model's input rate.
#:
#: Groq's cache is automatic: there is no marker to send, no TTL to choose,
#: and **no surcharge for populating it** — hence a write multiplier of
#: exactly 1, which is not a placeholder. `cached_write_tokens` is always zero
#: on this provider anyway; the constant stays so the arithmetic can describe
#: a provider that does charge, and so the day one is added the change is a
#: number here rather than a new term in `cost_of`.
#:
#: A cached read is billed at half. That is a real discount but a much smaller
#: one than a provider with an explicit cache offers, so cache hits move this
#: ledger less than the same hit rate used to.
CACHE_WRITE_MULTIPLIER: Final = Decimal(1)
CACHE_READ_MULTIPLIER: Final = Decimal("0.5")


class ModelRate(NamedTuple):
    model: str
    input_per_mtok: Decimal
    output_per_mtok: Decimal
    #: Shortest prefix that will cache. Below it nothing is cached and no error
    #: is raised — the cached counts simply stay zero. Not uniform across
    #: models, and not monotonic across generations, so it is data rather than
    #: a constant. Groq does not publish a threshold; these are the observed
    #: floor and exist to keep the field honest rather than to be relied on,
    #: because a miss on this provider costs the discount and nothing else.
    cache_minimum_tokens: int


#: Read off console.groq.com/docs/models and the per-model pages on this date.
#: Aggregator sites disagree with the vendor on the 120B output rate; the
#: vendor's own figure is the one here.
VERIFIED_ON: Final = date(2026, 8, 19)

RATES: Final[dict[str, ModelRate]] = {
    "gemini-2.5-flash": ModelRate("gemini-2.5-flash", Decimal("0.30"), Decimal("2.50"), 1024),
    "openai/gpt-oss-120b": ModelRate("openai/gpt-oss-120b", Decimal("0.15"), Decimal("0.60"), 1024),
    "openai/gpt-oss-20b": ModelRate("openai/gpt-oss-20b", Decimal("0.075"), Decimal("0.30"), 1024),
}

#: Charged when a model we have no rate for is somehow called. Priced at the
#: most expensive rate on purpose: an unknown model should read as expensive in
#: the ledger and get noticed, not silently cost nothing.
FALLBACK_RATE: Final = ModelRate("unknown", Decimal("0.15"), Decimal("0.60"), 1024)


def rate_for(model: str) -> ModelRate:
    return RATES.get(model, FALLBACK_RATE)


def cost_of(
    *,
    model: str,
    input_tokens: int,
    output_tokens: int,
    cached_read_tokens: int = 0,
    cached_write_tokens: int = 0,
) -> Decimal:
    """Dollars for one call.

    `input_tokens` is the *uncached remainder* — the API reports cached reads
    and writes separately, and adding them back in would double-count the
    prompt and make caching look like it cost more, not less.
    """
    rate = rate_for(model)
    per_input = rate.input_per_mtok / MILLION
    per_output = rate.output_per_mtok / MILLION

    total = (
        Decimal(input_tokens) * per_input
        + Decimal(output_tokens) * per_output
        + Decimal(cached_read_tokens) * per_input * CACHE_READ_MULTIPLIER
        + Decimal(cached_write_tokens) * per_input * CACHE_WRITE_MULTIPLIER
    )
    return total.quantize(MICRO, rounding=ROUND_HALF_UP)
