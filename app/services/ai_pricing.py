"""What our own model calls cost us.

Deliberately **not** `model_pricing`. That table is user-facing product
content, edited by editorial staff on a whim and by design; billing internal
accounting off it would mean a content edit changes the books. These are the
rates we are charged, and they move on the provider's schedule, not ours.

Rates are per **million** tokens, matching how they are published. Everything
downstream works in dollars per token, so the division happens once, here.

Synthesis runs on Claude, so these are Anthropic's first-party API rates for
the models in `ai_prompts`, plus the model a refusal fallback can route to.
The multipliers below are Anthropic's too — they are not universal, which is
the reason they are named constants rather than literals at the arithmetic.
"""

from __future__ import annotations

from datetime import date
from decimal import ROUND_HALF_UP, Decimal
from typing import Final, NamedTuple

MILLION: Final = Decimal(1_000_000)
MICRO: Final = Decimal("0.000001")

#: Both are multipliers on the model's input rate.
#:
#: Caching is explicit here: nothing is cached without a `cache_control`
#: marker, and populating the cache **costs more** than sending the same tokens
#: uncached — 1.25x for the default five-minute TTL. A write that is never read
#: back is a small loss, which is why the marker sits only on the part of the
#: request that repeats.
#:
#: A cached read is billed at a tenth of the input rate, which is a large
#: enough discount that a working cache is visible in the ledger rather than
#: inferred from it.
CACHE_WRITE_MULTIPLIER: Final = Decimal("1.25")
CACHE_READ_MULTIPLIER: Final = Decimal("0.1")


class ModelRate(NamedTuple):
    model: str
    input_per_mtok: Decimal
    output_per_mtok: Decimal
    #: Shortest prefix that will cache. Below it nothing is cached and no error
    #: is raised — the cached counts simply stay zero. Not uniform across
    #: models, and not monotonic across generations, so it is data rather than
    #: a constant.
    cache_minimum_tokens: int


#: Anthropic's published first-party rates on this date. Output rates include
#: thinking tokens, which `usage.output_tokens` already counts.
VERIFIED_ON: Final = date(2026, 6, 24)

RATES: Final[dict[str, ModelRate]] = {
    "claude-opus-5": ModelRate("claude-opus-5", Decimal("5.00"), Decimal("25.00"), 512),
    # Where `fallbacks: "default"` sends a cyber-category refusal. Listed so an
    # answer it rescued is priced at its real rate rather than the fallback.
    "claude-opus-4-8": ModelRate("claude-opus-4-8", Decimal("5.00"), Decimal("25.00"), 1024),
}

#: Charged when a model we have no rate for is somehow called — including a
#: fallback model Anthropic starts routing to that is not listed above. Priced
#: at the most expensive rate in the table on purpose: an unknown model should
#: read as expensive in the ledger and get noticed, not silently cost nothing.
FALLBACK_RATE: Final = ModelRate("unknown", Decimal("5.00"), Decimal("25.00"), 1024)


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
