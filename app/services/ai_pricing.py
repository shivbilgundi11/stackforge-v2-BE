"""What our own model calls cost us.

Deliberately **not** `model_pricing`. That table is user-facing product
content, edited by editorial staff on a whim and by design; billing internal
accounting off it would mean a content edit changes the books. These are the
rates we are charged, and they move on Anthropic's schedule, not ours.

Rates are per **million** tokens, matching how they are published. Everything
downstream works in dollars per token, so the division happens once, here.
"""

from __future__ import annotations

from datetime import date
from decimal import ROUND_HALF_UP, Decimal
from typing import Final, NamedTuple

MILLION: Final = Decimal(1_000_000)
MICRO: Final = Decimal("0.000001")

#: Cache writes cost more than an ordinary input token and cache reads cost
#: far less. Both are multipliers on the input rate, published per TTL — these
#: are the 5-minute figures, which is the TTL used throughout.
CACHE_WRITE_MULTIPLIER: Final = Decimal("1.25")
CACHE_READ_MULTIPLIER: Final = Decimal("0.1")


class ModelRate(NamedTuple):
    model: str
    input_per_mtok: Decimal
    output_per_mtok: Decimal
    #: Shortest prefix that will cache. Below it nothing is cached and no error
    #: is raised — `cached_write_tokens` simply stays zero. Not uniform across
    #: models, and not monotonic across generations, so it is data rather than
    #: a constant.
    cache_minimum_tokens: int


VERIFIED_ON: Final = date(2026, 8, 10)

RATES: Final[dict[str, ModelRate]] = {
    "claude-opus-5": ModelRate("claude-opus-5", Decimal(5), Decimal(25), 512),
    "claude-sonnet-5": ModelRate("claude-sonnet-5", Decimal(3), Decimal(15), 1024),
    "claude-haiku-4-5": ModelRate("claude-haiku-4-5", Decimal(1), Decimal(5), 4096),
}

#: Charged when a model we have no rate for is somehow called. Priced at the
#: most expensive rate on purpose: an unknown model should read as expensive in
#: the ledger and get noticed, not silently cost nothing.
FALLBACK_RATE: Final = ModelRate("unknown", Decimal(5), Decimal(25), 1024)


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
