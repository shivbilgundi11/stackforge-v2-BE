"""The only model client in the process.

Routes never call the API. They call a domain service, which calls this. One
client means one place that knows the request-shape rules, one place that
handles failure, and one place that writes the usage row.

**`generate_json` returns `None` for every failure.** No key, network error,
timeout, rate limit, refusal, malformed output, quota exhausted — all of them
are `None`. Callers branch once and set `source="rule_based"`. Nothing above
this service ever sees an exception from a model, which is what makes the
fallback one code path instead of a `try/except` copied into eleven endpoints.

That property is the module (D-06). The rule engine has already produced a
complete, returnable answer before this is called; AI is a layer over it and
never a gate in front of it.

The provider is **Anthropic (Claude)**, everywhere. It was Groq, then Gemini,
and one provider is still the point: one request shape, one failure taxonomy,
and one set of quota arithmetic to reason about before answering "why did this
come back rule_based". It is also the vendor `tokenizer_service` already
counts with, so the process holds one AI key rather than two.

Four consequences of this API are worth stating rather than discovering:

* **Thinking is on, and it is billed as output.** Claude Opus 5 thinks
  adaptively by default, and `usage.output_tokens` already includes those
  tokens — there is no separate figure to fold back in, and nothing may
  subtract it out.
* **Thinking also comes out of `max_tokens`.** A reservation that thinking
  exhausts stops with `stop_reason: "max_tokens"` and the JSON cut off, or
  never started. That is why the reservations in `ai_prompts` are sized
  against the thinking as well as the prose, and why the stop reason is
  recorded on the ledger row.
* **`output_config.effort` is the depth knob**, and it is the direct lever on
  both latency and spend. Every prompt in the registry sits at `low` or
  `medium` for that reason.
* **Caching is explicit.** Nothing caches without a `cache_control` marker, a
  cache write costs 1.25x the input rate, and a prefix under the model's
  minimum silently never caches. The marker sits on the system block, the
  only part of the request that is byte-identical per purpose.
"""

from __future__ import annotations

import json
import time
from collections.abc import Awaitable, Callable
from decimal import Decimal
from typing import Any, Final, NamedTuple

import anthropic
from anthropic import AsyncAnthropic
from anthropic.types.beta import BetaMessage, BetaTextBlock
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import Identity
from app.core.config import settings
from app.core.database import utcnow
from app.core.logging import get_logger
from app.models.ai import AiCall, AiOutcome
from app.models.billing import Metric
from app.schemas.tools import AiMeta, ToolOutput, ToolWarning
from app.services import ai_pricing, ai_prompts

logger = get_logger("ai")

#: A synthesis call that has not answered in this long is not going to save the
#: request. The deterministic result is already computed and waiting.
TIMEOUT_SECONDS: Final = 60.0

#: Server-side refusal fallback. When a safety classifier declines, the API
#: re-runs the same request on the model Anthropic recommends for that refusal
#: category, inside the same call, instead of handing back an empty answer.
#: `"default"` rather than a pinned model, so a retired fallback model is not a
#: migration this module owes. Whichever model served the answer is the one
#: priced and recorded.
FALLBACK_BETA: Final = "server-side-fallback-2026-07-01"


class AiResult(NamedTuple):
    data: dict[str, Any]
    meta: AiMeta


async def generate_json(
    db: AsyncSession,
    *,
    purpose: str,
    grounding: dict[str, Any],
    variables: dict[str, Any],
    identity: Identity,
    tool_slug: str | None = None,
) -> AiResult | None:
    """Run one synthesis call. `None` on any failure whatsoever.

    The schema comes from the registry, never from the caller, and the
    response is requested as structured output (`output_config.format`)
    rather than asked for in prose and parsed. Parsing prose JSON fails a few
    percent of the time, and each failure would silently degrade to
    `rule_based` with no signal separating "the prompt is wrong" from "the
    model was down".

    The stable half of the prompt goes in `system`, byte-identical per
    purpose and carrying the cache marker, and the rule-engine output that
    varies per request goes in the user turn after it. Caching is a prefix
    match, so anything variable placed ahead of the marker would turn every
    request into a cache write that is never read back.
    """
    prompt = ai_prompts.REGISTRY.get(purpose)
    if prompt is None:  # pragma: no cover — a programming error, not an input
        logger.error("ai.unknown_purpose", purpose=purpose)
        return None

    if not settings.ai_enabled:
        await _record(db, prompt, identity, tool_slug, AiOutcome.DISABLED, latency_ms=0)
        return None

    if _exhausted(await quota_remaining(db, identity)):
        await _record(db, prompt, identity, tool_slug, AiOutcome.QUOTA_EXCEEDED, latency_ms=0)
        return None

    started = time.perf_counter()
    try:
        # A client per call, as the `httpx` client before it was: nothing
        # outlives the event loop that opened it, which matters for the export
        # worker as much as for the test suite. Retries are off because a
        # retried call can no longer answer inside `TIMEOUT_SECONDS`, and the
        # rule-engine answer is already waiting.
        async with AsyncAnthropic(
            api_key=settings.anthropic_api_key, timeout=TIMEOUT_SECONDS, max_retries=0
        ) as client:
            response = await client.beta.messages.create(
                model=prompt.model,
                max_tokens=prompt.max_tokens,
                system=[
                    {
                        "type": "text",
                        "text": prompt.system,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[{"role": "user", "content": ai_prompts.user_turn(grounding, variables)}],
                # Adaptive is already Opus 5's default; stated so the request
                # says what it does. Thinking is billed as output and drawn
                # from `max_tokens`, so `effort` is a spend lever and a
                # truncation risk at once.
                thinking={"type": "adaptive"},
                output_config={
                    "effort": prompt.effort,
                    "format": {"type": "json_schema", "schema": prompt.schema},
                },
                betas=[FALLBACK_BETA],
                fallbacks="default",
            )
    except anthropic.APITimeoutError as exc:
        await _fail(db, prompt, identity, tool_slug, AiOutcome.TIMEOUT, started, exc)
        return None
    except anthropic.RateLimitError as exc:
        # 429 is the organisation's rate or spend limit, and it is the one
        # failure an operator can act on. Filed under `api_error` it would send
        # that investigation to the wrong place entirely.
        await _fail(db, prompt, identity, tool_slug, AiOutcome.RATE_LIMITED, started, exc)
        return None
    except (anthropic.APIStatusError, anthropic.APIConnectionError) as exc:
        # `APITimeoutError` is a connection error too, which is why it is
        # caught above rather than here.
        await _fail(db, prompt, identity, tool_slug, AiOutcome.API_ERROR, started, exc)
        return None
    except Exception as exc:
        # Deliberately last and deliberately broad. The contract is that
        # nothing from a model call escapes this function, and a contract that
        # only covers the exceptions we thought of is not one.
        await _fail(db, prompt, identity, tool_slug, AiOutcome.API_ERROR, started, exc)
        return None

    latency_ms = int((time.perf_counter() - started) * 1000)
    usage = _usage(response)
    # With a refusal fallback in play, the model that answered is not always
    # the model that was asked. The ledger names and prices the one that did
    # the work, or a row would reconcile against the wrong invoice line.
    served_by = response.model

    # A safety classifier can decline and still return a 200. With the fallback
    # on, this means every model in the chain declined. Reading the content
    # without checking would report the refusal as a schema failure, which is
    # the wrong thing to go and debug.
    if response.stop_reason == "refusal":
        category = response.stop_details.category if response.stop_details else None
        await _record(
            db,
            prompt,
            identity,
            tool_slug,
            AiOutcome.REFUSAL,
            latency_ms=latency_ms,
            usage=usage,
            model=served_by,
            detail=f"refusal:{category or 'unspecified'}",
        )
        await _consume_quota(db, identity)
        return None

    data = _answer_json(response)
    if data is None:
        await _record(
            db,
            prompt,
            identity,
            tool_slug,
            AiOutcome.INVALID_OUTPUT,
            latency_ms=latency_ms,
            usage=usage,
            model=served_by,
            # `max_tokens` here means thinking ate the reservation and the
            # JSON was cut off. Recording the reason is the difference between
            # raising `max_tokens` and rewriting a schema.
            detail=f"stop_reason={response.stop_reason}",
        )
        await _consume_quota(db, identity)
        return None

    cost = ai_pricing.cost_of(model=served_by, **usage)
    await _record(
        db,
        prompt,
        identity,
        tool_slug,
        AiOutcome.SUCCESS,
        latency_ms=latency_ms,
        usage=usage,
        cost=cost,
        model=served_by,
    )
    await _consume_quota(db, identity)

    logger.info(
        "ai.call",
        purpose=purpose,
        model=served_by,
        latency_ms=latency_ms,
        cached_read=usage["cached_read_tokens"],
        cost_usd=str(cost),
    )

    return AiResult(
        data=data,
        meta=AiMeta(
            model=served_by,
            prompt_version=ai_prompts.PROMPT_VERSION,
            input_tokens=usage["input_tokens"],
            output_tokens=usage["output_tokens"],
            cost_usd=cost,
            latency_ms=latency_ms,
        ),
    )


async def quota_remaining(db: AsyncSession, identity: Identity) -> int | None:
    """How many AI calls are left today. `None` is unlimited.

    AI calls are metered separately from tool runs because they carry a real
    marginal cost. Exhausting the allowance returns the **rule-based result**,
    not a 402: the user still gets their answer, with a note. Blocking a whole
    tool because the enrichment allowance ran out would be a worse product and
    a worse upgrade prompt.

    The limit itself comes from `plan_quotas` through `FeatureService` (M20),
    which is also where the fail-open-on-Redis-outage behaviour now lives.
    """
    from app.services import feature_service

    state = await feature_service.check(db, identity, Metric.AI_CALLS_PER_DAY)
    return state.remaining


def _exhausted(remaining: int | None) -> bool:
    """`None` is unlimited, so it is never exhausted.

    A plain `remaining <= 0` would read `None` as falsy in some hands and raise
    a TypeError in others; naming the question stops both.
    """
    return remaining is not None and remaining <= 0


async def _consume_quota(db: AsyncSession, identity: Identity) -> None:
    """Count a call that has already happened.

    `record` rather than `consume`: the decision to allow was made before the
    model call, and a paid call that succeeded must be counted whether or not
    the allowance has since been reached.
    """
    from app.services import feature_service

    await feature_service.record(db, identity, Metric.AI_CALLS_PER_DAY)


def enrichment(
    db: AsyncSession,
    *,
    purpose: str,
    identity: Identity,
    variables: dict[str, Any],
    tool_slug: str,
    apply: Callable[[ToolOutput, dict[str, Any]], None],
    grounding: Callable[[ToolOutput], dict[str, Any]] | None = None,
    generate: Callable[..., Awaitable[AiResult | None]] = generate_json,
) -> Callable[[ToolOutput], Awaitable[AiMeta | None]]:
    """Build the `enrich` callable `run_tool` takes.

    Keeps a synthesis endpoint the same three lines as every other endpoint.
    `grounding` extracts the facts to hand the model — by default the whole
    deterministic result, which is what "the model argues about what the
    engine chose" means in practice — and `apply` merges the prose back in.

    An exhausted AI quota returns the rule result with a note rather than a
    402. The user still gets an answer; blocking the tool because the
    enrichment allowance ran out would be a worse product and a worse upgrade
    prompt.
    """

    async def enrich(output: ToolOutput) -> AiMeta | None:
        if _exhausted(await quota_remaining(db, identity)):
            output.warnings.append(
                ToolWarning(
                    level="info",
                    message=(
                        "AI analysis is unavailable — you have used today's allowance on "
                        "this plan. Everything above is the rule engine's own output and "
                        "is complete; only the written commentary is missing."
                    ),
                )
            )
            await _record(
                db,
                ai_prompts.REGISTRY[purpose],
                identity,
                tool_slug,
                AiOutcome.QUOTA_EXCEEDED,
                latency_ms=0,
            )
            return None

        facts = grounding(output) if grounding else _default_grounding(output)
        result = await generate(
            db,
            purpose=purpose,
            grounding=facts,
            variables=variables,
            identity=identity,
            tool_slug=tool_slug,
        )
        if result is None:
            return None

        apply(output, result.data)
        return result.meta

    return enrich


def chain(
    *enrichers: Callable[[ToolOutput], Awaitable[AiMeta | None]],
) -> Callable[[ToolOutput], Awaitable[AiMeta | None]]:
    """Run several enrichments over one result and report them as one.

    `run_tool` takes a single `enrich` and stores a single `AiMeta`, because a
    run has one source and one cost line. A tool that needs two passes — the
    Architect wants a grounded assessment *and* a roadmap, and they are two
    prompts answering two questions — composes them here rather than growing a
    second AI field on the wire shape.

    **Sequential, deliberately.** The passes share one `AsyncSession`, and
    concurrent writes on a single session are a race, not a speed-up. They
    also both draw on one daily request allowance, so nothing is saved by
    spending it faster.

    A pass that returns `None` is skipped, not fatal: partial enrichment is
    the normal outcome when an allowance runs out mid-run, and one written
    section is worth more than none. `None` comes back only when every pass
    failed, which is what keeps `source` honest — `hybrid` means at least one
    model actually contributed.
    """

    async def enrich(output: ToolOutput) -> AiMeta | None:
        metas = [meta for enricher in enrichers if (meta := await enricher(output)) is not None]
        if not metas:
            return None
        return _merged(metas)

    return enrich


def _merged(metas: list[AiMeta]) -> AiMeta:
    """One usage line from several calls.

    Tokens and cost add up; latency adds up too, because the passes ran one
    after another and the figure is meant to answer "how long did the AI part
    of this request take". Models are joined rather than picked: the tiers
    bill at different rates, so a row naming one of them would hide the other
    from anyone reconciling the ledger against an invoice.
    """
    return AiMeta(
        model="+".join(dict.fromkeys(meta.model for meta in metas)),
        prompt_version=ai_prompts.PROMPT_VERSION,
        input_tokens=sum(meta.input_tokens for meta in metas),
        output_tokens=sum(meta.output_tokens for meta in metas),
        cost_usd=sum((meta.cost_usd for meta in metas), Decimal(0)),
        latency_ms=sum(meta.latency_ms for meta in metas),
    )


#: How much word overlap makes two sentences the same advice. Chosen against
#: real output: a paraphrase of a grounded line lands above 0.65, and two
#: genuinely different recommendations about the same component land under
#: 0.4 even when they share the component's name.
_ECHO_THRESHOLD: Final = 0.6

#: Words that carry no signal about *what* is being said, so counting them
#: makes every pair of English sentences look alike.
_STOPWORDS: Final = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "for",
        "from",
        "has",
        "have",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "were",
        "will",
        "with",
        "you",
        "your",
    ]
)


def echoes(candidate: str, existing: str) -> bool:
    """Whether the model has restated something it was already shown.

    Grounding a prompt in the rule engine's own words is what keeps the model
    honest, and it is also an invitation to paraphrase them back. An exact
    match is easy to drop; the real output is a rewording, and a page showing
    the same advice twice in slightly different English reads as a bug in the
    tool rather than as emphasis.

    Overlap coefficient on content words — the intersection over the *smaller*
    of the two, not over the union. Jaccard was the first attempt and got this
    wrong in the case it exists for: the engine's rows are long and carry
    their own arithmetic, the model's restatement is one short sentence, and
    a short sentence entirely contained in a long one still scores under 0.2
    on Jaccard. Asking "is the shorter one already inside the longer one" is
    the actual question.

    Deliberately crude beyond that. The failure that matters is a near-copy,
    which scores far above anything genuinely new, so a cleverer measure would
    buy precision the decision does not use.
    """

    def words(text: str) -> set[str]:
        return {
            word
            for word in "".join(c.lower() if c.isalnum() else " " for c in text).split()
            if word not in _STOPWORDS and len(word) > 2
        }

    left, right = words(candidate), words(existing)
    if not left or not right:
        return False
    return len(left & right) / min(len(left), len(right)) >= _ECHO_THRESHOLD


def _default_grounding(output: ToolOutput) -> dict[str, Any]:
    """The deterministic result, as the model sees it.

    Artifacts are excluded on purpose: they are large, they are generated from
    the same metrics and tables, and including them would spend input tokens
    restating what the model has already been given.
    """
    return {
        "metrics": {key: str(value) for key, value in output.metrics.items()},
        "tables": output.tables,
        "warnings": [
            {"level": warning.level, "message": warning.message} for warning in output.warnings
        ],
    }


def _usage(response: BetaMessage) -> dict[str, int]:
    """Token counts, in this module's own vocabulary.

    The API already reports them the way the ledger wants them. `input_tokens`
    is the uncached remainder billed at the full rate; cache reads and cache
    writes arrive as their own figures rather than folded into it. Adding them
    back together would bill the cached prompt twice and make a working cache
    read as *more* expensive rather than less.

    `output_tokens` includes thinking, which is billed at the output rate. A
    short structured answer routinely costs several times the JSON it
    produces, and the ledger has to say so.

    The cache figures are optional on the wire and arrive as `None` when
    nothing was cached, which is the shape the reader has to survive.
    """
    usage = response.usage
    return {
        "input_tokens": max(usage.input_tokens, 0),
        "output_tokens": max(usage.output_tokens, 0),
        "cached_read_tokens": max(usage.cache_read_input_tokens or 0, 0),
        "cached_write_tokens": max(usage.cache_creation_input_tokens or 0, 0),
    }


def _answer_json(response: BetaMessage) -> dict[str, Any] | None:
    """The answer, with the model's own reasoning left out of it.

    Thinking arrives as its own content blocks ahead of the answer, and a
    refusal fallback adds a marker block where one model handed over to the
    next. Only `text` blocks are the answer: concatenating everything and
    parsing the result is what the first version of this reader did, and it
    fails the moment anything else is in the list.
    """
    text = "".join(block.text for block in response.content if isinstance(block, BetaTextBlock))
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


async def _fail(
    db: AsyncSession,
    prompt: ai_prompts.Prompt,
    identity: Identity,
    tool_slug: str | None,
    outcome: AiOutcome,
    started: float,
    exc: Exception,
) -> None:
    """Log and record. Callers return `None` themselves, so the contract that
    every failure path yields `None` is visible at the call site."""
    latency_ms = int((time.perf_counter() - started) * 1000)
    logger.warning(
        "ai.failed",
        purpose=prompt.purpose,
        outcome=outcome.value,
        error=type(exc).__name__,
        latency_ms=latency_ms,
    )
    await _record(
        db,
        prompt,
        identity,
        tool_slug,
        outcome,
        latency_ms=latency_ms,
        detail=f"{type(exc).__name__}: {exc}"[:500],
    )
    return None


async def _record(
    db: AsyncSession,
    prompt: ai_prompts.Prompt,
    identity: Identity,
    tool_slug: str | None,
    outcome: AiOutcome,
    *,
    latency_ms: int,
    usage: dict[str, int] | None = None,
    cost: Decimal = Decimal(0),
    detail: str | None = None,
    model: str | None = None,
) -> None:
    """Write the ledger row.

    Failures are logged too. A table that only records successes cannot answer
    "how often does this not work", which is the question this table exists
    for. Recording must never be the reason a request fails, so it is wrapped.

    `model` is the one that answered, when a response says so; a call that
    never got a response records the model it asked for.
    """
    counts = usage or {
        "input_tokens": 0,
        "output_tokens": 0,
        "cached_read_tokens": 0,
        "cached_write_tokens": 0,
    }
    try:
        db.add(
            AiCall(
                purpose=prompt.purpose,
                model=model or prompt.model,
                prompt_version=ai_prompts.PROMPT_VERSION,
                user_id=identity.user.id,
                tool_slug=tool_slug,
                outcome=outcome,
                latency_ms=latency_ms,
                cost_usd=cost,
                error_detail=detail,
                created_at=utcnow(),
                **counts,
            )
        )
        await db.flush()
    except Exception as exc:
        logger.warning("ai.record_failed", error=str(exc))
