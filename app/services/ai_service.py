"""The only Groq client in the process.

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

The provider is Groq, whose API is OpenAI-shaped: one `chat.completions`
call, the system prompt as the first message, `response_format` carrying the
schema, and `reasoning_effort` as the depth knob. Three consequences are worth
stating rather than discovering:

* **Reasoning tokens are billed as output.** Groq reports them inside
  `completion_tokens`, so `effort` is a direct lever on spend and every prompt
  in the registry sits at `low` or `medium` for that reason.
* **The prompt cache is automatic.** There is no `cache_control` marker to
  send and no way to ask for one; the provider reuses a recent shared prefix
  or it does not. That is why the stable-first message order below is the only
  thing this module does about caching — and why a run reporting zero cached
  tokens is a miss, not a bug.
* **`prompt_tokens` includes the cached part**, unlike the provider this
  module was originally written against. It is split back out in `_usage_of`,
  because `input_tokens` meaning "the uncached remainder" is what stops a
  working cache from reading as *more* expensive in the ledger.
"""

from __future__ import annotations

import json
import time
from collections.abc import Awaitable, Callable
from decimal import Decimal
from typing import Any, Final, NamedTuple

import groq
from groq import AsyncGroq
from groq.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
from groq.types.chat.completion_create_params import ResponseFormatResponseFormatJsonSchema
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

#: `finish_reason` values that mean the model declined rather than answered.
#: A declined request is a 200 with no usable content, so it has to be named
#: here or it arrives as "malformed output" and gets debugged as a bad schema.
_REFUSAL_REASONS: Final = frozenset({"content_filter", "refusal"})

_client: AsyncGroq | None = None


def get_client() -> AsyncGroq | None:
    """The process-wide client, or `None` when no key is configured.

    Built lazily so importing this module never needs a key — which is what
    lets the whole test suite run, and the app boot, with `GROQ_API_KEY`
    unset.
    """
    global _client
    if not settings.ai_enabled:
        return None
    if _client is None:
        _client = AsyncGroq(api_key=settings.groq_api_key, timeout=TIMEOUT_SECONDS)
    return _client


def set_client(client: AsyncGroq | None) -> None:
    """Test seam. Nothing in the app calls this."""
    global _client
    _client = client


class AiResult(NamedTuple):
    data: dict[str, Any]
    meta: AiMeta


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


def _response_format(prompt: ai_prompts.Prompt) -> ResponseFormatResponseFormatJsonSchema:
    """The schema, in the shape the provider enforces it.

    `strict` is the whole point. Without it the schema is a suggestion the
    model usually follows, and "usually" means a few percent of requests
    degrade to `rule_based` with nothing separating "the prompt is wrong" from
    "the model was down".
    """
    return {
        "type": "json_schema",
        "json_schema": {
            "name": prompt.purpose,
            "strict": True,
            "schema": prompt.schema,
        },
    }


def _is_throttled(exc: groq.APIStatusError) -> bool:
    """Whether a non-429 status is really a rate limit.

    Exceeding the per-minute **token** allowance returns `413 Request too
    large` with `code: rate_limit_exceeded` in the body — a payload-size
    status for a throughput problem. The SDK maps 413 to a plain status error,
    so without this the one failure an operator can actually act on (the
    prompt is too big for the tier) is recorded identically to the provider
    being down.

    The body code is what is trusted; the status is only a cheap prefilter, so
    a future status carrying the same code is classified correctly too.
    """
    if exc.status_code == 429:
        return True
    body: object = exc.body
    if not isinstance(body, dict):
        return False
    error: object = body.get("error")
    if not isinstance(error, dict):
        return False
    return str(error.get("code") or "") == "rate_limit_exceeded"


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

    The schema comes from the registry, never from the caller, and the response
    is requested as structured output rather than asked for in prose and
    parsed. Parsing prose JSON fails a few percent of the time, and each
    failure would silently degrade to `rule_based` with no signal separating
    "the prompt is wrong" from "the model was down".
    """
    prompt = ai_prompts.REGISTRY.get(purpose)
    if prompt is None:  # pragma: no cover — a programming error, not an input
        logger.error("ai.unknown_purpose", purpose=purpose)
        return None

    client = get_client()
    if client is None:
        await _record(db, prompt, identity, tool_slug, AiOutcome.DISABLED, latency_ms=0)
        return None

    if _exhausted(await quota_remaining(db, identity)):
        await _record(db, prompt, identity, tool_slug, AiOutcome.QUOTA_EXCEEDED, latency_ms=0)
        return None

    # The stable half of the prompt first, byte-identical per purpose; the
    # rule-engine output that varies per request second. That order is the
    # only lever there is on the automatic prefix cache — reversing it would
    # give consecutive requests no shared prefix at all. No assistant prefill:
    # the last turn must be the user's.
    system: ChatCompletionSystemMessageParam = {"role": "system", "content": prompt.system}
    user: ChatCompletionUserMessageParam = {
        "role": "user",
        "content": ai_prompts.user_turn(grounding, variables),
    }
    messages: list[ChatCompletionMessageParam] = [system, user]

    started = time.perf_counter()
    try:
        response = await client.chat.completions.create(
            model=prompt.model,
            # `max_completion_tokens`, not `max_tokens`: the latter is
            # deprecated on this API and does not bound reasoning tokens,
            # which is the half of the output that actually runs away.
            max_completion_tokens=prompt.max_tokens,
            reasoning_effort=prompt.effort,
            response_format=_response_format(prompt),
            messages=messages,
        )
    except groq.APITimeoutError as exc:
        await _fail(db, prompt, identity, tool_slug, AiOutcome.TIMEOUT, started, exc)
        return None
    except groq.RateLimitError as exc:
        await _fail(db, prompt, identity, tool_slug, AiOutcome.RATE_LIMITED, started, exc)
        return None
    except groq.APIStatusError as exc:
        # Not all throttling arrives as a 429 (see `_is_throttled`), and the
        # ledger exists to answer "why does this not work" — a rate limit
        # filed under `api_error` sends that investigation to the wrong place.
        outcome = AiOutcome.RATE_LIMITED if _is_throttled(exc) else AiOutcome.API_ERROR
        await _fail(db, prompt, identity, tool_slug, outcome, started, exc)
        return None
    except groq.APIError as exc:
        await _fail(db, prompt, identity, tool_slug, AiOutcome.API_ERROR, started, exc)
        return None
    except Exception as exc:
        # Deliberately last and deliberately broad. The contract is that
        # nothing from a model call escapes this function, and a contract that
        # only covers the exceptions we thought of is not one.
        await _fail(db, prompt, identity, tool_slug, AiOutcome.API_ERROR, started, exc)
        return None

    latency_ms = int((time.perf_counter() - started) * 1000)
    usage = _usage_of(response)
    finish_reason = _finish_reason(response)

    # Safety classifiers can decline a request and still return a 200. Reading
    # `choices[0].message.content` without checking would raise on an empty
    # list, which is the one failure mode that would escape this function.
    if finish_reason in _REFUSAL_REASONS:
        await _record(
            db,
            prompt,
            identity,
            tool_slug,
            AiOutcome.REFUSAL,
            latency_ms=latency_ms,
            usage=usage,
            detail=f"refusal:{finish_reason}",
        )
        await _consume_quota(db, identity)
        return None

    data = _first_json(response)
    if data is None:
        await _record(
            db,
            prompt,
            identity,
            tool_slug,
            AiOutcome.INVALID_OUTPUT,
            latency_ms=latency_ms,
            usage=usage,
            detail=f"finish_reason={finish_reason}",
        )
        await _consume_quota(db, identity)
        return None

    cost = ai_pricing.cost_of(model=prompt.model, **usage)
    await _record(
        db,
        prompt,
        identity,
        tool_slug,
        AiOutcome.SUCCESS,
        latency_ms=latency_ms,
        usage=usage,
        cost=cost,
    )
    await _consume_quota(db, identity)

    logger.info(
        "ai.call",
        purpose=purpose,
        model=prompt.model,
        latency_ms=latency_ms,
        cached_read=usage["cached_read_tokens"],
        cost_usd=str(cost),
    )

    return AiResult(
        data=data,
        meta=AiMeta(
            model=prompt.model,
            prompt_version=ai_prompts.PROMPT_VERSION,
            input_tokens=usage["input_tokens"],
            output_tokens=usage["output_tokens"],
            cost_usd=cost,
            latency_ms=latency_ms,
        ),
    )


def enrichment(
    db: AsyncSession,
    *,
    purpose: str,
    identity: Identity,
    variables: dict[str, Any],
    tool_slug: str,
    apply: Callable[[ToolOutput, dict[str, Any]], None],
    grounding: Callable[[ToolOutput], dict[str, Any]] | None = None,
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
        result = await generate_json(
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


def _usage_of(response: Any) -> dict[str, int]:
    """Token counts, in this module's own vocabulary.

    Groq reports OpenAI-shaped names — `prompt_tokens` and `completion_tokens`
    — and `completion_tokens` already includes reasoning tokens, which is the
    figure that matters because they are billed at the output rate.

    `prompt_tokens` is the **whole** prompt, cached part included, so the
    cached count is subtracted back out: downstream, `input_tokens` means the
    uncached remainder billed at the full rate, and folding the two together
    would make a working cache look like it cost more rather than less. The
    subtraction is clamped, because a provider figure that exceeds the total
    it is part of should degrade to zero rather than bill a negative.

    `cached_write_tokens` is always zero here. Populating the cache on this
    provider is automatic and carries no surcharge, so there is nothing to
    count; the field stays because the ledger has to be able to describe a
    provider that does charge for it.
    """
    usage = getattr(response, "usage", None)
    details = getattr(usage, "prompt_tokens_details", None)

    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    cached = min(int(getattr(details, "cached_tokens", 0) or 0), prompt_tokens)

    return {
        "input_tokens": prompt_tokens - cached,
        "output_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "cached_read_tokens": cached,
        "cached_write_tokens": 0,
    }


def _finish_reason(response: Any) -> str | None:
    for choice in getattr(response, "choices", None) or []:
        return str(getattr(choice, "finish_reason", None) or "") or None
    return None


def _first_json(response: Any) -> dict[str, Any] | None:
    """The structured payload, or `None` if there is not one.

    Structured output guarantees the message content is valid JSON matching
    the schema — but a `length` stop and a refusal both produce a 200 with
    something else, so this stays defensive.
    """
    for choice in getattr(response, "choices", None) or []:
        content = getattr(getattr(choice, "message", None), "content", None)
        # A refusal arrives as a null content with a 200. Named here as well as
        # in the refusal branch, because a `finish_reason` this module has not
        # learned about yet must still land on "no JSON", never on a TypeError.
        if not isinstance(content, str):
            return None
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


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
) -> None:
    """Write the ledger row.

    Failures are logged too. A table that only records successes cannot answer
    "how often does this not work", which is the question this table exists
    for. Recording must never be the reason a request fails, so it is wrapped.
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
                model=prompt.model,
                prompt_version=ai_prompts.PROMPT_VERSION,
                user_id=identity.user.id if identity.user else None,
                anonymous_session_id=None if identity.user else identity.anonymous_id,
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
