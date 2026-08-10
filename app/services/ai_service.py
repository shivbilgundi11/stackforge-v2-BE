"""The only Anthropic client in the process.

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
"""

from __future__ import annotations

import json
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any, Final, NamedTuple

import anthropic
from anthropic import AsyncAnthropic
from anthropic.types import (
    MessageParam,
    OutputConfigParam,
    TextBlockParam,
    ThinkingConfigAdaptiveParam,
)
from redis.exceptions import RedisError
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import Identity
from app.core.config import settings
from app.core.database import utcnow
from app.core.logging import get_logger
from app.core.redis import Keys, get_redis
from app.models.ai import AiCall, AiOutcome
from app.models.user import Plan
from app.schemas.tools import AiMeta, ToolOutput, ToolWarning
from app.services import ai_pricing, ai_prompts

logger = get_logger("ai")

#: AI calls are metered separately from tool runs because they carry a real
#: marginal cost. Exceeding this returns the **rule-based result**, not a 402:
#: the user still gets their answer with a note. Blocking a whole tool because
#: the enrichment allowance ran out would be a worse product and a worse
#: upgrade prompt.
DAILY_AI_LIMIT: Final[dict[str, int]] = {
    "anonymous": 1,
    Plan.FREE.value: 3,
    Plan.PRO.value: 100,
    Plan.TEAM.value: 300,
    Plan.ENTERPRISE.value: 2_000,
}

#: A synthesis call that has not answered in this long is not going to save the
#: request. The deterministic result is already computed and waiting.
TIMEOUT_SECONDS: Final = 60.0

_client: AsyncAnthropic | None = None


def get_client() -> AsyncAnthropic | None:
    """The process-wide client, or `None` when no key is configured.

    Built lazily so importing this module never needs a key — which is what
    lets the whole test suite run, and the app boot, with `ANTHROPIC_API_KEY`
    unset.
    """
    global _client
    if not settings.ai_enabled:
        return None
    if _client is None:
        _client = AsyncAnthropic(api_key=settings.anthropic_api_key, timeout=TIMEOUT_SECONDS)
    return _client


def set_client(client: AsyncAnthropic | None) -> None:
    """Test seam. Nothing in the app calls this."""
    global _client
    _client = client


class AiResult(NamedTuple):
    data: dict[str, Any]
    meta: AiMeta


def _plan_key(identity: Identity) -> str:
    return identity.plan.value if identity.is_authenticated else "anonymous"


def _period() -> tuple[str, datetime]:
    now = utcnow()
    tomorrow = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return now.strftime("%Y-%m-%d"), tomorrow.replace(tzinfo=UTC)


async def quota_remaining(identity: Identity) -> int:
    """Fail-open on a Redis outage, exactly as tool quota does.

    An unavailable cache must not stop people using the product; the trade is
    that an outage window is uncapped, which is the right way round.
    """
    period, _ = _period()
    limit = DAILY_AI_LIMIT.get(_plan_key(identity), DAILY_AI_LIMIT[Plan.FREE.value])
    try:
        raw = await get_redis().get(Keys.quota("ai_calls", identity.key, period))
        used = int(raw) if raw else 0
    except (RedisError, OSError, ValueError) as exc:
        logger.warning("ai.quota_read_failed", error=str(exc))
        return limit
    return max(0, limit - used)


async def _consume_quota(identity: Identity) -> None:
    period, resets_at = _period()
    key = Keys.quota("ai_calls", identity.key, period)
    try:
        redis = get_redis()
        value = await redis.incr(key)
        if value == 1:
            await redis.expireat(key, int(resets_at.timestamp()) + 60)
    except (RedisError, OSError) as exc:
        logger.warning("ai.quota_increment_failed", error=str(exc))


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

    if await quota_remaining(identity) <= 0:
        await _record(db, prompt, identity, tool_slug, AiOutcome.QUOTA_EXCEEDED, latency_ms=0)
        return None

    # The stable half of the prompt, marked cacheable. Byte-identical per
    # purpose, so the second call for a purpose reads it rather than paying
    # for it — which is why nothing variable may be interpolated into it.
    system: list[TextBlockParam] = [
        {
            "type": "text",
            "text": prompt.system,
            "cache_control": {"type": "ephemeral"},
        }
    ]
    # Adaptive thinking, depth controlled by effort. No `budget_tokens`
    # (removed — a 400 on these models), no `temperature`/`top_p`/`top_k`
    # (removed likewise), and no assistant prefill (also a 400).
    thinking: ThinkingConfigAdaptiveParam = {"type": "adaptive"}
    output_config: OutputConfigParam = {
        "effort": prompt.effort,  # type: ignore[typeddict-item]
        "format": {"type": "json_schema", "schema": prompt.schema},
    }
    messages: list[MessageParam] = [
        {"role": "user", "content": ai_prompts.user_turn(grounding, variables)}
    ]

    started = time.perf_counter()
    try:
        response = await client.messages.create(
            model=prompt.model,
            max_tokens=prompt.max_tokens,
            system=system,
            thinking=thinking,
            output_config=output_config,
            messages=messages,
        )
    except anthropic.APITimeoutError as exc:
        await _fail(db, prompt, identity, tool_slug, AiOutcome.TIMEOUT, started, exc)
        return None
    except anthropic.RateLimitError as exc:
        await _fail(db, prompt, identity, tool_slug, AiOutcome.RATE_LIMITED, started, exc)
        return None
    except anthropic.APIError as exc:
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

    # Safety classifiers can decline a request and still return a 200. Reading
    # `content[0]` without checking would raise on an empty list, which is the
    # one failure mode that would escape this function.
    if getattr(response, "stop_reason", None) == "refusal":
        await _record(
            db,
            prompt,
            identity,
            tool_slug,
            AiOutcome.REFUSAL,
            latency_ms=latency_ms,
            usage=usage,
            detail=_refusal_category(response),
        )
        await _consume_quota(identity)
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
            detail=f"stop_reason={getattr(response, 'stop_reason', None)}",
        )
        await _consume_quota(identity)
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
    await _consume_quota(identity)

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
        if await quota_remaining(identity) <= 0:
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
    """Token counts, with cached reads and writes kept separate.

    `input_tokens` from the API is the *uncached remainder*; folding the cached
    figures into it would make a working cache look like it cost more.
    """
    usage = getattr(response, "usage", None)
    return {
        "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
        "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
        "cached_read_tokens": int(getattr(usage, "cache_read_input_tokens", 0) or 0),
        "cached_write_tokens": int(getattr(usage, "cache_creation_input_tokens", 0) or 0),
    }


def _refusal_category(response: Any) -> str:
    details = getattr(response, "stop_details", None)
    return f"refusal:{getattr(details, 'category', None)}"


def _first_json(response: Any) -> dict[str, Any] | None:
    """The structured payload, or `None` if there is not one.

    Structured output guarantees the first text block is valid JSON matching
    the schema — but `max_tokens` truncation and refusals both produce a 200
    with something else, so this stays defensive.
    """
    for block in getattr(response, "content", None) or []:
        if getattr(block, "type", None) != "text":
            continue
        try:
            parsed = json.loads(block.text)
        except (json.JSONDecodeError, TypeError, AttributeError):
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
