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

The provider is **Gemini**, everywhere. It was Groq for everything except the
Architect, and running two providers meant two request shapes, two failure
taxonomies, and two sets of quota arithmetic to reason about before answering
"why did this come back rule_based". One provider is the point.

Four consequences of this API are worth stating rather than discovering:

* **Thinking tokens are billed as output and are reported separately.**
  `candidatesTokenCount` excludes them; `thoughtsTokenCount` holds them. They
  are folded together in `_gemini_usage`, because a ledger that reports only
  the visible half of what it paid for is understating cost by more than the
  visible half on a short answer.
* **They also come out of `maxOutputTokens`.** A reservation that thinking
  exhausts returns a 200 with `finishReason: MAX_TOKENS` and *no parts at
  all* — not a truncated answer, an empty one. That is why the reservations
  in `ai_prompts` are sized against the thinking budget rather than against
  the length of the prose.
* **`thinkingLevel` is the depth knob**, and it is the direct lever on both
  latency and spend. Every prompt in the registry sits at `low` or `medium`
  for that reason.
* **The free tier is metered in requests per day, per model** — 20 of them,
  not 20 per minute. That is why `ai_prompts` tiers across two models rather
  than pointing every prompt at one: the allowances are separate, so tiering
  is the difference between a product that works all day and one that stops
  after the twentieth request.
"""

from __future__ import annotations

import json
import time
from collections.abc import Awaitable, Callable
from decimal import Decimal
from typing import Any, Final, NamedTuple

import httpx
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
GEMINI_GENERATE_CONTENT_URL: Final = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)

#: `finishReason` values that mean the model declined rather than answered.
#: A declined request is a 200 with no usable content, so it has to be named
#: here or it arrives as "malformed output" and gets debugged as a bad schema.
_REFUSAL_REASONS: Final = frozenset(
    {"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "IMAGE_SAFETY", "RECITATION"}
)

#: The one that is neither a refusal nor a bad schema: the reservation ran out.
#: On this provider that is a 200 carrying an empty `parts` list, which reads
#: as malformed output unless it is named.
_TRUNCATED: Final = "MAX_TOKENS"


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
    response is requested as structured output rather than asked for in prose
    and parsed. Parsing prose JSON fails a few percent of the time, and each
    failure would silently degrade to `rule_based` with no signal separating
    "the prompt is wrong" from "the model was down".

    The stable half of the prompt goes in `systemInstruction`, byte-identical
    per purpose, and the rule-engine output that varies per request goes in
    the user turn after it. That order is the only lever there is on implicit
    context caching, which is automatic here — there is no marker to send and
    no way to ask for one.
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
    payload = {
        "systemInstruction": {"parts": [{"text": prompt.system}]},
        "contents": [
            {"role": "user", "parts": [{"text": ai_prompts.user_turn(grounding, variables)}]}
        ],
        "generationConfig": {
            "maxOutputTokens": prompt.max_tokens,
            "responseMimeType": "application/json",
            "responseJsonSchema": prompt.schema,
            # Thinking is billed as output and comes out of the reservation
            # above, so this is a spend lever and a truncation risk at once.
            "thinkingConfig": {"thinkingLevel": prompt.effort},
        },
    }
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS) as client:
            response = await client.post(
                GEMINI_GENERATE_CONTENT_URL.format(model=prompt.model),
                headers={"x-goog-api-key": settings.gemini_api_key},
                json=payload,
            )
            response.raise_for_status()
            body = response.json()
    except httpx.TimeoutException as exc:
        await _fail(db, prompt, identity, tool_slug, AiOutcome.TIMEOUT, started, exc)
        return None
    except httpx.HTTPStatusError as exc:
        # 429 is the daily request allowance on the free tier, and it is the
        # one failure an operator can act on. Filed under `api_error` it would
        # send that investigation to the wrong place entirely.
        outcome = AiOutcome.RATE_LIMITED if exc.response.status_code == 429 else AiOutcome.API_ERROR
        await _fail(db, prompt, identity, tool_slug, outcome, started, exc)
        return None
    except (httpx.HTTPError, ValueError) as exc:
        await _fail(db, prompt, identity, tool_slug, AiOutcome.API_ERROR, started, exc)
        return None
    except Exception as exc:
        # Deliberately last and deliberately broad. The contract is that
        # nothing from a model call escapes this function, and a contract that
        # only covers the exceptions we thought of is not one.
        await _fail(db, prompt, identity, tool_slug, AiOutcome.API_ERROR, started, exc)
        return None

    latency_ms = int((time.perf_counter() - started) * 1000)
    usage = _gemini_usage(body)
    finish_reason = _finish_reason(body)

    # A safety classifier can decline and still return a 200 with no content.
    # Reading the parts without checking would report the refusal as a schema
    # failure, which is the wrong thing to go and debug.
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

    data = _gemini_json(body)
    if data is None:
        await _record(
            db,
            prompt,
            identity,
            tool_slug,
            AiOutcome.INVALID_OUTPUT,
            latency_ms=latency_ms,
            usage=usage,
            # `MAX_TOKENS` here means thinking ate the reservation and the
            # answer never started. Recording the reason is the difference
            # between raising `max_tokens` and rewriting a schema.
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


def _gemini_usage(body: dict[str, Any]) -> dict[str, int]:
    """Token counts, in this module's own vocabulary.

    `candidatesTokenCount` is the visible answer only. Thinking is reported
    separately as `thoughtsTokenCount` and is billed at the **output** rate,
    so the two are added: a ledger that counted only the visible half would
    understate a short structured answer by more than it counted, because
    reasoning routinely runs several times the length of the JSON it produces.

    `promptTokenCount` includes the implicitly cached part, so the cached
    count is subtracted back out — downstream, `input_tokens` means the
    uncached remainder billed at the full rate, and folding the two together
    would make a working cache read as *more* expensive rather than less. The
    subtraction is clamped, because a provider figure that exceeds the total
    it is part of should degrade to zero rather than bill a negative.

    `cached_write_tokens` is always zero. Context caching here is implicit and
    carries no surcharge for populating it, so there is nothing to charge.
    """
    usage = body.get("usageMetadata")
    if not isinstance(usage, dict):
        usage = {}
    prompt_tokens = max(int(usage.get("promptTokenCount") or 0), 0)
    cached = min(max(int(usage.get("cachedContentTokenCount") or 0), 0), prompt_tokens)
    answer = max(int(usage.get("candidatesTokenCount") or 0), 0)
    thoughts = max(int(usage.get("thoughtsTokenCount") or 0), 0)
    return {
        "input_tokens": prompt_tokens - cached,
        "output_tokens": answer + thoughts,
        "cached_read_tokens": cached,
        "cached_write_tokens": 0,
    }


def _finish_reason(body: dict[str, Any]) -> str | None:
    candidates = body.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        # No candidate at all is how a prompt blocked before generation
        # arrives. Reporting the block reason keeps it out of the bucket
        # labelled "the model returned something we could not parse".
        feedback = body.get("promptFeedback")
        if isinstance(feedback, dict) and feedback.get("blockReason"):
            return str(feedback["blockReason"])
        return None
    first = candidates[0]
    if not isinstance(first, dict):
        return None
    reason = first.get("finishReason")
    return str(reason) if reason else None


def _gemini_json(body: dict[str, Any]) -> dict[str, Any] | None:
    """The answer, with the model's own reasoning left out of it.

    Thinking arrives as extra `parts` on the same candidate, marked `thought`.
    Concatenating every part and parsing the result is what the first version
    did, and it fails the moment the model narrates before answering — the
    JSON is valid and the string it is glued to is not.
    """
    candidates = body.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        return None
    first = candidates[0]
    if not isinstance(first, dict):
        return None
    content = first.get("content")
    if not isinstance(content, dict):
        return None
    parts = content.get("parts")
    if not isinstance(parts, list):
        return None
    text = "".join(
        str(part.get("text") or "")
        for part in parts
        if isinstance(part, dict) and not part.get("thought")
    )
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
