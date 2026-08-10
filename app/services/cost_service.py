"""Cost Planner arithmetic. Four pure functions, one per P1 tool.

Each takes plain values plus already-fetched catalog rows and returns a
`ToolOutput`. No session, no request object, no I/O — so every figure in here
is testable by hand, which for a product whose whole claim is "these numbers
are right" is not optional.

Money is `Decimal` end to end. A float here would be a rounding error in
someone's budget approval.
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Final

from app.schemas.catalog import ModelOut
from app.schemas.tools import Artifact, ToolOutput, ToolWarning

# 365.25 / 12. Chosen over a flat 30 so that annual is exactly twelve months
# and the 12-month projection sums to the annual figure — a projection whose
# total disagrees with the headline is the fastest way to lose a reader.
DAYS_PER_MONTH: Final = Decimal("30.4375")
MONTHS_PER_YEAR: Final = Decimal(12)
THOUSAND: Final = Decimal(1000)

CENTS = Decimal("0.01")
MICRO = Decimal("0.000001")


def _money(value: Decimal) -> Decimal:
    """Six decimals. Per-request costs are routinely sub-cent."""
    return value.quantize(MICRO, rounding=ROUND_HALF_UP)


def _display(value: Decimal) -> Decimal:
    return value.quantize(CENTS, rounding=ROUND_HALF_UP)


# ── llm-pricing ──────────────────────────────────────────────────────────────


def cost_per_request(
    model: ModelOut,
    *,
    input_tokens: int,
    output_tokens: int,
    cached_input_ratio: Decimal = Decimal(0),
) -> Decimal:
    """One request's cost, with prompt caching accounted for.

    Caching is a first-class input because ignoring it makes every RAG and
    agent estimate wrong in the direction users care most about: a cached
    system prompt is routinely 80-90% of the input tokens and costs a tenth of
    the standard rate, so a calculator that cannot express it over-quotes such
    a workload by several times.
    """
    ratio = max(Decimal(0), min(Decimal(1), cached_input_ratio))
    cached_rate = model.cached_input_cost_per_1k

    if cached_rate is None:
        # No published cached rate means caching cannot be priced, not that it
        # is free. Charge the full rate and let the caller warn.
        effective_input_rate = model.input_cost_per_1k
    else:
        effective_input_rate = model.input_cost_per_1k * (1 - ratio) + cached_rate * ratio

    input_cost = Decimal(input_tokens) / THOUSAND * effective_input_rate
    output_rate = model.output_cost_per_1k or Decimal(0)
    output_cost = Decimal(output_tokens) / THOUSAND * output_rate
    return _money(input_cost + output_cost)


def llm_pricing(
    *,
    model: ModelOut,
    input_tokens: int,
    output_tokens: int,
    requests_per_day: int,
    cached_input_ratio: Decimal = Decimal(0),
    alternatives: list[ModelOut] | None = None,
) -> ToolOutput:
    per_request = cost_per_request(
        model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_input_ratio=cached_input_ratio,
    )
    daily = _money(per_request * requests_per_day)
    monthly = _money(daily * DAYS_PER_MONTH)
    annual = _money(monthly * MONTHS_PER_YEAR)

    warnings: list[ToolWarning] = []
    if cached_input_ratio > 0 and model.cached_input_cost_per_1k is None:
        warnings.append(
            ToolWarning(
                level="warning",
                field="cached_input_ratio",
                message=(
                    f"{model.display_name} has no published cached-input rate, so the "
                    f"caching discount was not applied. This estimate is an upper bound."
                ),
            )
        )

    context_needed = input_tokens + output_tokens
    if model.context_window and context_needed > model.context_window:
        warnings.append(
            ToolWarning(
                level="critical",
                message=(
                    f"{context_needed:,} tokens exceeds {model.display_name}'s "
                    f"{model.context_window:,}-token context window. These requests "
                    f"would fail."
                ),
            )
        )

    # Only models that can actually hold this workload are comparable. A
    # cheaper model with a window too small to run the job is not an
    # alternative, it is a different job.
    comparable = [
        candidate
        for candidate in (alternatives or [])
        if candidate.model_id != model.model_id
        and (candidate.context_window or 0) >= context_needed
    ]

    rows: list[dict[str, Any]] = []
    for candidate in comparable:
        candidate_monthly = _money(
            cost_per_request(
                candidate,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cached_input_ratio=cached_input_ratio,
            )
            * requests_per_day
            * DAYS_PER_MONTH
        )
        rows.append(
            {
                "model": candidate.display_name,
                "model_id": candidate.model_id,
                "provider": candidate.provider,
                "context_window": candidate.context_window,
                "monthly_cost": str(_display(candidate_monthly)),
                "delta_vs_selected": str(_display(candidate_monthly - monthly)),
                "pct_of_selected": (
                    str((candidate_monthly / monthly * 100).quantize(CENTS))
                    if monthly > 0
                    else "0.00"
                ),
            }
        )
    rows.sort(key=lambda row: Decimal(str(row["monthly_cost"])))

    projection = [{"month": month, "cost": str(_display(monthly))} for month in range(1, 13)]

    # Only the selected model. The alternatives table is derived data whose
    # figures are attributed per row; folding every candidate's source into
    # the headline provenance produced nine chips for a one-model answer and
    # took the freshness variant from whichever unrelated provider happened
    # to be oldest.
    sourced = [model.id]

    return ToolOutput(
        metrics={
            "cost_per_request": per_request,
            "daily_cost": daily,
            "monthly_cost": monthly,
            "annual_cost": annual,
            "requests_per_month": int(Decimal(requests_per_day) * DAYS_PER_MONTH),
            "tokens_per_month": int(
                Decimal(input_tokens + output_tokens) * requests_per_day * DAYS_PER_MONTH
            ),
            "model": model.display_name,
        },
        tables={"model_alternatives": rows},
        series={"cost_projection": projection},
        artifacts=[_cost_report(model, per_request, daily, monthly, annual, rows)],
        warnings=warnings,
        sourced_from=sourced,
    )


def _usd(amount: object) -> str:
    """`-$414.27`, not `$-414.27`. The sign belongs outside the symbol."""
    value = Decimal(str(amount))
    sign = "-" if value < 0 else ""
    return f"{sign}${abs(value):,.2f}"


def _cost_report(
    model: ModelOut,
    per_request: Decimal,
    daily: Decimal,
    monthly: Decimal,
    annual: Decimal,
    alternatives: list[dict[str, Any]],
) -> Artifact:
    lines = [
        f"# Cost estimate — {model.display_name}",
        "",
        f"- **Per request:** ${per_request}",
        f"- **Daily:** {_usd(daily)}",
        f"- **Monthly:** {_usd(monthly)}",
        f"- **Annual:** {_usd(annual)}",
        "",
        f"Pricing verified {model.provenance.last_verified_at.date()} "
        f"({model.provenance.source_name}).",
    ]
    if alternatives:
        lines += [
            "",
            "## Cheaper models that fit the same context",
            "",
            "| Model | Provider | Monthly | vs selected |",
            "| --- | --- | ---: | ---: |",
        ]
        lines += [
            f"| {row['model']} | {row['provider']} | {_usd(row['monthly_cost'])} "
            f"| {_usd(row['delta_vs_selected'])} |"
            for row in alternatives[:10]
        ]
    return Artifact(
        type="cost-report",
        format="markdown",
        filename=f"cost-estimate-{model.model_id}.md",
        content="\n".join(lines) + "\n",
    )


# ── token-calculator ─────────────────────────────────────────────────────────

# Heuristic constants for English prose. ~4 characters per token is the widely
# used rule of thumb; the word-based floor catches text with long tokens
# (code, URLs) that the character rule under-counts.
CHARS_PER_TOKEN: Final = Decimal(4)
TOKENS_PER_WORD: Final = Decimal("1.3")


def estimate_tokens(text: str) -> tuple[int, str]:
    """The fallback count, for callers with no model to count against.

    `tokenizer_service` owns real counting now and this is what it degrades to.
    Kept here because `method` is on the response either way: the person
    reaching for a token calculator is precisely the person who needs to know
    whether they are looking at a real count or an approximation.
    """
    if not text:
        return 0, "heuristic"
    characters = Decimal(len(text))
    words = Decimal(len(text.split()))
    estimate = max(characters / CHARS_PER_TOKEN, words * TOKENS_PER_WORD)
    return int(estimate.to_integral_value(rounding=ROUND_HALF_UP)), "heuristic"


def token_calculator(
    *,
    text: str,
    model: ModelOut,
    candidates: list[ModelOut] | None = None,
    output_tokens: int = 0,
    counted: tuple[int, str] | None = None,
) -> ToolOutput:
    """`counted` is `(tokens, method)` from `tokenizer_service`.

    Passed in rather than computed, because counting a Claude model means an
    API call and this function is deliberately pure — that purity is what
    makes every figure in it assertable by hand.
    """
    tokens, method = counted if counted is not None else estimate_tokens(text)
    total_needed = tokens + output_tokens

    window = model.context_window or 0
    fits = window >= total_needed if window else True
    usage_pct = (
        (Decimal(total_needed) / Decimal(window) * 100).quantize(CENTS) if window else Decimal(0)
    )
    overflow = max(0, total_needed - window) if window else 0

    warnings: list[ToolWarning] = []
    if method == "heuristic":
        warnings.append(
            ToolWarning(
                level="info",
                message=(
                    "Counted with a character-and-word heuristic, not the model's "
                    "tokenizer. Expect roughly +/-10%; treat it as a planning "
                    "figure rather than a billing one."
                ),
            )
        )
    if not fits:
        warnings.append(
            ToolWarning(
                level="critical",
                message=(
                    f"This input is {overflow:,} tokens over {model.display_name}'s "
                    f"{window:,}-token window."
                ),
            )
        )

    # Every model the input fits in, and what one request costs there. This is
    # the context-fit panel merged in rather than split into a second tool:
    # "which models fit this and at what cost" is one question.
    rows: list[dict[str, Any]] = []
    for candidate in candidates or []:
        candidate_window = candidate.context_window or 0
        candidate_fits = candidate_window >= total_needed
        rows.append(
            {
                "model": candidate.display_name,
                "model_id": candidate.model_id,
                "provider": candidate.provider,
                "context_window": candidate_window,
                "fits": candidate_fits,
                "context_used_pct": (
                    str((Decimal(total_needed) / Decimal(candidate_window) * 100).quantize(CENTS))
                    if candidate_window
                    else "0.00"
                ),
                "cost_per_call": str(
                    cost_per_request(candidate, input_tokens=tokens, output_tokens=output_tokens)
                ),
            }
        )
    rows.sort(key=lambda row: (not row["fits"], Decimal(str(row["cost_per_call"]))))

    return ToolOutput(
        metrics={
            "tokens": tokens,
            "method": method,
            "characters": len(text),
            "words": len(text.split()),
            "context_window": window,
            "context_used_pct": usage_pct,
            "fits": "yes" if fits else "no",
            "overflow_tokens": overflow,
            "cost_per_call": cost_per_request(
                model, input_tokens=tokens, output_tokens=output_tokens
            ),
        },
        tables={"context_fit": rows},
        warnings=warnings,
        sourced_from=[model.id],
    )


# ── embedding-cost ───────────────────────────────────────────────────────────


def embedding_cost(
    *,
    model: ModelOut,
    document_count: int,
    avg_tokens_per_document: int,
    reembeds_per_month: int = 1,
    chunk_overlap_pct: Decimal = Decimal(0),
    alternatives: list[ModelOut] | None = None,
) -> ToolOutput:
    """Ingestion and re-embedding cost.

    Chunk overlap inflates the token count for real: overlapping windows mean
    the same text is embedded more than once, and a 20% overlap is a 20% larger
    bill. Ignoring it under-quotes every chunked RAG pipeline, which is all of
    them.
    """
    overlap = max(Decimal(0), min(Decimal(1), chunk_overlap_pct / 100))
    base_tokens = document_count * avg_tokens_per_document
    effective_tokens = int(Decimal(base_tokens) * (1 + overlap))

    rate = model.input_cost_per_1k
    ingestion_cost = _money(Decimal(effective_tokens) / THOUSAND * rate)
    monthly_tokens = effective_tokens * reembeds_per_month
    monthly_cost = _money(Decimal(monthly_tokens) / THOUSAND * rate)

    rows: list[dict[str, Any]] = []
    for candidate in [model, *(alternatives or [])]:
        candidate_monthly = _money(Decimal(monthly_tokens) / THOUSAND * candidate.input_cost_per_1k)
        rows.append(
            {
                "model": candidate.display_name,
                "model_id": candidate.model_id,
                "provider": candidate.provider,
                "dimensions": candidate.dimensions,
                "cost_per_1k_tokens": str(candidate.input_cost_per_1k),
                "ingestion_cost": str(
                    _display(Decimal(effective_tokens) / THOUSAND * candidate.input_cost_per_1k)
                ),
                "monthly_cost": str(_display(candidate_monthly)),
                "selected": candidate.model_id == model.model_id,
            }
        )
    rows.sort(key=lambda row: Decimal(str(row["monthly_cost"])))

    warnings: list[ToolWarning] = []
    if model.dimensions and model.dimensions >= 3072:
        warnings.append(
            ToolWarning(
                level="info",
                message=(
                    f"{model.display_name} produces {model.dimensions}-dimension "
                    f"vectors. Embedding is cheap; storing and searching them is not "
                    f"— check the vector-database estimate before committing."
                ),
            )
        )
    if model.context_window and avg_tokens_per_document > model.context_window:
        warnings.append(
            ToolWarning(
                level="warning",
                message=(
                    f"Average document is {avg_tokens_per_document:,} tokens but "
                    f"{model.display_name} accepts {model.context_window:,}. "
                    f"Documents must be chunked before embedding."
                ),
            )
        )

    return ToolOutput(
        metrics={
            # `dimensions` is on the response so `vectordb-estimate` can consume
            # it directly — storage cost scales linearly with it.
            "dimensions": model.dimensions or 0,
            "total_tokens": effective_tokens,
            "monthly_tokens": monthly_tokens,
            "ingestion_cost": ingestion_cost,
            "monthly_cost": monthly_cost,
            "annual_cost": _money(monthly_cost * MONTHS_PER_YEAR),
            "cost_per_document": _money(
                ingestion_cost / document_count if document_count else Decimal(0)
            ),
        },
        tables={"provider_comparison": rows},
        warnings=warnings,
        sourced_from=[model.id],
    )


# ── budget-estimator ─────────────────────────────────────────────────────────


class WorkloadLine:
    """One line of a budget. Deliberately not a Pydantic model — this layer
    takes plain values so it stays trivially callable from a test."""

    __slots__ = ("input_tokens", "model", "name", "output_tokens", "requests_per_day")

    def __init__(
        self,
        *,
        name: str,
        model: ModelOut,
        requests_per_day: int,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        self.name = name
        self.model = model
        self.requests_per_day = requests_per_day
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


def budget_estimator(
    *,
    lines: list[WorkloadLine],
    monthly_growth_pct: Decimal = Decimal(0),
    infrastructure_monthly: Decimal = Decimal(0),
    embedding_monthly: Decimal = Decimal(0),
    user_count: int | None = None,
) -> ToolOutput:
    growth = monthly_growth_pct / 100

    breakdown: list[dict[str, Any]] = []
    llm_monthly = Decimal(0)
    for line in lines:
        per_request = cost_per_request(
            line.model,
            input_tokens=line.input_tokens,
            output_tokens=line.output_tokens,
        )
        line_monthly = _money(per_request * line.requests_per_day * DAYS_PER_MONTH)
        llm_monthly += line_monthly
        breakdown.append(
            {
                "name": line.name,
                "model": line.model.display_name,
                "model_id": line.model.model_id,
                "requests_per_day": line.requests_per_day,
                "cost_per_request": str(per_request),
                "monthly_cost": str(_display(line_monthly)),
                "_monthly": line_monthly,
            }
        )

    base_monthly = _money(llm_monthly + infrastructure_monthly + embedding_monthly)

    for row in breakdown:
        line_monthly = Decimal(str(row.pop("_monthly")))
        share = (
            (line_monthly / base_monthly * 100).quantize(CENTS) if base_monthly > 0 else Decimal(0)
        )
        row["pct_of_total"] = str(share)
    breakdown.sort(key=lambda row: Decimal(str(row["monthly_cost"])), reverse=True)

    # Month 1 is today's run rate; growth compounds from month 2.
    projection: list[dict[str, Any]] = []
    running_total = Decimal(0)
    for month in range(1, 13):
        value = _money(base_monthly * (1 + growth) ** (month - 1))
        running_total += value
        projection.append(
            {
                "month": month,
                "cost": str(_display(value)),
                "cumulative": str(_display(running_total)),
            }
        )

    month_12 = Decimal(str(projection[-1]["cost"]))
    recommendations = _budget_recommendations(lines, llm_monthly)

    metrics: dict[str, Decimal | int | str] = {
        "monthly_cost": base_monthly,
        "llm_monthly_cost": _money(llm_monthly),
        "infrastructure_monthly_cost": _money(infrastructure_monthly),
        "embedding_monthly_cost": _money(embedding_monthly),
        "month_12_cost": month_12,
        "year_1_total": _money(running_total),
        "workload_lines": len(lines),
    }
    if user_count:
        metrics["cost_per_user"] = _money(base_monthly / user_count)

    return ToolOutput(
        metrics=metrics,
        tables={"breakdown": breakdown, "recommendations": recommendations},
        series={"growth_projection": projection},
        warnings=_budget_warnings(base_monthly, growth),
        sourced_from=[line.model.id for line in lines],
    )


def _budget_recommendations(
    lines: list[WorkloadLine], llm_monthly: Decimal
) -> list[dict[str, Any]]:
    """Concrete, costed suggestions rather than generic advice.

    "Consider a cheaper model" is worthless; "this line is 68% of your bill and
    a cached prompt would cut it by $410/month" is actionable.
    """
    recommendations: list[dict[str, Any]] = []
    if llm_monthly <= 0:
        return recommendations

    for line in lines:
        line_monthly = _money(
            cost_per_request(
                line.model, input_tokens=line.input_tokens, output_tokens=line.output_tokens
            )
            * line.requests_per_day
            * DAYS_PER_MONTH
        )
        share = line_monthly / llm_monthly * 100

        if share >= 40 and line.model.cached_input_cost_per_1k and line.input_tokens >= 2000:
            cached = _money(
                cost_per_request(
                    line.model,
                    input_tokens=line.input_tokens,
                    output_tokens=line.output_tokens,
                    cached_input_ratio=Decimal("0.8"),
                )
                * line.requests_per_day
                * DAYS_PER_MONTH
            )
            recommendations.append(
                {
                    "line": line.name,
                    "kind": "caching",
                    "detail": (
                        f"{line.name} is {share.quantize(CENTS)}% of LLM spend and sends "
                        f"{line.input_tokens:,} input tokens per request. Caching a stable "
                        f"prompt prefix at 80% would cost ${_display(cached)}/month."
                    ),
                    "monthly_saving": str(_display(line_monthly - cached)),
                }
            )

        if line.requests_per_day >= 1000 and line.output_tokens <= 500:
            recommendations.append(
                {
                    "line": line.name,
                    "kind": "batching",
                    "detail": (
                        f"{line.name} runs {line.requests_per_day:,} short requests a day. "
                        f"If latency is not user-facing, the Batch API is half price."
                    ),
                    "monthly_saving": str(_display(line_monthly / 2)),
                }
            )
    return recommendations


def _budget_warnings(base_monthly: Decimal, growth: Decimal) -> list[ToolWarning]:
    warnings: list[ToolWarning] = []
    if growth >= Decimal("0.20"):
        warnings.append(
            ToolWarning(
                level="warning",
                field="monthly_growth_pct",
                message=(
                    f"At {(growth * 100).quantize(CENTS)}% monthly growth, spend "
                    f"multiplies by {((1 + growth) ** 12).quantize(CENTS)}x over a year. "
                    f"Check that against your actual funnel before planning on it."
                ),
            )
        )
    if base_monthly == 0:
        warnings.append(
            ToolWarning(level="info", message="Every workload line is zero-cost as configured.")
        )
    return warnings
