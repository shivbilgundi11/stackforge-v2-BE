"""Cost arithmetic, verified by hand.

Every assertion is a computed value. The point of making `compute` a pure
function was to make this file possible: no database, no fixtures, no client —
just numbers you can check with a calculator.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from app.schemas.catalog import ModelOut, ProvenanceOut
from app.services import cost_service
from app.services.cost_service import WorkloadLine

PROVENANCE = ProvenanceOut(
    last_verified_at=datetime(2026, 8, 9, tzinfo=UTC),
    age_days=0,
    variant="fresh",
    source_name="OpenAI - API pricing",
    source_url="https://developers.openai.com/api/docs/pricing",
    source_kind="scrape",
)


def model(
    *,
    model_id: str = "gpt-4o-mini",
    input_per_1k: str = "0.000150",
    output_per_1k: str | None = "0.000600",
    cached_per_1k: str | None = "0.000075",
    context_window: int | None = 128_000,
    dimensions: int | None = None,
    family: str = "chat",
) -> ModelOut:
    return ModelOut(
        id=f"mdl_{model_id}",
        provider="openai",
        model_id=model_id,
        display_name=model_id,
        family=family,
        input_cost_per_1k=Decimal(input_per_1k),
        output_cost_per_1k=Decimal(output_per_1k) if output_per_1k else None,
        cached_input_cost_per_1k=Decimal(cached_per_1k) if cached_per_1k else None,
        context_window=context_window,
        max_output_tokens=16_384,
        dimensions=dimensions,
        capabilities={"vision": True, "tools": True, "thinking": False},
        tokenizer="tiktoken:o200k_base",
        status="active",
        provenance=PROVENANCE,
    )


# ── llm-pricing ──────────────────────────────────────────────────────────────


def test_cost_per_request_is_exact() -> None:
    """1000 in x $0.00015/1k + 500 out x $0.0006/1k = 0.00015 + 0.0003."""
    assert cost_service.cost_per_request(model(), input_tokens=1000, output_tokens=500) == Decimal(
        "0.000450"
    )


def test_llm_pricing_daily_monthly_annual() -> None:
    result = cost_service.llm_pricing(
        model=model(), input_tokens=1000, output_tokens=500, requests_per_day=100
    )

    assert result.metrics["cost_per_request"] == Decimal("0.000450")
    assert result.metrics["daily_cost"] == Decimal("0.045000")
    # 0.045 * 30.4375 days
    assert result.metrics["monthly_cost"] == Decimal("1.369688")
    # monthly * 12, so the annual figure and the projection agree
    assert result.metrics["annual_cost"] == Decimal("16.436256")


def test_annual_is_exactly_twelve_months() -> None:
    """A projection whose total disagrees with the headline destroys trust."""
    result = cost_service.llm_pricing(
        model=model(), input_tokens=4000, output_tokens=1200, requests_per_day=5000
    )
    monthly = result.metrics["monthly_cost"]
    assert result.metrics["annual_cost"] == (monthly * 12).quantize(Decimal("0.000001"))


def test_cached_input_reduces_cost_by_the_exact_expected_amount() -> None:
    """80% cached at half rate: input drops from 0.00015 to 0.000105 per 1k.

    0.2 * 0.00015 + 0.8 * 0.000075 = 0.00003 + 0.00006 = 0.00009 per 1k.
    Over 10k input tokens that is 0.0009, plus 500 output at 0.0006/1k = 0.0003.
    """
    cached = cost_service.cost_per_request(
        model(),
        input_tokens=10_000,
        output_tokens=500,
        cached_input_ratio=Decimal("0.8"),
    )
    assert cached == Decimal("0.001200")

    uncached = cost_service.cost_per_request(model(), input_tokens=10_000, output_tokens=500)
    assert uncached == Decimal("0.001800")
    assert cached < uncached


def test_caching_without_a_published_rate_warns_and_does_not_discount() -> None:
    """Silently applying a discount that does not exist would under-quote."""
    no_cache = model(model_id="mistral-large", cached_per_1k=None)
    result = cost_service.llm_pricing(
        model=no_cache,
        input_tokens=10_000,
        output_tokens=500,
        requests_per_day=100,
        cached_input_ratio=Decimal("0.9"),
    )

    assert result.metrics["cost_per_request"] == cost_service.cost_per_request(
        no_cache, input_tokens=10_000, output_tokens=500
    )
    assert any("no published cached-input rate" in w.message for w in result.warnings)


def test_alternatives_exclude_models_whose_window_is_too_small() -> None:
    """A cheaper model that cannot hold the prompt is not an alternative."""
    big = model(model_id="big", input_per_1k="0.005000", context_window=1_000_000)
    small = model(model_id="small", input_per_1k="0.000010", context_window=8_000)

    result = cost_service.llm_pricing(
        model=big,
        input_tokens=200_000,
        output_tokens=1000,
        requests_per_day=10,
        alternatives=[small, big],
    )
    listed = {row["model_id"] for row in result.tables["model_alternatives"]}
    assert "small" not in listed


def test_exceeding_the_context_window_is_critical() -> None:
    result = cost_service.llm_pricing(
        model=model(context_window=8_000),
        input_tokens=10_000,
        output_tokens=500,
        requests_per_day=1,
    )
    critical = [w for w in result.warnings if w.level == "critical"]
    assert critical and "exceeds" in critical[0].message


def test_sub_cent_pricing_never_rounds_to_zero() -> None:
    """GPT-5 nano: $0.00005/1k in. One 100-token request must not cost $0.00."""
    nano = model(model_id="gpt-5-nano", input_per_1k="0.000050", output_per_1k="0.000400")
    per_request = cost_service.cost_per_request(nano, input_tokens=100, output_tokens=50)

    assert per_request > 0
    assert per_request == Decimal("0.000025")


def test_a_cost_report_artifact_is_generated() -> None:
    result = cost_service.llm_pricing(
        model=model(), input_tokens=1000, output_tokens=500, requests_per_day=100
    )
    artifact = result.artifacts[0]

    assert artifact.format == "markdown"
    assert artifact.filename.endswith(".md")
    assert "Cost estimate" in artifact.content
    assert "2026-08-09" in artifact.content  # the provenance date, not today's


def test_sourced_from_carries_the_rows_that_were_read() -> None:
    """`run_tool` turns this into the provenance block."""
    other = model(model_id="gpt-5-mini")
    result = cost_service.llm_pricing(
        model=model(),
        input_tokens=100,
        output_tokens=100,
        requests_per_day=1,
        alternatives=[other],
    )
    assert "mdl_gpt-4o-mini" in result.sourced_from
    assert "mdl_gpt-5-mini" in result.sourced_from


# ── token-calculator ─────────────────────────────────────────────────────────


def test_token_estimate_reports_its_method() -> None:
    tokens, method = cost_service.estimate_tokens("hello world " * 100)
    assert tokens > 0
    assert method == "heuristic"


def test_empty_text_is_zero_tokens() -> None:
    assert cost_service.estimate_tokens("") == (0, "heuristic")


def test_token_calculator_reports_heuristic_honestly() -> None:
    """The person using a token calculator needs to know how much to trust it."""
    result = cost_service.token_calculator(text="hello world", model=model())

    assert result.metrics["method"] == "heuristic"
    assert any("heuristic" in w.message for w in result.warnings)


def test_context_overflow_reports_fits_false_and_the_amount() -> None:
    text = "x" * 40_000  # ~10,000 tokens at 4 chars each
    result = cost_service.token_calculator(
        text=text, model=model(context_window=8_000), output_tokens=0
    )

    assert result.metrics["fits"] == "no"
    assert result.metrics["overflow_tokens"] == 2000
    assert any(w.level == "critical" for w in result.warnings)


def test_context_fit_table_lists_every_candidate_with_a_verdict() -> None:
    small = model(model_id="small", context_window=4_000)
    large = model(model_id="large", context_window=1_000_000)
    text = "x" * 40_000  # ~10,000 tokens

    result = cost_service.token_calculator(text=text, model=large, candidates=[small, large])
    fits = {row["model_id"]: row["fits"] for row in result.tables["context_fit"]}

    assert fits == {"small": False, "large": True}


# ── embedding-cost ───────────────────────────────────────────────────────────


def test_embedding_monthly_tokens_is_exact() -> None:
    """1000 docs x 800 tokens x 2 runs = 1,600,000 tokens."""
    embed = model(
        model_id="text-embedding-3-small",
        family="embedding",
        input_per_1k="0.000020",
        output_per_1k=None,
        cached_per_1k=None,
        dimensions=1536,
        context_window=8191,
    )
    result = cost_service.embedding_cost(
        model=embed,
        document_count=1000,
        avg_tokens_per_document=800,
        reembeds_per_month=2,
    )

    assert result.metrics["total_tokens"] == 800_000
    assert result.metrics["monthly_tokens"] == 1_600_000
    # 1,600,000 / 1000 * 0.00002
    assert result.metrics["monthly_cost"] == Decimal("0.032000")
    assert result.metrics["ingestion_cost"] == Decimal("0.016000")


def test_embedding_returns_dimensions_for_the_vector_db_estimate() -> None:
    embed = model(family="embedding", output_per_1k=None, dimensions=3072)
    result = cost_service.embedding_cost(
        model=embed, document_count=10, avg_tokens_per_document=100
    )
    assert result.metrics["dimensions"] == 3072


def test_chunk_overlap_inflates_the_token_count() -> None:
    """Overlapping windows embed the same text twice; the bill reflects it."""
    embed = model(family="embedding", output_per_1k=None, dimensions=1536)
    plain = cost_service.embedding_cost(
        model=embed, document_count=1000, avg_tokens_per_document=800
    )
    overlapped = cost_service.embedding_cost(
        model=embed,
        document_count=1000,
        avg_tokens_per_document=800,
        chunk_overlap_pct=Decimal(20),
    )

    assert plain.metrics["total_tokens"] == 800_000
    assert overlapped.metrics["total_tokens"] == 960_000


def test_documents_larger_than_the_window_warn() -> None:
    embed = model(family="embedding", output_per_1k=None, context_window=8191)
    result = cost_service.embedding_cost(
        model=embed, document_count=10, avg_tokens_per_document=50_000
    )
    assert any("chunked" in w.message for w in result.warnings)


# ── budget-estimator ─────────────────────────────────────────────────────────


def _lines() -> list[WorkloadLine]:
    return [
        WorkloadLine(
            name="chat",
            model=model(),
            requests_per_day=1000,
            input_tokens=1000,
            output_tokens=500,
        ),
        WorkloadLine(
            name="summaries",
            model=model(model_id="gpt-5-mini", input_per_1k="0.000250", output_per_1k="0.002000"),
            requests_per_day=200,
            input_tokens=4000,
            output_tokens=800,
        ),
        WorkloadLine(
            name="classification",
            model=model(model_id="gpt-5-nano", input_per_1k="0.000050", output_per_1k="0.000400"),
            requests_per_day=5000,
            input_tokens=500,
            output_tokens=50,
        ),
    ]


def test_budget_sums_the_lines() -> None:
    result = cost_service.budget_estimator(lines=_lines())

    #   chat:  (1000/1000*0.00015 + 500/1000*0.0006) * 1000 * 30.4375 = 13.696875
    #   summ:  (4000/1000*0.00025 + 800/1000*0.002)  *  200 * 30.4375 = 15.827500
    #   class: (500/1000*0.00005  +  50/1000*0.0004) * 5000 * 30.4375 =  6.848438
    assert result.metrics["monthly_cost"] == Decimal("36.372813")
    assert result.metrics["workload_lines"] == 3


def test_growth_compounds_from_month_two() -> None:
    """Month 1 is today's run rate; month 12 is base * 1.1^11."""
    result = cost_service.budget_estimator(lines=_lines(), monthly_growth_pct=Decimal(10))
    base = Decimal("36.372813")
    expected = (base * Decimal("1.1") ** 11).quantize(Decimal("0.000001"))

    assert result.metrics["month_12_cost"] == expected.quantize(Decimal("0.01"))
    assert result.series["growth_projection"][0]["cost"] == "36.37"
    assert len(result.series["growth_projection"]) == 12


def test_cumulative_projection_reaches_the_year_total() -> None:
    result = cost_service.budget_estimator(lines=_lines(), monthly_growth_pct=Decimal(5))
    last = result.series["growth_projection"][-1]
    assert Decimal(last["cumulative"]) == result.metrics["year_1_total"].quantize(Decimal("0.01"))


def test_infrastructure_and_embedding_lines_are_included() -> None:
    result = cost_service.budget_estimator(
        lines=_lines(),
        infrastructure_monthly=Decimal(500),
        embedding_monthly=Decimal(25),
    )
    assert result.metrics["monthly_cost"] == Decimal("561.372813")
    assert result.metrics["llm_monthly_cost"] == Decimal("36.372813")


def test_cost_per_user_when_a_user_count_is_given() -> None:
    result = cost_service.budget_estimator(lines=_lines(), user_count=100)
    assert result.metrics["cost_per_user"] == Decimal("0.363728")


def test_breakdown_shares_add_up() -> None:
    result = cost_service.budget_estimator(lines=_lines())
    shares = sum(Decimal(row["pct_of_total"]) for row in result.tables["breakdown"])
    assert abs(shares - 100) < Decimal("0.05")


def test_caching_recommendation_is_costed_not_generic() -> None:
    """ "Consider caching" is worthless; a dollar figure is actionable."""
    heavy = [
        WorkloadLine(
            name="rag",
            model=model(),
            requests_per_day=5000,
            input_tokens=12_000,
            output_tokens=400,
        )
    ]
    result = cost_service.budget_estimator(lines=heavy)
    caching = [row for row in result.tables["recommendations"] if row["kind"] == "caching"]

    assert caching, "a 12k-token prompt at 5k requests/day should suggest caching"
    assert Decimal(caching[0]["monthly_saving"]) > 0


def test_aggressive_growth_is_flagged() -> None:
    result = cost_service.budget_estimator(lines=_lines(), monthly_growth_pct=Decimal(30))
    assert any(w.field == "monthly_growth_pct" for w in result.warnings)


@pytest.mark.parametrize("ratio", [Decimal("-1"), Decimal("5")])
def test_cached_ratio_is_clamped(ratio: Decimal) -> None:
    """Out-of-range input must not produce a negative or inflated price."""
    cost = cost_service.cost_per_request(
        model(), input_tokens=1000, output_tokens=0, cached_input_ratio=ratio
    )
    assert Decimal("0.000075") <= cost <= Decimal("0.000150")
