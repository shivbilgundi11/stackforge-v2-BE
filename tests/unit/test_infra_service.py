"""Infra arithmetic, against hand-computed values.

The VRAM figures below were worked out from the architecture constants, not
read off the implementation. Anyone can check them against `nvidia-smi`, which
is exactly why they have to be right.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from app.data.architectures_seed import BY_KEY
from app.schemas.catalog import GpuOut, ModelOut, ProvenanceOut
from app.services.infra_service import (
    cloud_cost,
    gpu_cost,
    kv_bytes_per_token,
    max_context_for,
    readiness_checklist,
    vram_breakdown,
    vram_estimate,
)

GIB = Decimal(1024**3)
NOW = datetime(2026, 8, 9, tzinfo=UTC)
PROV = ProvenanceOut(
    last_verified_at=NOW,
    age_days=0,
    variant="fresh",
    source_name="Test",
    source_url="https://example.test",
    source_kind="vendor",
)

LLAMA_8B = BY_KEY["llama-3.1-8b"]
LLAMA_70B = BY_KEY["llama-3.3-70b"]
PHI_MINI = BY_KEY["phi-3-mini"]  # multi-head attention
CODELLAMA = BY_KEY["codellama-13b"]  # multi-head attention


def gpu(
    instance: str,
    *,
    vram_total: int,
    hourly: str = "2.00",
    count: int = 1,
    spot: bool = False,
) -> GpuOut:
    return GpuOut(
        id=f"gpu_{instance}",
        provider="test",
        instance_name=instance,
        gpu_model="H100",
        gpu_count=count,
        vram_gb=vram_total // count,
        vram_total_gb=vram_total,
        hourly_cost_usd=Decimal(hourly),
        monthly_cost_usd=Decimal(hourly) * 730,
        region="us-east-1",
        spot=spot,
        provenance=PROV,
    )


def api_model(input_per_1k: str = "0.00015", output_per_1k: str = "0.0006") -> ModelOut:
    return ModelOut(
        id="gpt-4o-mini",
        provider="openai",
        model_id="gpt-4o-mini",
        display_name="GPT-4o mini",
        family="chat",
        input_cost_per_1k=Decimal(input_per_1k),
        output_cost_per_1k=Decimal(output_per_1k),
        status="active",
        provenance=PROV,
    )


# ── KV cache and GQA ─────────────────────────────────────────────────────────


def test_kv_cache_uses_kv_heads_not_query_heads() -> None:
    """The single most important line in the estimator.

    Llama 3.1 8B: 2 x 32 layers x 8 KV heads x 128 head_dim x 2 bytes
    = 131,072 bytes per token. Using its 32 *query* heads would give 524,288 —
    four times too much, on the model people most often self-host.
    """
    assert kv_bytes_per_token(LLAMA_8B) == Decimal(131_072)

    mha_equivalent = Decimal(2) * 32 * 32 * 128 * 2
    assert mha_equivalent == kv_bytes_per_token(LLAMA_8B) * 4


def test_a_multi_head_model_has_no_gqa_saving() -> None:
    # Phi-3 Mini: 2 x 32 x 32 heads x 96 x 2 = 393,216 bytes/token.
    assert not PHI_MINI.uses_gqa
    assert kv_bytes_per_token(PHI_MINI) == Decimal(393_216)


def test_quantising_the_cache_halves_it() -> None:
    fp16 = kv_bytes_per_token(LLAMA_8B, kv_precision="fp16")
    fp8 = kv_bytes_per_token(LLAMA_8B, kv_precision="fp8")
    assert fp8 * 2 == fp16


# ── vram-estimate ────────────────────────────────────────────────────────────


def test_llama_8b_fp16_at_8k_context() -> None:
    """M13's worked example.

    weights = 8.03e9 x 2 = 16,059,999,998 B = 14.96 GiB
    kv      = 131,072 x 8192 x 1 = 1,073,741,824 B = 1.00 GiB exactly
    """
    parts = vram_breakdown(LLAMA_8B, quantisation="fp16", context=8192, concurrency=1)

    assert parts["weights"] == Decimal(LLAMA_8B.params) * 2
    assert (parts["weights"] / GIB).quantize(Decimal("0.01")) == Decimal("14.96")
    assert parts["kv_cache"] == Decimal(1_073_741_824)
    assert (parts["kv_cache"] / GIB) == Decimal(1)


def test_int4_weights_are_exactly_a_quarter_of_fp16() -> None:
    fp16 = vram_breakdown(LLAMA_8B, quantisation="fp16", context=8192)
    int4 = vram_breakdown(LLAMA_8B, quantisation="int4", context=8192)
    assert int4["weights"] * 4 == fp16["weights"]


def test_the_kv_cache_overtakes_the_weights_at_long_context_and_concurrency() -> None:
    """M13's headline claim, checked.

    131,072 B/token x 32,768 tokens x 8 concurrent = 34,359,738,368 B = 32 GiB,
    against 14.96 GiB of weights.
    """
    parts = vram_breakdown(LLAMA_8B, quantisation="fp16", context=32_768, concurrency=8)

    assert parts["kv_cache"] == Decimal(34_359_738_368)
    assert (parts["kv_cache"] / GIB) == Decimal(32)
    assert parts["kv_cache"] > parts["weights"]


def test_long_context_raises_the_kv_warning() -> None:
    result = vram_estimate(LLAMA_8B, context=32_768, concurrency=8, gpus=[])
    assert any("KV cache" in w.message for w in result.warnings)


def test_the_breakdown_reports_every_term_separately() -> None:
    result = vram_estimate(LLAMA_8B, context=8192, gpus=[])
    components = {row["component"] for row in result.tables["breakdown"]}

    assert "Weights" in components
    assert "KV cache" in components
    assert "Activations" in components
    assert "Total" in components


def test_runtime_overhead_is_applied_and_differs_by_runtime() -> None:
    vllm = vram_breakdown(LLAMA_8B, runtime="vllm", context=8192)
    transformers = vram_breakdown(LLAMA_8B, runtime="transformers", context=8192)
    assert transformers["total"] > vllm["total"]
    # vLLM is 1.10, so overhead is a tenth of the subtotal.
    assert (vllm["overhead"] / (vllm["total"] - vllm["overhead"])).quantize(
        Decimal("0.001")
    ) == Decimal("0.100")


# ── the three-state fit list ─────────────────────────────────────────────────


def test_the_fit_list_distinguishes_three_states_not_two() -> None:
    """ "Does it fit" usually has a "yes, if you shorten the context" answer,
    and a boolean hides exactly the option the user would have taken."""
    result = vram_estimate(
        LLAMA_70B,
        quantisation="fp16",
        context=32_768,
        concurrency=4,
        gpus=[
            gpu("8xH100", vram_total=640, hourly="55.04", count=8),
            gpu("1xA100-80", vram_total=80, hourly="1.79"),
            gpu("1xL4", vram_total=24, hourly="0.80"),
        ],
    )
    verdicts = {row["instance"]: row["verdict"] for row in result.tables["gpu_fit"]}

    # 70B at fp16 is ~131 GiB of weights alone.
    assert verdicts["8xH100"] == "fits"
    assert verdicts["1xA100-80"] == "does not fit"
    assert verdicts["1xL4"] == "does not fit"


def test_a_card_that_holds_the_weights_but_not_the_cache_reports_reduced_context() -> None:
    # 8B at fp16 is 14.96 GiB of weights; a 24 GB card has room for the
    # weights and some cache, but not for 128k tokens of it.
    result = vram_estimate(
        LLAMA_8B,
        quantisation="fp16",
        context=131_072,
        concurrency=1,
        gpus=[gpu("1xL4", vram_total=24, hourly="0.80")],
    )
    row = result.tables["gpu_fit"][0]

    assert row["verdict"] == "fits with reduced context"
    assert "tokens" in row["detail"]


def test_the_fit_list_is_ordered_by_verdict_then_price() -> None:
    result = vram_estimate(
        LLAMA_8B,
        context=8192,
        gpus=[
            gpu("expensive", vram_total=80, hourly="9.00"),
            gpu("cheap", vram_total=80, hourly="1.00"),
            gpu("tiny", vram_total=8, hourly="0.10"),
        ],
    )
    order = [row["instance"] for row in result.tables["gpu_fit"]]
    assert order[:2] == ["cheap", "expensive"]
    assert order[-1] == "tiny"


def test_max_context_returns_zero_when_the_weights_alone_do_not_fit() -> None:
    fits_nothing = max_context_for(
        LLAMA_70B,
        available_bytes=Decimal(24) * GIB,
        quantisation="fp16",
        concurrency=1,
        kv_precision="fp16",
        runtime="vllm",
    )
    assert fits_nothing == 0


def test_a_context_beyond_the_models_training_is_critical() -> None:
    result = vram_estimate(BY_KEY["gemma-2-9b"], context=32_768, gpus=[])
    assert any(w.level == "critical" for w in result.warnings)


def test_an_mha_model_is_called_out() -> None:
    result = vram_estimate(CODELLAMA, context=8192, gpus=[])
    assert any("multi-head" in w.message for w in result.warnings)


# ── gpu-cost ─────────────────────────────────────────────────────────────────


def test_break_even_volume_is_the_output_that_decides_the_question() -> None:
    """$2/h x 24 x 30 = $1,440/mo self-hosted.

    The API costs 2000/1000 x $0.00015 + 500/1000 x $0.0006 = $0.0006 per
    request. Break-even is 1440 / (0.0006 x 30.4375) = 78,850 requests a day.
    """
    result = gpu_cost(
        gpu("1xH100", vram_total=80, hourly="2.00"),
        hours_per_day=Decimal(24),
        days_per_month=30,
        utilisation_pct=Decimal(100),
        api_model=api_model(),
        input_tokens=2000,
        output_tokens=500,
        requests_per_day=1000,
    )

    assert result.metrics["self_host_monthly"] == Decimal("1440.000000")
    assert result.metrics["break_even_requests_per_day"] == 78_850


def test_low_utilisation_raises_the_effective_hourly_cost() -> None:
    result = gpu_cost(
        gpu("1xH100", vram_total=80, hourly="2.00"),
        hours_per_day=Decimal(24),
        days_per_month=30,
        utilisation_pct=Decimal(25),
        api_model=api_model(),
    )
    # $2.00 at 25% utilisation is $8.00 per productive hour.
    assert result.metrics["effective_hourly"] == Decimal("8.000000")
    assert any(w.field == "utilisation_pct" for w in result.warnings)


def test_a_spot_rate_is_flagged_as_unplannable() -> None:
    result = gpu_cost(
        gpu("spot", vram_total=80, hourly="0.60", spot=True),
        hours_per_day=Decimal(24),
        days_per_month=30,
        utilisation_pct=Decimal(80),
    )
    assert any("spot" in w.message.lower() for w in result.warnings)


def test_the_crossover_series_is_produced_for_the_chart() -> None:
    result = gpu_cost(
        gpu("1xH100", vram_total=80, hourly="2.00"),
        hours_per_day=Decimal(24),
        days_per_month=30,
        utilisation_pct=Decimal(100),
        api_model=api_model(),
        requests_per_day=100_000,
    )
    series = result.series["crossover"]
    assert len(series) == 13
    # Self-hosting is flat; the API line rises and must overtake it.
    assert Decimal(series[0]["api"]) < Decimal(series[0]["self_host"])
    assert Decimal(series[-1]["api"]) > Decimal(series[-1]["self_host"])


# ── cloud-cost ───────────────────────────────────────────────────────────────


def test_cloud_cost_totals_every_line_and_names_the_driver() -> None:
    result = cloud_cost(
        provider="aws",
        compute_monthly=Decimal(800),
        database_monthly=Decimal(200),
        cache_monthly=Decimal(50),
        storage_gb=Decimal(1000),  # x $0.023 = $23
        egress_gb=Decimal(2000),  # x $0.09  = $180
        load_balancer_monthly=Decimal(25),
    )
    # 800 + 200 + 50 + 23 + 180 + 25 = 1278
    assert result.metrics["monthly_total"] == Decimal("1278.000000")
    assert result.metrics["egress_cost"] == Decimal("180.000000")
    assert result.metrics["dominant_driver"] == "Compute"


def test_egress_rates_differ_by_provider() -> None:
    common = {
        "compute_monthly": Decimal(0),
        "database_monthly": Decimal(0),
        "cache_monthly": Decimal(0),
        "storage_gb": Decimal(0),
        "egress_gb": Decimal(1000),
        "load_balancer_monthly": Decimal(0),
    }
    aws = cloud_cost(provider="aws", **common)  # type: ignore[arg-type]
    gcp = cloud_cost(provider="gcp", **common)  # type: ignore[arg-type]

    assert aws.metrics["egress_cost"] == Decimal("90.000000")
    assert gcp.metrics["egress_cost"] == Decimal("120.000000")


def test_egress_dominating_the_bill_is_flagged() -> None:
    result = cloud_cost(
        provider="aws",
        compute_monthly=Decimal(100),
        database_monthly=Decimal(0),
        cache_monthly=Decimal(0),
        storage_gb=Decimal(0),
        egress_gb=Decimal(5000),
        load_balancer_monthly=Decimal(0),
    )
    assert any("Egress" in w.message for w in result.warnings)


def test_zero_egress_is_challenged_rather_than_accepted() -> None:
    result = cloud_cost(
        provider="aws",
        compute_monthly=Decimal(100),
        database_monthly=Decimal(0),
        cache_monthly=Decimal(0),
        storage_gb=Decimal(0),
        egress_gb=Decimal(0),
        load_balancer_monthly=Decimal(0),
    )
    assert any(w.field == "egress_gb" for w in result.warnings)


# ── readiness-checklist ──────────────────────────────────────────────────────


def test_the_score_is_deterministic_for_a_given_answer_set() -> None:
    answers = ["Health checks on every service", "Budget alerts on the provider account"]
    first = readiness_checklist(self_hosted=True, has_rag=True, completed=answers)
    second = readiness_checklist(self_hosted=True, has_rag=True, completed=answers)
    assert first.metrics["score"] == second.metrics["score"]


def test_an_empty_checklist_scores_zero_and_a_full_one_scores_a_hundred() -> None:
    empty = readiness_checklist(self_hosted=False, has_rag=True, completed=[])
    assert empty.metrics["score"] == 0

    every_item = [row["item"] for row in empty.tables["checklist"]]
    full = readiness_checklist(self_hosted=False, has_rag=True, completed=every_item)
    assert full.metrics["score"] == 100


def test_items_are_conditioned_on_the_stack_described() -> None:
    """A managed-API stack is not asked whether its GPU nodes autoscale."""
    managed = readiness_checklist(self_hosted=False, has_rag=True)
    hosted = readiness_checklist(self_hosted=True, has_rag=True)

    managed_items = {row["item"] for row in managed.tables["checklist"]}
    hosted_items = {row["item"] for row in hosted.tables["checklist"]}

    assert "GPU node failure tested" in hosted_items
    assert "GPU node failure tested" not in managed_items
    assert "Fallback model configured" in managed_items


def test_a_stack_without_rag_is_not_asked_about_prompt_injection() -> None:
    with_rag = readiness_checklist(self_hosted=False, has_rag=True)
    without = readiness_checklist(self_hosted=False, has_rag=False)

    assert any("injection" in row["item"].lower() for row in with_rag.tables["checklist"])
    assert not any("injection" in row["item"].lower() for row in without.tables["checklist"])


def test_outstanding_critical_items_are_surfaced() -> None:
    result = readiness_checklist(self_hosted=True, has_rag=True, completed=[])
    assert result.metrics["critical_outstanding"] > 0
    assert any(w.level == "warning" for w in result.warnings)


@pytest.mark.parametrize("self_hosted", [True, False])
def test_the_score_is_the_completed_weight_over_the_applicable_weight(
    self_hosted: bool,
) -> None:
    result = readiness_checklist(self_hosted=self_hosted, has_rag=True, completed=[])
    rows = result.tables["checklist"]

    half = [row["item"] for row in rows if row["weight"] >= 5]
    scored = readiness_checklist(self_hosted=self_hosted, has_rag=True, completed=half)

    total_weight = sum(row["weight"] for row in rows)
    done_weight = sum(row["weight"] for row in rows if row["item"] in set(half))
    assert scored.metrics["score"] == round(done_weight / total_weight * 100)
