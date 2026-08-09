"""Infra Planner arithmetic (WF4). Pure functions.

The VRAM estimator is the most checkable thing in the product: anyone can
load the model and read `nvidia-smi`. So the terms are computed separately and
reported separately, and the docstrings say where each one comes from.

KV cache is the term that decides whether a deployment works. At long context
and any real concurrency it exceeds the weights - a Llama 3.1 8B is 15 GiB of
weights and 32 GiB of cache at 32k context with 8 concurrent requests - and an
estimator that ignores it, or that assumes multi-head attention on a
grouped-query model, is wrong by multiples on exactly the models people are
deploying.
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Final

from app.data.architectures_seed import (
    KV_PRECISION,
    QUANTISATION,
    RUNTIME_OVERHEAD,
    Architecture,
)
from app.schemas.catalog import GpuOut, ModelOut
from app.schemas.tools import ToolOutput, ToolWarning
from app.services.cost_service import DAYS_PER_MONTH

GIB: Final = Decimal(1024**3)
CENTS: Final = Decimal("0.01")
MICRO: Final = Decimal("0.000001")
HOURS_PER_MONTH: Final = Decimal(730)

# Every modern server prefills in chunks rather than materialising activations
# for the whole sequence at once, so activation memory is bounded by the chunk
# rather than by the context length. 2048 is the common default.
PREFILL_CHUNK: Final = 2048


def _gib(value: Decimal) -> Decimal:
    return (value / GIB).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def _money(value: Decimal) -> Decimal:
    return value.quantize(MICRO, rounding=ROUND_HALF_UP)


def _usd(value: Decimal) -> str:
    amount = f"{abs(value):,.2f}"
    return f"-${amount}" if value < 0 else f"${amount}"


# ── vram-estimate ────────────────────────────────────────────────────────────


def kv_bytes_per_token(arch: Architecture, *, kv_precision: str = "fp16") -> Decimal:
    """Key and value cache for one token, one sequence.

        2 (K and V) x layers x kv_heads x head_dim x bytes

    `kv_heads`, not `heads`. On Llama 3.1 8B that is 8 against 32, so using
    the query-head count would quadruple the figure.
    """
    element = Decimal(str(KV_PRECISION.get(kv_precision, 2.0)))
    return Decimal(2) * arch.layers * arch.kv_heads * arch.head_dim * element


def vram_breakdown(
    arch: Architecture,
    *,
    quantisation: str = "fp16",
    context: int = 8192,
    concurrency: int = 1,
    kv_precision: str = "fp16",
    runtime: str = "vllm",
    lora_finetune: bool = False,
) -> dict[str, Decimal]:
    """Every term, unrounded and separate. Rounding happens at the edge."""
    bytes_per_param = Decimal(str(QUANTISATION.get(quantisation, 2.0)))
    weights = Decimal(arch.params) * bytes_per_param

    kv = kv_bytes_per_token(arch, kv_precision=kv_precision) * context * concurrency

    element = Decimal(str(KV_PRECISION.get(kv_precision, 2.0)))
    activations = (
        Decimal(concurrency) * Decimal(min(context, PREFILL_CHUNK)) * arch.hidden_size * element * 2
    )

    # LoRA training adds gradients and optimizer state for the adapter, plus
    # activations retained for the backward pass. The adapter itself is tiny;
    # the retained activations are not.
    training = Decimal(0)
    if lora_finetune:
        training = activations * Decimal(3) + weights * Decimal("0.02")

    subtotal = weights + kv + activations + training
    multiplier = Decimal(str(RUNTIME_OVERHEAD.get(runtime, 1.10)))
    overhead = subtotal * (multiplier - Decimal(1))

    return {
        "weights": weights,
        "kv_cache": kv,
        "activations": activations,
        "training": training,
        "overhead": overhead,
        "total": subtotal + overhead,
    }


def max_context_for(
    arch: Architecture,
    *,
    available_bytes: Decimal,
    quantisation: str,
    concurrency: int,
    kv_precision: str,
    runtime: str,
) -> int:
    """Longest context that fits in `available_bytes`, or 0 if none does.

    This is what turns "does it fit" from a boolean into a useful answer. The
    honest reply is usually "yes, at 16k rather than the 128k you asked for",
    and a yes/no hides exactly the option the user would have taken.
    """
    fixed = vram_breakdown(
        arch,
        quantisation=quantisation,
        context=0,
        concurrency=concurrency,
        kv_precision=kv_precision,
        runtime=runtime,
    )
    multiplier = Decimal(str(RUNTIME_OVERHEAD.get(runtime, 1.10)))
    headroom = available_bytes / multiplier - (fixed["weights"] + fixed["activations"])
    if headroom <= 0:
        return 0

    per_token = kv_bytes_per_token(arch, kv_precision=kv_precision) * concurrency
    return min(arch.max_context, int(headroom / per_token))


def vram_estimate(
    arch: Architecture,
    *,
    quantisation: str = "fp16",
    context: int = 8192,
    concurrency: int = 1,
    kv_precision: str = "fp16",
    runtime: str = "vllm",
    lora_finetune: bool = False,
    gpus: list[GpuOut] | None = None,
) -> ToolOutput:
    parts = vram_breakdown(
        arch,
        quantisation=quantisation,
        context=context,
        concurrency=concurrency,
        kv_precision=kv_precision,
        runtime=runtime,
        lora_finetune=lora_finetune,
    )
    total = parts["total"]

    fit_rows: list[dict[str, Any]] = []
    for gpu in gpus or []:
        node_bytes = Decimal(gpu.vram_total_gb) * GIB
        if node_bytes >= total:
            verdict, note = "fits", f"{_gib(node_bytes - total)} GiB spare"
        else:
            reduced = max_context_for(
                arch,
                available_bytes=node_bytes,
                quantisation=quantisation,
                concurrency=concurrency,
                kv_precision=kv_precision,
                runtime=runtime,
            )
            if reduced >= 1024:
                verdict = "fits with reduced context"
                note = f"up to {reduced:,} tokens"
            else:
                verdict = "does not fit"
                note = f"needs {_gib(total - node_bytes)} GiB more"

        fit_rows.append(
            {
                "gpu": f"{gpu.gpu_model} x{gpu.gpu_count}",
                "instance": gpu.instance_name,
                "provider": gpu.provider,
                "vram_gb": gpu.vram_total_gb,
                "verdict": verdict,
                "detail": note,
                "hourly": _usd(gpu.hourly_cost_usd),
                "_sort": (
                    {"fits": 0, "fits with reduced context": 1, "does not fit": 2}[verdict],
                    gpu.hourly_cost_usd,
                ),
            }
        )

    fit_rows.sort(key=lambda row: row["_sort"])
    for row in fit_rows:
        del row["_sort"]

    recommended = next((row for row in fit_rows if row["verdict"] == "fits"), None)

    warnings: list[ToolWarning] = []
    if parts["kv_cache"] > parts["weights"]:
        ratio = parts["kv_cache"] / parts["weights"]
        warnings.append(
            ToolWarning(
                level="warning",
                message=(
                    f"KV cache is {ratio:.1f}x the weights at {context:,} tokens and "
                    f"concurrency {concurrency}. Quantising the cache to FP8 halves it, "
                    "and it is the term that decides which card this runs on."
                ),
            )
        )
    if not arch.uses_gqa:
        warnings.append(
            ToolWarning(
                level="info",
                message=(
                    f"{arch.name} uses multi-head attention, so its KV cache is "
                    f"{arch.heads // max(1, arch.kv_heads)}x what a grouped-query model "
                    "of the same size would need. It scales badly with context."
                ),
            )
        )
    if context > arch.max_context:
        warnings.append(
            ToolWarning(
                level="critical",
                field="context",
                message=(
                    f"{arch.name} supports {arch.max_context:,} tokens; "
                    f"{context:,} is beyond what the model was trained for."
                ),
            )
        )
    if recommended is None and fit_rows:
        warnings.append(
            ToolWarning(
                level="warning",
                message=(
                    "Nothing in the catalog fits this at full context. The options are "
                    "a shorter context, a heavier quantisation, or sharding across a "
                    "multi-GPU node."
                ),
            )
        )

    return ToolOutput(
        metrics={
            "total_vram_gb": _gib(total),
            "weights_gb": _gib(parts["weights"]),
            "kv_cache_gb": _gib(parts["kv_cache"]),
            "activations_gb": _gib(parts["activations"]),
            "overhead_gb": _gib(parts["overhead"]),
            "kv_bytes_per_token": int(kv_bytes_per_token(arch, kv_precision=kv_precision)),
            "attention": "GQA" if arch.uses_gqa else "MHA",
            "recommended_gpu": recommended["instance"] if recommended else "none in catalog",
        },
        tables={
            "breakdown": [
                {"component": "Weights", "gb": str(_gib(parts["weights"]))},
                {"component": "KV cache", "gb": str(_gib(parts["kv_cache"]))},
                {"component": "Activations", "gb": str(_gib(parts["activations"]))},
                *(
                    [{"component": "Training state", "gb": str(_gib(parts["training"]))}]
                    if lora_finetune
                    else []
                ),
                {"component": f"Runtime overhead ({runtime})", "gb": str(_gib(parts["overhead"]))},
                {"component": "Total", "gb": str(_gib(total))},
            ],
            "gpu_fit": fit_rows,
        },
        warnings=warnings,
        sourced_from=[gpu.id for gpu in (gpus or [])[:6]],
    )


# ── gpu-cost ─────────────────────────────────────────────────────────────────


def gpu_cost(
    gpu: GpuOut,
    *,
    hours_per_day: Decimal,
    days_per_month: int,
    utilisation_pct: Decimal,
    api_model: ModelOut | None = None,
    input_tokens: int = 2000,
    output_tokens: int = 500,
    requests_per_day: int = 1000,
) -> ToolOutput:
    """Self-host against managed API, and the volume where they cross.

    The break-even figure is the output that actually decides the question.
    Two monthly totals leave the reader to do the interesting arithmetic, and
    the interesting arithmetic is "how much traffic do I need before owning
    the hardware pays".
    """
    monthly_hours = hours_per_day * Decimal(days_per_month)
    self_host = _money(gpu.hourly_cost_usd * monthly_hours)

    utilisation = max(Decimal("0.01"), utilisation_pct / Decimal(100))
    effective_hourly = _money(gpu.hourly_cost_usd / utilisation)

    api_monthly = Decimal(0)
    per_request = Decimal(0)
    break_even: int | str = "n/a"

    if api_model is not None:
        per_request = Decimal(input_tokens) / Decimal(1000) * api_model.input_cost_per_1k + Decimal(
            output_tokens
        ) / Decimal(1000) * (api_model.output_cost_per_1k or Decimal(0))
        api_monthly = _money(per_request * Decimal(requests_per_day) * DAYS_PER_MONTH)

        if per_request > 0:
            # Requests per day at which the API bill reaches the self-hosting
            # bill. Below this, the API is cheaper.
            break_even = int(self_host / (per_request * DAYS_PER_MONTH))

    projection: list[dict[str, Any]] = []
    if api_model is not None and per_request > 0:
        step = max(1, requests_per_day // 6)
        for index in range(13):
            volume = step * index
            projection.append(
                {
                    "requests_per_day": volume,
                    "self_host": str(self_host.quantize(CENTS)),
                    "api": str(
                        _money(per_request * Decimal(volume) * DAYS_PER_MONTH).quantize(CENTS)
                    ),
                }
            )

    warnings: list[ToolWarning] = []
    if utilisation_pct < 30:
        warnings.append(
            ToolWarning(
                level="warning",
                field="utilisation_pct",
                message=(
                    f"At {utilisation_pct}% utilisation the effective cost is "
                    f"{_usd(effective_hourly)} per productive hour against a list price of "
                    f"{_usd(gpu.hourly_cost_usd)}. Idle GPU time is the single biggest "
                    "reason self-hosting loses."
                ),
            )
        )
    if gpu.spot:
        warnings.append(
            ToolWarning(
                level="info",
                message=(
                    "This is a spot or interruptible rate. It is not a price you can "
                    "plan a production SLA against, and the instance can be reclaimed."
                ),
            )
        )
    if isinstance(break_even, int) and break_even > requests_per_day * 10:
        warnings.append(
            ToolWarning(
                level="info",
                message=(
                    f"Break-even is {break_even:,} requests a day against your stated "
                    f"{requests_per_day:,}. At this volume the managed API is not close "
                    "to being the expensive option."
                ),
            )
        )

    return ToolOutput(
        metrics={
            "self_host_monthly": self_host,
            "api_monthly": api_monthly,
            "effective_hourly": effective_hourly,
            "break_even_requests_per_day": break_even,
            "monthly_hours": int(monthly_hours),
            "cheaper": ("self-host" if api_model and api_monthly > self_host else "managed API"),
        },
        tables={
            "comparison": [
                {
                    "option": f"Self-host ({gpu.instance_name})",
                    "monthly": _usd(self_host.quantize(CENTS)),
                    "note": f"{int(monthly_hours)} h at {_usd(gpu.hourly_cost_usd)}/h",
                },
                {
                    "option": f"Managed API ({api_model.display_name})"
                    if api_model
                    else "Managed API",
                    "monthly": _usd(api_monthly.quantize(CENTS)),
                    "note": (
                        f"{requests_per_day:,} req/day at {_usd(per_request)} each"
                        if api_model
                        else "no model selected"
                    ),
                },
            ]
        },
        series={"crossover": projection},
        warnings=warnings,
        sourced_from=[gpu.id] + ([api_model.id] if api_model else []),
    )


# ── cloud-cost ───────────────────────────────────────────────────────────────

# Egress per GB, on-demand, first tier. Included because it is the line that
# surprises people and the one no provider calculator puts on the front page.
EGRESS_PER_GB: Final[dict[str, Decimal]] = {
    "aws": Decimal("0.09"),
    "gcp": Decimal("0.12"),
    "azure": Decimal("0.087"),
}


def cloud_cost(
    *,
    provider: str,
    compute_monthly: Decimal,
    database_monthly: Decimal,
    cache_monthly: Decimal,
    storage_gb: Decimal,
    egress_gb: Decimal,
    load_balancer_monthly: Decimal,
    storage_per_gb: Decimal = Decimal("0.023"),
) -> ToolOutput:
    egress_rate = EGRESS_PER_GB.get(provider, EGRESS_PER_GB["aws"])
    egress = _money(egress_gb * egress_rate)
    storage = _money(storage_gb * storage_per_gb)

    lines = [
        ("Compute", compute_monthly),
        ("Database", database_monthly),
        ("Cache", cache_monthly),
        ("Storage", storage),
        ("Egress", egress),
        ("Load balancer", load_balancer_monthly),
    ]
    total = sum((value for _, value in lines), Decimal(0))
    driver = max(lines, key=lambda item: item[1])

    rows = [
        {
            "line": name,
            "monthly": _usd(value.quantize(CENTS)),
            "share": f"{(value / total * Decimal(100)):.1f}%" if total else "—",
        }
        for name, value in lines
    ]

    warnings: list[ToolWarning] = []
    if total and egress / total > Decimal("0.15"):
        warnings.append(
            ToolWarning(
                level="warning",
                message=(
                    f"Egress is {(egress / total * 100):.0f}% of this bill. It is billed "
                    "per GB leaving the region, so a CDN in front of the API, or keeping "
                    "traffic in-region, moves it more than any instance change."
                ),
            )
        )
    if egress_gb == 0:
        warnings.append(
            ToolWarning(
                level="info",
                field="egress_gb",
                message=(
                    "Zero egress is rarely true. Every API response, log shipment, and "
                    "backup that leaves the region is billed at "
                    f"{_usd(egress_rate)} per GB."
                ),
            )
        )

    return ToolOutput(
        metrics={
            "monthly_total": _money(total),
            "annual_total": _money(total * Decimal(12)),
            "egress_cost": egress,
            "dominant_driver": driver[0],
            "egress_rate_per_gb": egress_rate,
        },
        tables={"lines": rows},
        series={
            "composition": [
                {"line": name, "cost": str(value.quantize(CENTS))}
                for name, value in lines
                if value > 0
            ]
        },
        warnings=warnings,
    )


# ── readiness-checklist ──────────────────────────────────────────────────────

ChecklistItem = dict[str, Any]

# Items are conditioned on the described deployment. A managed-API stack is
# not asked whether its GPU nodes autoscale, and a self-hosted one is not
# allowed to skip the question.
_BASE_ITEMS: Final[tuple[tuple[str, str, str, int], ...]] = (
    (
        "reliability",
        "Health checks on every service",
        "A container that is up is not a service that works.",
        5,
    ),
    ("reliability", "Automated backups, restore tested", "An untested backup is a hope.", 5),
    (
        "reliability",
        "Retries with backoff on provider calls",
        "Providers rate-limit and time out.",
        4,
    ),
    (
        "security",
        "Secrets in a manager, not env files in git",
        "The most common breach in this stack.",
        5,
    ),
    (
        "security",
        "API keys scoped and rotatable",
        "A single unrotatable key is an outage waiting.",
        4,
    ),
    (
        "security",
        "Prompt-injection handling on user input",
        "Any RAG or agent surface is an injection surface.",
        4,
    ),
    (
        "observability",
        "Request tracing across the whole call path",
        "Debugging an agent without traces is guesswork.",
        4,
    ),
    (
        "observability",
        "Token and cost per request recorded",
        "You cannot control a bill you cannot attribute.",
        5,
    ),
    ("observability", "Alerting on error rate and latency", "Not on CPU.", 4),
    ("scaling", "Load tested at 3x expected peak", "Peak is not average.", 3),
    (
        "cost",
        "Budget alerts on the provider account",
        "The first sign should not be the invoice.",
        5,
    ),
    ("cost", "Caching for repeated prompts", "The cheapest token is the one not sent.", 3),
)

_SELF_HOSTED_ITEMS: Final[tuple[tuple[str, str, str, int], ...]] = (
    (
        "reliability",
        "GPU node failure tested",
        "A single-node deployment has a single point of failure.",
        5,
    ),
    (
        "scaling",
        "Autoscaling on queue depth, not CPU",
        "GPU serving is queue-bound, and CPU is flat while it saturates.",
        4,
    ),
    (
        "cost",
        "Idle GPU shutdown outside peak hours",
        "Idle time is most of the bill on a self-hosted deployment.",
        5,
    ),
    (
        "observability",
        "GPU utilisation and VRAM headroom monitored",
        "OOM at 3am is otherwise the first signal.",
        4,
    ),
)

_MANAGED_ITEMS: Final[tuple[tuple[str, str, str, int], ...]] = (
    (
        "reliability",
        "Fallback model configured",
        "Providers have outages, and a single provider is a single point of failure.",
        4,
    ),
    (
        "scaling",
        "Rate-limit headroom checked against peak",
        "The provider's limit binds before your code does.",
        4,
    ),
)


def readiness_checklist(
    *,
    self_hosted: bool,
    has_rag: bool,
    completed: list[str] | None = None,
) -> ToolOutput:
    """A scored checklist, conditioned on the stack described.

    Deterministic for a given answer set: the score is the completed weight
    over the applicable weight, so the same inputs always give the same number
    and two runs are comparable.
    """
    items = list(_BASE_ITEMS)
    items += list(_SELF_HOSTED_ITEMS if self_hosted else _MANAGED_ITEMS)
    if not has_rag:
        items = [item for item in items if "injection" not in item[1].lower()]

    done = set(completed or [])
    rows: list[ChecklistItem] = []
    earned = possible = 0

    for area, label, why, weight in items:
        possible += weight
        is_done = label in done
        if is_done:
            earned += weight
        rows.append(
            {
                "area": area,
                "item": label,
                "why": why,
                "weight": weight,
                "done": "yes" if is_done else "no",
            }
        )

    score = round(earned / possible * 100) if possible else 0
    missing_critical = [row["item"] for row in rows if row["done"] == "no" and row["weight"] == 5]

    warnings: list[ToolWarning] = []
    if missing_critical:
        warnings.append(
            ToolWarning(
                level="warning",
                message=(
                    f"{len(missing_critical)} of the highest-weighted items are "
                    f"outstanding, starting with: {missing_critical[0]}"
                ),
            )
        )

    return ToolOutput(
        metrics={
            "score": score,
            "completed": len(done & {row["item"] for row in rows}),
            "applicable_items": len(rows),
            "critical_outstanding": len(missing_critical),
            "profile": "self-hosted" if self_hosted else "managed",
        },
        tables={"checklist": rows},
        warnings=warnings,
    )
