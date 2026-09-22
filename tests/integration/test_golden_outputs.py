"""Every rule-based tool, pinned output, one file.

Twenty-one of the tool endpoints answer with arithmetic and nothing else. The
nine AI-backed ones are excluded on purpose: `enrich` adds prose to a run and
cannot be snapshotted, and their numeric core is a `compute` covered by the
unit suites either way.

The unit suites already check these formulas at 99% line coverage. What they
cannot check is the *whole answer*: a handler that drops a table, a rename that
empties a metric key the frontend reads, a rounding change three helpers deep
that moves a figure by a cent. Each of those leaves every unit test green,
because each unit test asserts the thing it was written to assert. This file
asserts everything each tool returns, which is the only way a change nobody
predicted shows up as a failure.

**The catalog is pinned, not real.** `pinned_catalog` rewrites the handful of
model and GPU rows these cases touch to fixed prices inside the test's own
transaction, so a genuine price change in `models_seed.py` cannot move a single
golden. That separation is the point. Catalog drift is tracked by
`pricing_history` and reviewed when the seed is re-read; this file is about
whether the arithmetic on top of it still does what it did yesterday. Without
the pin, the September price sweep would have reported twenty-one failures and
meant nothing by them.

Provenance, `run_id`, `duration_ms` and `created_at` are dropped before
comparison: they are respectively source metadata, a ULID, a stopwatch and a
clock. Pinning them would be pinning the test harness.

Regenerating, after a change you have read and believe:

    UPDATE_GOLDEN=1 uv run pytest tests/integration/test_golden_outputs.py

That rewrites `tests/golden/rule_based.json`. The diff is the review: every
figure that moved is on screen, and a reviewer who cannot say why a number
changed has found the bug this file exists to catch.
"""

from __future__ import annotations

import io
import json
import os
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from httpx import AsyncClient
from pypdf import PdfWriter
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

GOLDEN_PATH = Path(__file__).parent.parent / "golden" / "rule_based.json"
UPDATING = os.environ.get("UPDATE_GOLDEN") == "1"

# ── The pinned catalog ───────────────────────────────────────────────────────

# Chosen to be the cheapest rows to reason about rather than the most realistic:
# a reviewer reading a golden diff should be able to divide by these in their
# head. They are deliberately *not* the real September prices, so that nobody
# reads a golden figure as a quotable cost.
PINNED_MODELS: dict[str, dict[str, Any]] = {
    "gpt-4o-mini": {
        "input_cost_per_1k": Decimal("0.000150"),
        "output_cost_per_1k": Decimal("0.000600"),
        "cached_input_cost_per_1k": Decimal("0.000075"),
        "context_window": 128_000,
    },
    "gpt-4o": {
        "input_cost_per_1k": Decimal("0.002500"),
        "output_cost_per_1k": Decimal("0.010000"),
        "cached_input_cost_per_1k": Decimal("0.001250"),
        "context_window": 128_000,
    },
    "text-embedding-3-small": {
        "input_cost_per_1k": Decimal("0.000020"),
        "output_cost_per_1k": None,
        "cached_input_cost_per_1k": None,
        "context_window": 8_191,
    },
    "rerank-4-fast": {
        "input_cost_per_1k": Decimal("0.002000"),
        "output_cost_per_1k": None,
        "cached_input_cost_per_1k": None,
        "context_window": 32_768,
    },
}

PINNED_GPU_SLUG = "lambda-gpu-1x-h100-pcie"
PINNED_GPU_HOURLY = Decimal("2.490000")


@pytest.fixture
async def pinned_catalog(db: AsyncSession, seeded_catalog: None) -> None:
    """Freeze the catalog rows these cases read.

    Written inside the per-test transaction, so it is invisible to every other
    test and rolled back at the end.

    Each write asserts it hit exactly one row. A model dropped from the seed
    would otherwise leave the pin silently unapplied and the goldens quietly
    tracking real prices again, which is the one failure mode that would make
    this whole file lie.
    """
    # Spelled out rather than built from the dict's keys: a generated SET
    # clause reads as an injection vector to any reader and to the linter,
    # and four column names are not worth the argument.
    update_model = text(
        "UPDATE model_pricing SET "
        "  input_cost_per_1k = :input_cost_per_1k,"
        "  output_cost_per_1k = :output_cost_per_1k,"
        "  cached_input_cost_per_1k = :cached_input_cost_per_1k,"
        "  context_window = :context_window "
        "WHERE model_id = :model_id"
    )
    for model_id, columns in PINNED_MODELS.items():
        result = await db.execute(update_model, {**columns, "model_id": model_id})
        assert result.rowcount == 1, f"pinned model {model_id!r} matched {result.rowcount} rows"

    provider, _, instance = PINNED_GPU_SLUG.partition("-")
    result = await db.execute(
        text(
            "UPDATE gpu_pricing SET hourly_cost_usd = :cost "
            "WHERE provider = :provider AND replace(instance_name, '_', '-') = :instance"
        ),
        {"cost": PINNED_GPU_HOURLY, "provider": provider, "instance": instance},
    )
    assert result.rowcount == 1, f"pinned GPU {PINNED_GPU_SLUG!r} matched {result.rowcount} rows"
    await db.flush()


# ── The cases ────────────────────────────────────────────────────────────────

# One entry per rule-based endpoint. Inputs are round numbers for the same
# reason the prices are: a golden nobody can check by hand is a golden nobody
# checks.
CASES: dict[str, tuple[str, dict[str, Any]]] = {
    # cost
    "llm-pricing": (
        "/api/v1/tools/cost/llm-pricing",
        {
            "model_id": "gpt-4o-mini",
            "input_tokens": 2000,
            "output_tokens": 500,
            "requests_per_day": 1000,
            "cached_input_ratio": "0.5",
        },
    ),
    "token-calculator": (
        "/api/v1/tools/cost/token-calculator",
        {"text": "The quick brown fox jumps over the lazy dog. " * 10, "model_id": "gpt-4o-mini"},
    ),
    "embedding-cost": (
        "/api/v1/tools/cost/embedding-cost",
        {
            "model_id": "text-embedding-3-small",
            "document_count": 10_000,
            "avg_tokens_per_document": 800,
            "reembeds_per_month": 1,
        },
    ),
    # rag
    "chunk-estimate": (
        "/api/v1/tools/rag/chunk-estimate",
        {
            "document_count": 5000,
            "avg_tokens_per_document": 2400,
            "chunk_size": 512,
            "overlap": 64,
            "model_id": "text-embedding-3-small",
        },
    ),
    "vectordb-estimate": (
        "/api/v1/tools/rag/vectordb-estimate",
        {"vector_count": 1_000_000, "dimensions": 1536, "index_type": "hnsw", "replicas": 2},
    ),
    "pipeline-cost": (
        "/api/v1/tools/rag/pipeline-cost",
        {
            "document_count": 5000,
            "avg_tokens_per_document": 2400,
            "chunk_size": 512,
            "overlap": 64,
            "queries_per_day": 1000,
            "chunks_retrieved": 5,
            "embedding_model_id": "text-embedding-3-small",
            "generation_model_id": "gpt-4o-mini",
            "rerank_model_id": "rerank-4-fast",
            "output_tokens": 400,
        },
    ),
    "chunking-strategy": (
        "/api/v1/tools/rag/chunking-strategy",
        {"document_type": "docs", "avg_tokens_per_document": 3000},
    ),
    # agents
    "mcp-config": (
        "/api/v1/tools/agents/mcp-config",
        {
            "server_name": "golden-server",
            "description": "A server that exists to be snapshotted.",
            "transport": "stdio",
            "tools": [
                {
                    "name": "search",
                    "description": "Search the corpus.",
                    "parameters": [
                        {
                            "name": "query",
                            "type": "string",
                            "description": "What to look for.",
                            "required": True,
                        }
                    ],
                }
            ],
        },
    ),
    "agent-cost": (
        "/api/v1/tools/agents/agent-cost",
        {
            "agents": [{"role": "planner", "model_id": "gpt-4o-mini", "steps_per_task": 4}],
            "tasks_per_day": 200,
            "input_tokens_per_step": 1500,
            "output_tokens_per_step": 400,
            "tool_count": 3,
            "retry_rate_pct": 10,
        },
    ),
    "function-schema": (
        "/api/v1/tools/agents/function-schema",
        {
            "tools": [
                {
                    "name": "get_weather",
                    "description": "Current conditions for a city.",
                    "parameters": [
                        {
                            "name": "city",
                            "type": "string",
                            "description": "City name.",
                            "required": True,
                        }
                    ],
                }
            ],
            "target": "openai",
        },
    ),
    "rate-limits": (
        "/api/v1/tools/agents/rate-limits",
        {
            "provider": "openai",
            "tier": "tier-2",
            "requests_per_min": 300,
            "input_tokens_per_request": 2000,
            "output_tokens_per_request": 500,
            "concurrency": 8,
        },
    ),
    # infra
    "vram-estimate": (
        "/api/v1/tools/infra/vram-estimate",
        {
            "architecture_key": "llama-3.1-8b",
            "quantisation": "fp16",
            "context": 8192,
            "concurrency": 4,
        },
    ),
    "gpu-cost": (
        "/api/v1/tools/infra/gpu-cost",
        {
            "gpu": PINNED_GPU_SLUG,
            "hours_per_day": "24",
            "days_per_month": 30,
            "utilisation_pct": "60",
            "api_model_id": "gpt-4o-mini",
            "input_tokens": 2000,
            "output_tokens": 500,
            "requests_per_day": 1000,
        },
    ),
    "cloud-cost": (
        "/api/v1/tools/infra/cloud-cost",
        {
            "provider": "aws",
            "compute_monthly": "500",
            "database_monthly": "200",
            "cache_monthly": "50",
            "storage_gb": 500,
            "egress_gb": 200,
        },
    ),
    "docker-compose": (
        "/api/v1/tools/infra/docker-compose",
        {"archetype": "rag-stack", "model": "meta-llama/Llama-3.1-8B", "gpu": True},
    ),
    "k8s-estimate": (
        "/api/v1/tools/infra/k8s-estimate",
        {
            "name": "golden",
            "replicas": 3,
            "requests_per_second": 50,
            "gpu_count": 1,
            "vram_required_gb": "24",
        },
    ),
    "readiness-checklist": (
        "/api/v1/tools/infra/readiness-checklist",
        {"self_hosted": True, "has_rag": True, "completed": []},
    ),
    # roi
    "hours-saved": (
        "/api/v1/tools/roi/hours-saved",
        {
            "affected_users": 25,
            "hours_saved_per_user_per_week": "3",
            "fully_loaded_hourly_cost": "95",
            "adoption_rate_pct": "70",
        },
    ),
    "model-roi": (
        "/api/v1/tools/roi/model-roi",
        {
            "current_monthly_cost": "20000",
            "ai_monthly_cost": "6000",
            "implementation_cost": "50000",
            "adoption_ramp_months": 3,
            "horizon_months": 36,
        },
    ),
    "implementation-cost": (
        "/api/v1/tools/roi/implementation-cost",
        {
            "roles": [
                {"name": "Backend engineer", "hours": 320, "hourly_rate": "120"},
                {"name": "ML engineer", "hours": 160, "hourly_rate": "150"},
            ],
            "duration_months": 3,
            "infrastructure_setup": "5000",
            "contingency_pct": "15",
        },
    ),
}


def _pdf(pages: int = 3) -> bytes:
    """Structurally valid pages with no text: the scanned-document case."""
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=612, height=792)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


# ── Comparison ───────────────────────────────────────────────────────────────

# Everything that is not the answer. `provenance` carries the catalog's
# verification dates, which move every time the seed is re-read and say nothing
# about whether the arithmetic is right.
VOLATILE = {"run_id", "duration_ms", "created_at", "provenance", "ai", "source"}


def _answer(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key not in VOLATILE}


def _load() -> dict[str, Any]:
    if not GOLDEN_PATH.exists():
        return {}
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


def _store(golden: dict[str, Any]) -> None:
    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN_PATH.write_text(
        json.dumps(golden, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _compare(name: str, actual: dict[str, Any]) -> None:
    golden = _load()

    if UPDATING:
        golden[name] = actual
        _store(golden)
        return

    assert name in golden, (
        f"no golden recorded for {name!r}. If this tool is new, run "
        f"UPDATE_GOLDEN=1 pytest and review the added block."
    )
    expected = golden[name]
    if actual != expected:
        diff = json.dumps(actual, indent=2, sort_keys=True)
        was = json.dumps(expected, indent=2, sort_keys=True)
        pytest.fail(
            f"{name} no longer returns what it used to.\n\n"
            f"--- recorded ---\n{was}\n\n--- now ---\n{diff}\n\n"
            f"If the change is intended, regenerate with UPDATE_GOLDEN=1 and "
            f"explain every moved figure in the commit message."
        )


# ── Tests ────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", sorted(CASES))
async def test_rule_based_output_is_unchanged(
    name: str, client: AsyncClient, pinned_catalog: None
) -> None:
    path, payload = CASES[name]
    response = await client.post(path, json=payload)

    assert response.status_code == 200, f"{name}: {response.status_code} {response.text}"
    _compare(name, _answer(response.json()["data"]))


async def test_pdf_tokens_output_is_unchanged(client: AsyncClient, pinned_catalog: None) -> None:
    """The one endpoint that takes a file rather than JSON."""
    response = await client.post(
        "/api/v1/tools/rag/pdf-tokens",
        files={"file": ("golden.pdf", _pdf(), "application/pdf")},
        data={"model_id": "gpt-4o-mini"},
    )

    assert response.status_code == 200, response.text
    _compare("pdf-tokens", _answer(response.json()["data"]))


async def test_every_rule_based_endpoint_has_a_case() -> None:
    """The guard that keeps this file honest as tools are added.

    A new rule-based endpoint with no golden is the normal way a suite like
    this rots: it keeps passing, on a shrinking share of the product. Reading
    the routes off the live app means a tool cannot be added without either
    appearing here or failing this test.
    """
    from app.main import app

    ai_backed = {
        "/api/v1/tools/cost/budget-estimator",
        "/api/v1/tools/rag/architecture",
        "/api/v1/tools/agents/workflow-plan",
        "/api/v1/tools/roi/build-vs-buy",
        "/api/v1/tools/compare/models",
        "/api/v1/tools/compare/vector-db",
        "/api/v1/tools/compare/stacks",
        "/api/v1/tools/compare/build-vs-buy",
        "/api/v1/architect/recommend",
        "/api/v1/architect/score",
    }

    posts = {
        route.path
        for route in app.routes
        if "POST" in getattr(route, "methods", set())
        and (route.path.startswith("/api/v1/tools/") or route.path.startswith("/api/v1/architect/"))
    }
    covered = {path for path, _ in CASES.values()} | {"/api/v1/tools/rag/pdf-tokens"}

    uncovered = posts - covered - ai_backed
    assert not uncovered, (
        f"rule-based endpoints with no golden case: {sorted(uncovered)}. "
        f"Add one to CASES, or to `ai_backed` if it calls a model."
    )
