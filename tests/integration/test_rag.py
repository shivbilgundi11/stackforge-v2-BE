"""WF2 endpoints, including the one that takes a file."""

from __future__ import annotations

import asyncio
import io
from pathlib import Path

import pytest
from httpx import AsyncClient
from pypdf import PdfWriter

pytestmark = pytest.mark.usefixtures("seeded_catalog")

CHUNK = "/api/v1/tools/rag/chunk-estimate"
VECTORDB = "/api/v1/tools/rag/vectordb-estimate"
PIPELINE = "/api/v1/tools/rag/pipeline-cost"
STRATEGY = "/api/v1/tools/rag/chunking-strategy"
ARCHITECTURE = "/api/v1/tools/rag/architecture"
PDF = "/api/v1/tools/rag/pdf-tokens"


def _pdf(pages: int = 2) -> bytes:
    """A real PDF, built in memory.

    pypdf writes structurally valid pages with no text content, which is
    exactly the scanned-document case: a page that exists and yields nothing.
    """
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=612, height=792)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def _pdfs_under(root: Path) -> set[Path]:
    """A recursive scan is blocking; run it off the loop."""
    return set(root.rglob("*.pdf"))


# ── chunk-estimate ───────────────────────────────────────────────────────────


async def test_chunk_estimate_end_to_end(client: AsyncClient) -> None:
    response = await client.post(
        CHUNK,
        json={
            "document_count": 10,
            "avg_tokens_per_document": 1000,
            "chunk_size": 500,
            "overlap": 100,
        },
    )
    assert response.status_code == 200

    data = response.json()["data"]
    assert data["metrics"]["total_chunks"] == 30
    assert data["tables"]["quality_factors"]


async def test_overlap_at_or_above_chunk_size_is_rejected(client: AsyncClient) -> None:
    """A splitter that never advances is not a configuration to clamp."""
    response = await client.post(
        CHUNK,
        json={
            "document_count": 10,
            "avg_tokens_per_document": 1000,
            "chunk_size": 500,
            "overlap": 500,
        },
    )
    assert response.status_code == 422


# ── vectordb-estimate ────────────────────────────────────────────────────────


async def test_vectordb_estimate_compares_providers(client: AsyncClient) -> None:
    response = await client.post(
        VECTORDB,
        json={"vector_count": 1_000_000, "dimensions": 1536, "index_type": "hnsw"},
    )
    assert response.status_code == 200

    data = response.json()["data"]
    assert data["metrics"]["raw_gb"] == "5.72"
    assert data["metrics"]["index_overhead_gb"] == "2.86"
    assert len(data["tables"]["providers"]) >= 3


# ── pipeline-cost ────────────────────────────────────────────────────────────


async def test_pipeline_cost_uses_live_catalog_rates(client: AsyncClient) -> None:
    response = await client.post(
        PIPELINE,
        json={
            "document_count": 1000,
            "avg_tokens_per_document": 1000,
            "chunk_size": 500,
            "overlap": 100,
            "queries_per_day": 500,
            "chunks_retrieved": 5,
            "embedding_model_id": "text-embedding-3-small",
            "generation_model_id": "gpt-4o-mini",
        },
    )
    assert response.status_code == 200

    data = response.json()["data"]
    assert data["metrics"]["total_chunks"] == 3000
    # Rates came from the catalog, so the run carries their verification dates.
    assert data["provenance"]["sources"]


async def test_a_chat_model_in_the_embedding_slot_is_a_field_error(client: AsyncClient) -> None:
    response = await client.post(
        PIPELINE,
        json={
            "document_count": 10,
            "avg_tokens_per_document": 1000,
            "queries_per_day": 10,
            "embedding_model_id": "gpt-4o-mini",
            "generation_model_id": "gpt-4o-mini",
        },
    )
    assert response.status_code == 422

    fields = response.json()["error"]["details"]["fields"]
    # `path`, not `field`: the client calls setError with it, and setError on
    # undefined silently does nothing while the toast stays suppressed.
    assert fields[0]["path"] == "embedding_model_id"


async def test_a_per_search_reranker_does_not_explode_the_bill(client: AsyncClient) -> None:
    """D-18, end to end against the seeded catalog."""
    body = {
        "document_count": 1000,
        "avg_tokens_per_document": 1000,
        "queries_per_day": 1000,
        "chunks_retrieved": 20,
        "embedding_model_id": "text-embedding-3-small",
        "generation_model_id": "gpt-4o-mini",
    }
    cohere = await client.post(PIPELINE, json={**body, "rerank_model_id": "rerank-4-fast"})
    assert cohere.status_code == 200

    monthly = float(cohere.json()["data"]["metrics"]["monthly_cost"])
    # ~30,400 queries at $0.002 a search is about $61 of reranking. Charged
    # per token instead it would be six figures.
    assert monthly < 1000


# ── chunking-strategy ────────────────────────────────────────────────────────


async def test_chunking_strategy_recommends_and_explains(client: AsyncClient) -> None:
    response = await client.post(
        STRATEGY, json={"document_type": "docs", "avg_tokens_per_document": 3000}
    )
    assert response.status_code == 200

    data = response.json()["data"]
    assert data["metrics"]["strategy_key"] == "markdown"
    assert data["metrics"]["reasoning"]
    assert len(data["tables"]["alternatives"]) == 3


# ── rag-architecture ─────────────────────────────────────────────────────────


async def test_architecture_returns_components_and_a_diagram(client: AsyncClient) -> None:
    response = await client.post(
        ARCHITECTURE,
        json={"use_case": "docs", "corpus_documents": 50_000, "sensitivity": "internal"},
    )
    assert response.status_code == 200

    data = response.json()["data"]
    assert len(data["tables"]["components"]) == 7

    diagram = next(a for a in data["artifacts"] if a["format"] == "mermaid")
    assert diagram["content"].startswith("graph LR")
    assert "-->" in diagram["content"]


async def test_architecture_runs_rule_based_without_ai(client: AsyncClient) -> None:
    """D-06: no AI, same components, an honest source label.

    M16 has not shipped, so this is the live behaviour rather than a simulated
    outage — which makes it the strongest form of this test available.
    """
    response = await client.post(
        ARCHITECTURE, json={"use_case": "support", "corpus_documents": 1000}
    )
    data = response.json()["data"]

    assert data["source"] == "rule_based"
    assert data["ai"] is None
    assert data["tables"]["components"]
    assert data["metrics"]["summary"]


async def test_restricted_sensitivity_excludes_managed_vector_stores(
    client: AsyncClient,
) -> None:
    response = await client.post(
        ARCHITECTURE,
        json={
            "use_case": "policy",
            "corpus_documents": 10_000,
            "sensitivity": "restricted",
        },
    )
    data = response.json()["data"]
    store = next(c for c in data["tables"]["components"] if c["stage"] == "store")

    # Positive assertion first. The earlier version of this test only checked
    # that the store was not Pinecone, which "None available" also satisfies —
    # so it passed for months against a selector that returned nothing at all.
    assert store["choice"] != "None available"
    assert data["metrics"]["store"] != "none"

    # And the chosen store is genuinely self-hostable.
    catalog = await client.get("/api/v1/catalog/tools", params={"category": "vector-db"})
    by_slug = {tool["slug"]: tool for tool in catalog.json()["data"]}
    assert by_slug[data["metrics"]["store"]]["self_hostable"] is True

    assert data["metrics"]["self_hosted"] == "yes"
    assert "Self-hosted" in next(
        c["choice"] for c in data["tables"]["components"] if c["stage"] == "embedder"
    )


async def test_an_unrestricted_design_may_use_a_hosted_store(client: AsyncClient) -> None:
    """The other half of the pair: without the constraint, the filter is off."""
    response = await client.post(
        ARCHITECTURE,
        json={"use_case": "support", "corpus_documents": 100_000, "sensitivity": "public"},
    )
    data = response.json()["data"]

    assert data["metrics"]["store"] != "none"
    assert data["metrics"]["self_hosted"] == "no"


async def test_a_tight_latency_budget_excludes_cross_encoder_reranking(
    client: AsyncClient,
) -> None:
    response = await client.post(
        ARCHITECTURE,
        json={
            "use_case": "support",
            "corpus_documents": 10_000,
            "latency_target_ms": 200,
        },
    )
    data = response.json()["data"]

    assert data["metrics"]["reranking"] == "no"
    assert any("reranking" in w["message"] for w in data["warnings"])


async def test_a_generous_latency_budget_keeps_the_reranker(client: AsyncClient) -> None:
    response = await client.post(
        ARCHITECTURE,
        json={
            "use_case": "support",
            "corpus_documents": 10_000,
            "latency_target_ms": 3000,
        },
    )
    assert response.json()["data"]["metrics"]["reranking"] == "yes"


# ── pdf-tokens ───────────────────────────────────────────────────────────────


async def test_pdf_upload_returns_page_and_token_counts(client: AsyncClient) -> None:
    response = await client.post(
        PDF,
        files={"file": ("doc.pdf", _pdf(pages=3), "application/pdf")},
        data={"model_id": "text-embedding-3-small"},
    )
    assert response.status_code == 200

    metrics = response.json()["data"]["metrics"]
    assert metrics["pages"] == 3
    assert len(response.json()["data"]["tables"]["pages"]) == 3


async def test_a_scanned_page_raises_an_ocr_warning(client: AsyncClient) -> None:
    """Blank pages are structurally valid and yield no text — the scanned case."""
    response = await client.post(
        PDF,
        files={"file": ("scan.pdf", _pdf(pages=2), "application/pdf")},
    )
    data = response.json()["data"]

    assert data["metrics"]["pages_needing_ocr"] == 2
    assert any("OCR" in w["message"] for w in data["warnings"])


async def test_an_oversized_upload_is_refused_with_413(client: AsyncClient) -> None:
    oversized = b"%PDF-1.4\n" + b"0" * (26 * 1024 * 1024)
    response = await client.post(
        PDF,
        files={"file": ("big.pdf", oversized, "application/pdf")},
    )
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "PAYLOAD_TOO_LARGE"


async def test_a_non_pdf_is_refused(client: AsyncClient) -> None:
    response = await client.post(
        PDF,
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )
    assert response.status_code == 422


async def test_an_unreadable_pdf_is_a_422_not_a_500(client: AsyncClient) -> None:
    response = await client.post(
        PDF,
        files={"file": ("broken.pdf", b"%PDF-1.4 not really a pdf", "application/pdf")},
    )
    assert response.status_code == 422


async def test_the_uploaded_file_is_never_written_to_disk(
    client: AsyncClient, tmp_path: Path
) -> None:
    """The promise the UI makes, asserted rather than trusted.

    Checks two things: no new file appears anywhere under the working tree or
    the temp directory, and the stored run payload contains metadata only. A
    planning tool that quietly retains a confidential corpus is a liability
    dressed as a feature.
    """
    before = await asyncio.to_thread(_pdfs_under, Path.cwd())

    response = await client.post(
        PDF,
        files={"file": ("confidential.pdf", _pdf(pages=2), "application/pdf")},
    )
    assert response.status_code == 200

    after = await asyncio.to_thread(_pdfs_under, Path.cwd())
    assert after == before, f"a PDF appeared on disk: {after - before}"
    assert not await asyncio.to_thread(_pdfs_under, tmp_path)

    # The run log keeps the filename and size, never the bytes.
    run_id = response.json()["data"]["run_id"]
    detail = await client.get(f"/api/v1/runs/{run_id}")
    stored_input = detail.json()["data"]["input"]

    assert stored_input["filename"] == "confidential.pdf"
    assert "content" not in stored_input
    assert not any(isinstance(value, str) and "%PDF" in value for value in stored_input.values())
