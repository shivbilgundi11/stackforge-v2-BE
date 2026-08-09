"""RAG arithmetic, against hand-computed values."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from app.schemas.catalog import ModelOut, ProvenanceOut, ToolOut
from app.services.rag_service import (
    INDEX_OVERHEAD,
    chunk_estimate,
    chunking_strategy,
    chunks_per_document,
    pdf_tokens,
    pipeline_cost,
    rerank_monthly_cost,
    vectordb_estimate,
)

NOW = datetime(2026, 8, 9, tzinfo=UTC)
PROV = ProvenanceOut(
    last_verified_at=NOW,
    age_days=0,
    variant="fresh",
    source_name="Test",
    source_url="https://example.test",
    source_kind="vendor",
)


def model(
    model_id: str,
    *,
    family: str = "embedding",
    input_per_1k: str = "0.00002",
    output_per_1k: str | None = None,
    dimensions: int | None = 1536,
    context_window: int | None = 8192,
    price_unit: str = "tokens",
) -> ModelOut:
    return ModelOut(
        id=model_id,
        provider="test",
        model_id=model_id,
        display_name=model_id,
        family=family,
        input_cost_per_1k=Decimal(input_per_1k),
        output_cost_per_1k=Decimal(output_per_1k) if output_per_1k else None,
        dimensions=dimensions,
        context_window=context_window,
        price_unit=price_unit,
        status="active",
        provenance=PROV,
    )


# ── chunk-estimate ───────────────────────────────────────────────────────────


def test_chunking_counts_per_document_not_across_the_corpus() -> None:
    """M11's worked example says 25; the honest answer is 30.

    A 1,000-token document at size 500 / overlap 100 yields windows at
    0-500, 400-900, 800-1000 — three chunks, the last a short tail. Ten
    documents is 30. Getting 25 requires treating the corpus as one
    continuous stream, which no splitter does, and it under-counts by exactly
    one partial chunk per document.
    """
    assert chunks_per_document(doc_tokens=1000, chunk_size=500, overlap=100) == 3

    result = chunk_estimate(
        document_count=10,
        avg_tokens_per_document=1000,
        chunk_size=500,
        overlap=100,
    )
    assert result.metrics["total_chunks"] == 30


def test_a_document_shorter_than_a_chunk_is_one_chunk() -> None:
    assert chunks_per_document(doc_tokens=300, chunk_size=500, overlap=100) == 1


def test_heavy_overlap_is_flagged_with_its_duplication_factor() -> None:
    result = chunk_estimate(
        document_count=10,
        avg_tokens_per_document=1000,
        chunk_size=500,
        overlap=400,
    )
    # Stride 100 → ceil(600/100) = 6 chunks per doc, 60 total, embedding
    # 30,000 tokens of a 10,000-token corpus.
    assert result.metrics["total_chunks"] == 60
    assert result.metrics["duplication_factor"] == Decimal("3.00")
    assert any(w.field == "overlap" and w.level == "warning" for w in result.warnings)


def test_thin_overlap_is_flagged_too() -> None:
    result = chunk_estimate(
        document_count=5, avg_tokens_per_document=4000, chunk_size=500, overlap=10
    )
    assert any(w.field == "overlap" and w.level == "info" for w in result.warnings)


def test_the_quality_score_shows_every_deduction_that_made_it() -> None:
    """A score without its basis is an opinion with a number stuck on it."""
    result = chunk_estimate(
        document_count=10,
        avg_tokens_per_document=8000,
        chunk_size=4000,
        overlap=100,
        query_type="factoid",
    )
    factors = result.tables["quality_factors"]

    assert result.metrics["retrieval_quality"] < 100
    assert len(factors) >= 2
    # Each factor carries its own arithmetic, so the score is reconstructable.
    deductions = sum(abs(int(row["impact"])) for row in factors)
    assert result.metrics["retrieval_quality"] == 100 - deductions


def test_a_clean_configuration_scores_full_marks() -> None:
    result = chunk_estimate(
        document_count=10,
        avg_tokens_per_document=5000,
        chunk_size=512,
        overlap=77,
        query_type="mixed",
    )
    assert result.metrics["retrieval_quality"] == 100


def test_chunks_beyond_the_embedding_window_are_a_critical_warning() -> None:
    result = chunk_estimate(
        document_count=10,
        avg_tokens_per_document=20_000,
        chunk_size=10_000,
        overlap=1_000,
        model=model("small", context_window=512),
    )
    assert any(w.level == "critical" for w in result.warnings)


# ── vectordb-estimate ────────────────────────────────────────────────────────


def test_hnsw_overhead_is_modelled_separately_from_the_vectors() -> None:
    """M11's worked example: 1M vectors x 1536 dims, HNSW.

    Raw = 1,000,000 x 1536 x 4 bytes = 6,144,000,000 = 5.72 GiB.
    HNSW adds 50% = 2.86 GiB. Metadata at 200 B/vector = 0.19 GiB.
    """
    result = vectordb_estimate(vector_count=1_000_000, dimensions=1536, index_type="hnsw")

    assert result.metrics["raw_gb"] == Decimal("5.72")
    assert result.metrics["index_overhead_gb"] == Decimal("2.86")
    assert result.metrics["metadata_gb"] == Decimal("0.19")
    assert result.metrics["total_gb"] == Decimal("8.77")


def test_index_overhead_differs_by_type() -> None:
    """A flat multiplier across HNSW, IVF, and flat is wrong by enough to
    change which provider you pick."""
    args = {"vector_count": 1_000_000, "dimensions": 1536}
    flat = vectordb_estimate(**args, index_type="flat")  # type: ignore[arg-type]
    ivf = vectordb_estimate(**args, index_type="ivf")  # type: ignore[arg-type]
    hnsw = vectordb_estimate(**args, index_type="hnsw")  # type: ignore[arg-type]

    assert flat.metrics["index_overhead_gb"] == Decimal("0.00")
    assert Decimal(str(ivf.metrics["total_gb"])) < Decimal(str(hnsw.metrics["total_gb"]))
    assert INDEX_OVERHEAD["hnsw"] > INDEX_OVERHEAD["ivf"] > INDEX_OVERHEAD["flat"]


def test_replicas_multiply_the_whole_footprint() -> None:
    one = vectordb_estimate(vector_count=1_000_000, dimensions=768, replicas=1)
    three = vectordb_estimate(vector_count=1_000_000, dimensions=768, replicas=3)

    # Compared with a cent of slack, not exactly. The service rounds once at
    # the end, which is right; rounding per replica and then multiplying would
    # drift, and asserting the drifted value would enshrine the wrong
    # behaviour.
    scaled = Decimal(str(one.metrics["total_gb"])) * 3
    assert abs(Decimal(str(three.metrics["total_gb"])) - scaled) <= Decimal("0.02")


def test_a_flat_index_at_scale_is_warned_about() -> None:
    result = vectordb_estimate(vector_count=5_000_000, dimensions=768, index_type="flat")
    assert any(w.field == "index_type" for w in result.warnings)


def test_provider_comparison_respects_the_monthly_minimum() -> None:
    cheap = ToolOut(
        id="t1",
        slug="cheap",
        name="Cheap DB",
        category="vector-db",
        description="",
        status="active",
        maturity_score=80,
        self_hostable=True,
        facts={"cost_per_m_vectors_month": 20, "min_monthly": 500},
        last_reviewed_at=NOW,
    )
    result = vectordb_estimate(vector_count=1_000_000, dimensions=768, providers=[cheap])
    row = result.tables["providers"][0]
    # $20 of usage against a $500 floor: the floor is what you pay.
    assert row["monthly_cost"] == "$500.00"
    assert row["at_minimum"] == "yes"


# ── rerank units (D-18) ──────────────────────────────────────────────────────


def test_a_per_search_reranker_is_not_charged_per_token() -> None:
    """The whole reason `price_unit` exists.

    Cohere bills $2.00 per 1K searches, stored as $0.002 per search. At 30,000
    queries that is $60. Charging the same figure per 1K tokens over
    30,000 x 20 x 512 tokens would give $614,400 — the error looks like a
    plausible price, which is what makes it dangerous.
    """
    cohere = model("rerank-4-fast", family="rerank", input_per_1k="0.002", price_unit="searches")

    cost = rerank_monthly_cost(
        cohere,
        queries_per_month=Decimal(30_000),
        documents_per_query=20,
        tokens_per_document=512,
    )
    assert cost == Decimal("60.000000")


def test_a_per_token_reranker_is_charged_per_token() -> None:
    voyage = model("rerank-2.5", family="rerank", input_per_1k="0.00005", price_unit="tokens")

    # 30,000 queries x 20 docs x 512 tokens = 307,200,000 tokens
    # = 307,200 thousand x $0.00005 = $15.36.
    cost = rerank_monthly_cost(
        voyage,
        queries_per_month=Decimal(30_000),
        documents_per_query=20,
        tokens_per_document=512,
    )
    assert cost == Decimal("15.360000")


def test_a_search_covers_up_to_a_hundred_documents() -> None:
    """Cohere defines one search as one query against up to 100 documents, so
    reranking 150 candidates is two searches, not one."""
    cohere = model("rerank-4-fast", family="rerank", input_per_1k="0.002", price_unit="searches")

    one = rerank_monthly_cost(
        cohere, queries_per_month=Decimal(1000), documents_per_query=50, tokens_per_document=512
    )
    two = rerank_monthly_cost(
        cohere, queries_per_month=Decimal(1000), documents_per_query=150, tokens_per_document=512
    )
    assert two == one * 2


# ── pipeline-cost ────────────────────────────────────────────────────────────


def test_pipeline_cost_pulls_every_rate_from_the_catalog() -> None:
    embed = model("embed", input_per_1k="0.00002")
    gen = model("gen", family="chat", input_per_1k="0.00015", output_per_1k="0.0006")

    result = pipeline_cost(
        document_count=1000,
        avg_tokens_per_document=1000,
        chunk_size=500,
        overlap=100,
        reindex_per_month=Decimal(1),
        queries_per_day=100,
        chunks_retrieved=5,
        embedding_model=embed,
        generation_model=gen,
    )

    # 3 chunks/doc x 1000 docs = 3000 chunks x 500 tokens = 1,500,000 tokens.
    # Ingestion: 1500 thousand x $0.00002 = $0.03.
    assert result.metrics["total_chunks"] == 3000
    assert result.metrics["ingestion_cost"] == Decimal("0.030000")
    assert result.metrics["dominant_cost"] in {"Generation", "Re-indexing", "Query embedding"}
    assert Decimal(str(result.metrics["cost_per_query"])) > 0


def test_the_breakdown_names_the_dominant_recurring_line() -> None:
    embed = model("embed", input_per_1k="0.00002")
    gen = model("gen", family="chat", input_per_1k="0.005", output_per_1k="0.025")

    result = pipeline_cost(
        document_count=100,
        avg_tokens_per_document=1000,
        chunk_size=500,
        overlap=100,
        reindex_per_month=Decimal(0),
        queries_per_day=1000,
        chunks_retrieved=10,
        embedding_model=embed,
        generation_model=gen,
    )
    # Expensive model, high query volume, no re-indexing: generation dominates.
    assert result.metrics["dominant_cost"] == "Generation"


def test_pipeline_cost_reports_the_rows_it_read() -> None:
    embed = model("embed")
    gen = model("gen", family="chat", input_per_1k="0.0001", output_per_1k="0.0004")
    rer = model("rer", family="rerank", input_per_1k="0.002", price_unit="searches")

    result = pipeline_cost(
        document_count=10,
        avg_tokens_per_document=500,
        chunk_size=500,
        overlap=0,
        reindex_per_month=Decimal(1),
        queries_per_day=10,
        chunks_retrieved=5,
        embedding_model=embed,
        generation_model=gen,
        rerank_model=rer,
    )
    assert set(result.sourced_from) == {"embed", "gen", "rer"}


# ── chunking-strategy ────────────────────────────────────────────────────────


def test_markdown_documents_get_the_markdown_splitter() -> None:
    result = chunking_strategy(document_type="docs", avg_tokens_per_document=3000)
    assert result.metrics["strategy_key"] == "markdown"


def test_factoid_queries_get_smaller_chunks_than_synthesis() -> None:
    factoid = chunking_strategy(
        document_type="mixed", avg_tokens_per_document=3000, query_pattern="factoid"
    )
    synthesis = chunking_strategy(
        document_type="mixed", avg_tokens_per_document=3000, query_pattern="synthesis"
    )
    assert factoid.metrics["chunk_size"] < synthesis.metrics["chunk_size"]


def test_semantic_chunking_is_penalised_on_a_large_corpus() -> None:
    """It needs an embedding pass before you have chunked anything."""
    result = chunking_strategy(document_type="research", avg_tokens_per_document=200_000)
    assert result.metrics["strategy_key"] != "semantic"


def test_alternatives_state_what_each_one_trades_away() -> None:
    result = chunking_strategy(document_type="articles", avg_tokens_per_document=2000)
    alternatives = result.tables["alternatives"]
    assert len(alternatives) == 3
    assert all(row["tradeoff"] for row in alternatives)


# ── pdf-tokens ───────────────────────────────────────────────────────────────


def test_pdf_token_count_and_cost() -> None:
    embed = model("embed", input_per_1k="0.00002")
    pages = ["x" * 4000, "y" * 4000]

    result = pdf_tokens(filename="doc.pdf", page_texts=pages, model=embed)

    # 8,000 characters at 4 chars/token = 2,000 tokens.
    assert result.metrics["pages"] == 2
    assert result.metrics["tokens"] == 2000
    assert result.metrics["ingestion_cost"] == Decimal("0.000040")


def test_a_page_with_no_text_raises_an_ocr_warning() -> None:
    embed = model("embed")
    result = pdf_tokens(filename="scan.pdf", page_texts=["text", "", "   "], model=embed)

    assert result.metrics["pages_needing_ocr"] == 2
    assert any("OCR" in w.message for w in result.warnings)


def test_the_response_says_the_file_was_not_kept() -> None:
    """Promised in the UI, so it has to be promised in the payload too."""
    result = pdf_tokens(filename="doc.pdf", page_texts=["hello"], model=model("embed"))
    assert any("discarded" in w.message for w in result.warnings)


def test_the_token_estimate_admits_it_is_an_estimate() -> None:
    result = pdf_tokens(filename="doc.pdf", page_texts=["hello"], model=model("embed"))
    assert any("estimated" in w.message.lower() for w in result.warnings)
