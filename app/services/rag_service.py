"""RAG Planner arithmetic (WF2). Pure functions, no I/O.

The through-line of this workflow is that each tool's output is the next
one's input: chunks size the vector store, the store's dimensions price the
embeddings, and the whole thing rolls up into a pipeline cost. So the figures
have to agree with each other, not just be individually defensible.
"""

from __future__ import annotations

import math
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Final

from app.schemas.catalog import ModelOut, ToolOut
from app.schemas.tools import ToolOutput, ToolWarning
from app.services.cost_service import DAYS_PER_MONTH

BYTES_PER_FLOAT32: Final = 4
BYTES_PER_GB: Final = Decimal(1024**3)
MICRO: Final = Decimal("0.000001")
CENTS: Final = Decimal("0.01")


def _money(value: Decimal) -> Decimal:
    return value.quantize(MICRO, rounding=ROUND_HALF_UP)


def _usd(value: Decimal) -> str:
    amount = f"{abs(value):,.2f}"
    return f"-${amount}" if value < 0 else f"${amount}"


# ── chunk-estimate ───────────────────────────────────────────────────────────

# Index overhead by type, as a multiple of the raw vector bytes.
#
# HNSW's 1.5x is the graph: bidirectional links across layers, which is the
# 40-60% the module cites and the number people forget. IVF stores coarse
# centroids and inverted lists, which is cheap. Flat is the vectors and
# nothing else, and is only viable below roughly a million vectors.
INDEX_OVERHEAD: Final[dict[str, Decimal]] = {
    "flat": Decimal("1.00"),
    "ivf": Decimal("1.10"),
    "hnsw": Decimal("1.50"),
}

# Where each embedding model does its best work. Chunks well beyond this get
# diluted - the vector averages over too much unrelated text and retrieval
# precision drops - and chunks far below it waste a call.
OPTIMAL_CHUNK_TOKENS: Final = 512


def chunks_per_document(*, doc_tokens: int, chunk_size: int, overlap: int) -> int:
    """Chunks a single document yields under a sliding window.

    Per document, not across the corpus. Real splitters never span a document
    boundary, so a corpus of ten 1,000-token documents at size 500 / overlap
    100 is 3 chunks each and 30 in total - the last window of every document
    is a short tail that still costs a call and still occupies a vector.
    Treating the corpus as one continuous stream gives 25 and quietly
    under-counts by the number of documents.
    """
    if doc_tokens <= chunk_size:
        return 1

    stride = chunk_size - overlap
    if stride <= 0:
        # Guarded rather than trusted: overlap >= size is an infinite loop in
        # every naive implementation of this, and the schema already rejects
        # it, so reaching here means something else is wrong.
        return 1

    return math.ceil((doc_tokens - overlap) / stride)


def chunk_estimate(
    *,
    document_count: int,
    avg_tokens_per_document: int,
    chunk_size: int,
    overlap: int,
    query_type: str = "mixed",
    model: ModelOut | None = None,
) -> ToolOutput:
    """Chunk counts, duplication, and a retrieval-quality score with its basis."""
    per_doc = chunks_per_document(
        doc_tokens=avg_tokens_per_document, chunk_size=chunk_size, overlap=overlap
    )
    total_chunks = per_doc * document_count

    source_tokens = document_count * avg_tokens_per_document
    embedded_tokens = total_chunks * min(chunk_size, avg_tokens_per_document)
    duplication = Decimal(embedded_tokens) / Decimal(source_tokens) if source_tokens else Decimal(1)

    overlap_pct = (
        Decimal(overlap) / Decimal(chunk_size) * Decimal(100) if chunk_size else Decimal(0)
    )
    score, factors = _retrieval_quality(
        chunk_size=chunk_size,
        overlap_pct=overlap_pct,
        query_type=query_type,
        model=model,
    )

    warnings: list[ToolWarning] = []
    if overlap_pct > 30:
        warnings.append(
            ToolWarning(
                level="warning",
                field="overlap",
                message=(
                    f"{overlap_pct:.0f}% overlap embeds the same text "
                    f"{duplication:.2f}x over. Above about 30% the index grows without "
                    "measurably better recall - you are paying twice for the same span."
                ),
            )
        )
    if overlap_pct < 10:
        warnings.append(
            ToolWarning(
                level="info",
                field="overlap",
                message=(
                    "Under 10% overlap loses context at chunk boundaries: a sentence "
                    "split across two chunks is retrievable from neither."
                ),
            )
        )
    if model and model.context_window and chunk_size > model.context_window:
        warnings.append(
            ToolWarning(
                level="critical",
                field="chunk_size",
                message=(
                    f"{chunk_size} tokens exceeds {model.display_name}'s "
                    f"{model.context_window}-token window. These chunks cannot be embedded."
                ),
            )
        )

    window = (model.context_window if model else None) or OPTIMAL_CHUNK_TOKENS * 4
    recommended_size = min(OPTIMAL_CHUNK_TOKENS, window)
    if query_type == "factoid":
        recommended_size = min(recommended_size, 256)
    elif query_type == "synthesis":
        recommended_size = min(1024, window)

    return ToolOutput(
        metrics={
            "total_chunks": total_chunks,
            "chunks_per_document": per_doc,
            "embedded_tokens": embedded_tokens,
            "source_tokens": source_tokens,
            "duplication_factor": duplication.quantize(Decimal("0.01")),
            "overlap_pct": overlap_pct.quantize(Decimal("0.1")),
            "retrieval_quality": score,
            "recommended_chunk_size": recommended_size,
            "recommended_overlap": int(recommended_size * Decimal("0.15")),
        },
        tables={"quality_factors": factors},
        warnings=warnings,
    )


def _retrieval_quality(
    *,
    chunk_size: int,
    overlap_pct: Decimal,
    query_type: str,
    model: ModelOut | None,
) -> tuple[int, list[dict[str, Any]]]:
    """A 0-100 score, and every deduction that produced it.

    The factors table is the point. "Your chunking looks good" invites no
    decision; "-15, chunks are 4x the model's optimal window" tells the user
    what to change. A score without its basis is an opinion with a number
    stuck on it.
    """
    factors: list[dict[str, Any]] = []
    score = 100

    def deduct(points: int, factor: str, detail: str) -> None:
        nonlocal score
        score -= points
        factors.append({"factor": factor, "impact": f"-{points}", "detail": detail})

    window = model.context_window if model else None
    if window and chunk_size > window:
        deduct(40, "Exceeds model window", f"{chunk_size} > {window} tokens; cannot embed")
    elif chunk_size > OPTIMAL_CHUNK_TOKENS * 4:
        deduct(
            20,
            "Chunks are very large",
            f"{chunk_size} tokens dilutes the embedding across too much text",
        )
    elif chunk_size > OPTIMAL_CHUNK_TOKENS * 2:
        deduct(10, "Chunks are large", f"{chunk_size} tokens is above the usual sweet spot")

    if overlap_pct < 10:
        deduct(15, "Overlap too low", "Context is lost at chunk boundaries")
    elif overlap_pct > 30:
        deduct(10, "Overlap too high", "Index bloat with no measurable recall gain")

    if query_type == "factoid" and chunk_size > 512:
        deduct(
            15,
            "Chunks mismatch query type",
            "Short factoid queries retrieve better against small chunks",
        )
    elif query_type == "synthesis" and chunk_size < 256:
        deduct(
            15,
            "Chunks mismatch query type",
            "Synthesis queries need enough context per chunk to reason over",
        )

    if not factors:
        factors.append(
            {"factor": "No penalties", "impact": "0", "detail": "Configuration is sound"}
        )

    return max(0, score), factors


# ── vectordb-estimate ────────────────────────────────────────────────────────


def vectordb_estimate(
    *,
    vector_count: int,
    dimensions: int,
    index_type: str = "hnsw",
    metadata_bytes_per_vector: int = 200,
    replicas: int = 1,
    providers: list[ToolOut] | None = None,
) -> ToolOutput:
    """Storage, index overhead by type, and a costed provider comparison."""
    raw_bytes = Decimal(vector_count) * Decimal(dimensions) * Decimal(BYTES_PER_FLOAT32)
    multiplier = INDEX_OVERHEAD.get(index_type, INDEX_OVERHEAD["hnsw"])
    index_bytes = raw_bytes * (multiplier - Decimal(1))
    metadata_bytes = Decimal(vector_count) * Decimal(metadata_bytes_per_vector)

    per_replica = raw_bytes + index_bytes + metadata_bytes
    total_bytes = per_replica * Decimal(replicas)

    def gb(value: Decimal) -> Decimal:
        return (value / BYTES_PER_GB).quantize(Decimal("0.01"))

    rows: list[dict[str, Any]] = []
    for provider in providers or []:
        facts = provider.facts or {}
        per_m = facts.get("cost_per_m_vectors_month")
        if per_m is None:
            continue
        monthly = Decimal(str(per_m)) * (Decimal(vector_count) / Decimal(1_000_000))
        floor = Decimal(str(facts.get("min_monthly", 0)))
        effective = max(monthly, floor)
        rows.append(
            {
                "provider": provider.name,
                "slug": provider.slug,
                "monthly_cost": _usd(effective.quantize(CENTS)),
                "_sort": effective,
                "at_minimum": "yes" if effective > monthly else "no",
                "managed": "yes" if facts.get("managed") else "no",
                "status": provider.status,
            }
        )
    rows.sort(key=lambda row: row["_sort"])
    for row in rows:
        del row["_sort"]

    warnings: list[ToolWarning] = []
    if index_type == "flat" and vector_count > 1_000_000:
        warnings.append(
            ToolWarning(
                level="warning",
                field="index_type",
                message=(
                    "A flat index scans every vector. Past about a million, query "
                    "latency grows linearly and HNSW is the usual answer despite its "
                    "50% memory premium."
                ),
            )
        )
    if gb(total_bytes) > 100:
        warnings.append(
            ToolWarning(
                level="info",
                message=(
                    f"{gb(total_bytes)} GB is past the point where managed per-vector "
                    "pricing usually loses to self-hosting. Check the comparison below "
                    "against a GPU or memory-optimised instance."
                ),
            )
        )

    return ToolOutput(
        metrics={
            "raw_gb": gb(raw_bytes),
            "index_overhead_gb": gb(index_bytes),
            "metadata_gb": gb(metadata_bytes),
            "total_gb": gb(total_bytes),
            "index_multiplier": multiplier,
            "bytes_per_vector": int(per_replica / Decimal(vector_count)) if vector_count else 0,
            "replicas": replicas,
        },
        tables={
            "providers": rows,
            "breakdown": [
                {"component": "Raw vectors", "gb": str(gb(raw_bytes))},
                {"component": f"{index_type.upper()} index overhead", "gb": str(gb(index_bytes))},
                {"component": "Metadata", "gb": str(gb(metadata_bytes))},
                {"component": f"Replicas (x{replicas})", "gb": str(gb(total_bytes))},
            ],
        },
        warnings=warnings,
    )


# ── pipeline-cost ────────────────────────────────────────────────────────────


def rerank_monthly_cost(
    model: ModelOut,
    *,
    queries_per_month: Decimal,
    documents_per_query: int,
    tokens_per_document: int,
) -> Decimal:
    """Reranking cost, respecting the model's own pricing unit (D-18).

    Cohere bills per 1K searches - one search being one query against up to
    100 documents - while Voyage and Jina bill per token. Both figures live in
    `input_cost_per_1k`, so charging one as if it were the other is a
    thousand-fold error that reads as a plausible price. The unit is on the
    row precisely so this function does not have to guess from the provider.
    """
    if model.price_unit == "searches":
        # The stored figure is already per single search.
        searches = queries_per_month * Decimal(max(1, math.ceil(documents_per_query / 100)))
        return _money(searches * model.input_cost_per_1k)

    tokens = queries_per_month * Decimal(documents_per_query) * Decimal(tokens_per_document)
    return _money(tokens / Decimal(1000) * model.input_cost_per_1k)


def pipeline_cost(
    *,
    document_count: int,
    avg_tokens_per_document: int,
    chunk_size: int,
    overlap: int,
    reindex_per_month: Decimal,
    queries_per_day: int,
    chunks_retrieved: int,
    embedding_model: ModelOut,
    generation_model: ModelOut,
    rerank_model: ModelOut | None = None,
    output_tokens: int = 500,
    vector_store_monthly: Decimal = Decimal(0),
) -> ToolOutput:
    """End-to-end monthly cost, every rate pulled from the catalog.

    The old build asked the user to type unit costs, which made every estimate
    exactly as good as their memory of a pricing page.
    """
    per_doc = chunks_per_document(
        doc_tokens=avg_tokens_per_document, chunk_size=chunk_size, overlap=overlap
    )
    total_chunks = per_doc * document_count
    chunk_tokens = min(chunk_size, avg_tokens_per_document)
    corpus_tokens = Decimal(total_chunks * chunk_tokens)

    queries_per_month = Decimal(queries_per_day) * DAYS_PER_MONTH

    ingestion = _money(corpus_tokens / Decimal(1000) * embedding_model.input_cost_per_1k)
    reindex = _money(ingestion * reindex_per_month)

    # One embedding call per query, for the query text itself. Small, but it
    # scales with traffic rather than corpus size, so at high query volume it
    # overtakes re-indexing.
    query_embedding = _money(
        queries_per_month * Decimal(30) / Decimal(1000) * embedding_model.input_cost_per_1k
    )

    rerank = Decimal(0)
    if rerank_model is not None:
        rerank = rerank_monthly_cost(
            rerank_model,
            queries_per_month=queries_per_month,
            documents_per_query=chunks_retrieved,
            tokens_per_document=chunk_tokens,
        )

    # Generation reads the retrieved chunks plus the question and a prompt.
    context_tokens = Decimal(chunks_retrieved * chunk_tokens + 200)
    generation_input = queries_per_month * context_tokens / Decimal(1000)
    generation_output = queries_per_month * Decimal(output_tokens) / Decimal(1000)
    generation = _money(
        generation_input * generation_model.input_cost_per_1k
        + generation_output * (generation_model.output_cost_per_1k or Decimal(0))
    )

    monthly = reindex + query_embedding + rerank + generation + vector_store_monthly
    per_query = _money(monthly / queries_per_month) if queries_per_month else Decimal(0)

    lines = [
        ("Ingestion (one-off)", ingestion),
        ("Re-indexing", reindex),
        ("Query embedding", query_embedding),
        ("Reranking", rerank),
        ("Generation", generation),
        ("Vector store", vector_store_monthly),
    ]
    breakdown = [
        {
            "line": name,
            "monthly": _usd(value.quantize(CENTS)),
            "share": (
                f"{(value / monthly * Decimal(100)):.1f}%"
                if monthly and name != "Ingestion (one-off)"
                else "—"
            ),
        }
        for name, value in lines
    ]

    recurring = [(name, value) for name, value in lines if name != "Ingestion (one-off)"]
    dominant = max(recurring, key=lambda item: item[1])

    warnings: list[ToolWarning] = []
    if generation > monthly * Decimal("0.7") and monthly:
        warnings.append(
            ToolWarning(
                level="info",
                message=(
                    "Generation is most of this bill. Retrieving fewer chunks or using "
                    "a cheaper generation model moves it far more than any change to "
                    "the embedding or storage side."
                ),
            )
        )
    if rerank_model is not None and rerank > generation:
        warnings.append(
            ToolWarning(
                level="warning",
                message=(
                    "Reranking costs more than generation here. Rerank fewer candidates, "
                    "or check whether the reranker is priced per search rather than per "
                    "token - the two differ by orders of magnitude at this volume."
                ),
            )
        )

    sourced = [embedding_model.id, generation_model.id]
    if rerank_model is not None:
        sourced.append(rerank_model.id)

    return ToolOutput(
        metrics={
            "monthly_cost": _money(monthly),
            "ingestion_cost": ingestion,
            "cost_per_query": per_query,
            "total_chunks": total_chunks,
            "queries_per_month": int(queries_per_month),
            "dominant_cost": dominant[0],
        },
        tables={"breakdown": breakdown},
        series={
            "composition": [
                {"line": name, "cost": str(value.quantize(CENTS))}
                for name, value in recurring
                if value > 0
            ]
        },
        warnings=warnings,
        sourced_from=sourced,
    )


# ── chunking-strategy ────────────────────────────────────────────────────────

STRATEGIES: Final[dict[str, dict[str, Any]]] = {
    "fixed": {
        "name": "Fixed-size",
        "why": "Predictable cost and index size. Splits mid-sentence, which costs recall.",
        "best_for": ["logs", "transcripts"],
    },
    "sentence": {
        "name": "Sentence-aware",
        "why": "Never splits a sentence. Chunk sizes vary, so cost is less predictable.",
        "best_for": ["articles", "support"],
    },
    "paragraph": {
        "name": "Paragraph",
        "why": "Follows the author's own units of meaning. Struggles with very long paragraphs.",
        "best_for": ["articles", "policy"],
    },
    "recursive": {
        "name": "Recursive character",
        "why": (
            "Splits on the largest natural boundary that fits, falling back through "
            "paragraph, sentence, word. The sane default when the corpus is mixed."
        ),
        "best_for": ["mixed", "articles", "support"],
    },
    "markdown": {
        "name": "Markdown-aware",
        "why": "Keeps headings with their content, so a retrieved chunk carries its own context.",
        "best_for": ["docs", "code"],
    },
    "semantic": {
        "name": "Semantic",
        "why": (
            "Splits where the embedding shifts topic. Best recall, and it costs an "
            "embedding pass over the corpus before you have chunked anything."
        ),
        "best_for": ["research", "policy", "mixed"],
    },
}


def chunking_strategy(
    *,
    document_type: str,
    avg_tokens_per_document: int,
    query_pattern: str = "mixed",
    model: ModelOut | None = None,
) -> ToolOutput:
    """Recommend a splitter, and say what each alternative trades away."""
    scores: dict[str, int] = {}
    for key, strategy in STRATEGIES.items():
        score = 50
        if document_type in strategy["best_for"]:
            score += 30
        if key == "recursive":
            score += 10  # the safe default
        if key == "semantic" and avg_tokens_per_document > 50_000:
            score -= 20  # the pre-pass gets expensive on a large corpus
        if key == "markdown" and document_type in {"docs", "code"}:
            score += 15
        if key == "fixed" and query_pattern == "factoid":
            score += 5
        scores[key] = score

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    winner_key, winner_score = ranked[0]
    winner = STRATEGIES[winner_key]

    size = 512 if query_pattern != "synthesis" else 1024
    if query_pattern == "factoid":
        size = 256
    if model and model.context_window:
        size = min(size, model.context_window)

    alternatives = [
        {
            "strategy": STRATEGIES[key]["name"],
            "score": score,
            "tradeoff": STRATEGIES[key]["why"],
        }
        for key, score in ranked[1:4]
    ]

    return ToolOutput(
        metrics={
            "strategy": winner["name"],
            "strategy_key": winner_key,
            "score": winner_score,
            "chunk_size": size,
            "overlap": int(size * Decimal("0.15")),
            "reasoning": winner["why"],
        },
        tables={"alternatives": alternatives},
        warnings=[],
    )


# ── pdf-tokens ───────────────────────────────────────────────────────────────


def pdf_tokens(
    *,
    filename: str,
    page_texts: list[str],
    model: ModelOut,
    chunk_size: int = 512,
    overlap: int = 76,
) -> ToolOutput:
    """Token count and ingestion cost for one uploaded document.

    Takes already-extracted page text rather than bytes: extraction is I/O and
    belongs at the edge, and keeping this function pure means the token
    arithmetic is unit-testable without a PDF fixture.
    """
    # Heuristic, and labelled as such on the response. The real tokenizer
    # arrives with M16; over-claiming precision now would be worse than the
    # approximation, because nobody re-checks a number that looked exact.
    total_chars = sum(len(text) for text in page_texts)
    tokens = math.ceil(total_chars / 4) if total_chars else 0

    empty_pages = [index + 1 for index, text in enumerate(page_texts) if not text.strip()]
    chunks = chunks_per_document(doc_tokens=tokens, chunk_size=chunk_size, overlap=overlap)
    cost = _money(Decimal(tokens) / Decimal(1000) * model.input_cost_per_1k)

    per_page = [
        {
            "page": index + 1,
            "characters": len(text),
            "tokens": math.ceil(len(text) / 4),
            "extracted": "no" if not text.strip() else "yes",
        }
        for index, text in enumerate(page_texts)
    ]

    warnings: list[ToolWarning] = [
        ToolWarning(
            level="info",
            message=(
                "The file was read in memory and discarded. Nothing was written to disk "
                "and no copy of the document exists on this server."
            ),
        ),
        ToolWarning(
            level="info",
            message=(
                "Token counts are estimated at 4 characters per token. Real tokenizer "
                "counts arrive with the AI service layer."
            ),
        ),
    ]
    if empty_pages:
        shown = ", ".join(str(page) for page in empty_pages[:10])
        warnings.append(
            ToolWarning(
                level="warning",
                message=(
                    f"{len(empty_pages)} page(s) yielded no text ({shown}"
                    f"{'…' if len(empty_pages) > 10 else ''}). These are probably scanned "
                    "images and need OCR before they can be embedded."
                ),
            )
        )

    return ToolOutput(
        metrics={
            "filename": filename,
            "pages": len(page_texts),
            "characters": total_chars,
            "tokens": tokens,
            "estimated_chunks": chunks,
            "ingestion_cost": cost,
            "pages_needing_ocr": len(empty_pages),
        },
        tables={"pages": per_page},
        warnings=warnings,
        sourced_from=[model.id],
    )
