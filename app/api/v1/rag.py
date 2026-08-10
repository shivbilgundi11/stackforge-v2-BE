"""RAG Planner endpoints (WF2).

`pdf-tokens` is the one endpoint in the product that takes a file, and it is
the one that has to be most careful. The cap is enforced while reading rather
than after, so an oversized upload is refused without ever being fully
buffered, and extraction runs in a worker thread so a 300-page document does
not stall the event loop for everyone else.

**Nothing is written to disk.** The bytes are read into memory, parsed, and
dropped when the request ends. A planning tool that quietly retains a
confidential corpus is a liability dressed as a feature.
"""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, File, Form, UploadFile
from pydantic import BaseModel

from app.api.deps import Db, RunIdentity
from app.core.errors import AppError, ValidationFailed
from app.core.logging import get_logger
from app.core.responses import Envelope, ok
from app.schemas.rag import (
    ChunkEstimateIn,
    ChunkingStrategyIn,
    PipelineCostIn,
    RagArchitectureIn,
    VectorDbEstimateIn,
)
from app.schemas.tools import ToolOutput, ToolRunOut, ToolWarning
from app.services import (
    ai_service,
    catalog_service,
    rag_architecture_service,
    rag_service,
    tool_service,
)

logger = get_logger("api.rag")

router = APIRouter(tags=["rag"])

WORKFLOW = "rag"

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
UPLOAD_CHUNK = 1024 * 1024


class PayloadTooLarge(AppError):
    code = "PAYLOAD_TOO_LARGE"
    http_status = 413
    default_message = "That file is larger than the upload limit."


class _PdfRunPayload(BaseModel):
    """What gets logged for a PDF run — metadata, never the document."""

    filename: str
    model_id: str
    chunk_size: int
    size_bytes: int


@router.post("/chunk-estimate", response_model=Envelope[ToolRunOut], name="run_chunk_estimate")
async def run_chunk_estimate(
    db: Db, identity: RunIdentity, payload: ChunkEstimateIn
) -> dict[str, Any]:
    model = await catalog_service.get_model(db, payload.model_id) if payload.model_id else None
    result = await tool_service.run_tool(
        db,
        slug="chunk-estimate",
        workflow=WORKFLOW,
        payload=payload,
        identity=identity,
        compute=lambda: rag_service.chunk_estimate(
            document_count=payload.document_count,
            avg_tokens_per_document=payload.avg_tokens_per_document,
            chunk_size=payload.chunk_size,
            overlap=payload.overlap,
            query_type=payload.query_type,
            model=model,
        ),
    )
    return ok(result)


@router.post(
    "/vectordb-estimate", response_model=Envelope[ToolRunOut], name="run_vectordb_estimate"
)
async def run_vectordb_estimate(
    db: Db, identity: RunIdentity, payload: VectorDbEstimateIn
) -> dict[str, Any]:
    providers = await catalog_service.list_tools(db, category="vector-db")

    result = await tool_service.run_tool(
        db,
        slug="vectordb-estimate",
        workflow=WORKFLOW,
        payload=payload,
        identity=identity,
        compute=lambda: rag_service.vectordb_estimate(
            vector_count=payload.vector_count,
            dimensions=payload.dimensions,
            index_type=payload.index_type,
            metadata_bytes_per_vector=payload.metadata_bytes_per_vector,
            replicas=payload.replicas,
            providers=providers,
        ),
    )
    return ok(result)


@router.post("/pipeline-cost", response_model=Envelope[ToolRunOut], name="run_pipeline_cost")
async def run_pipeline_cost(
    db: Db, identity: RunIdentity, payload: PipelineCostIn
) -> dict[str, Any]:
    embedding = await catalog_service.get_model(db, payload.embedding_model_id)
    generation = await catalog_service.get_model(db, payload.generation_model_id)
    reranker = (
        await catalog_service.get_model(db, payload.rerank_model_id)
        if payload.rerank_model_id
        else None
    )

    if embedding.family != "embedding":
        raise ValidationFailed.on_field(
            "embedding_model_id",
            f"{embedding.display_name} is a {embedding.family} model, not an embedding model.",
        )
    if reranker is not None and reranker.family != "rerank":
        raise ValidationFailed.on_field(
            "rerank_model_id",
            f"{reranker.display_name} is a {reranker.family} model, not a reranker.",
        )

    result = await tool_service.run_tool(
        db,
        slug="pipeline-cost",
        workflow=WORKFLOW,
        payload=payload,
        identity=identity,
        compute=lambda: rag_service.pipeline_cost(
            document_count=payload.document_count,
            avg_tokens_per_document=payload.avg_tokens_per_document,
            chunk_size=payload.chunk_size,
            overlap=payload.overlap,
            reindex_per_month=payload.reindex_per_month,
            queries_per_day=payload.queries_per_day,
            chunks_retrieved=payload.chunks_retrieved,
            embedding_model=embedding,
            generation_model=generation,
            rerank_model=reranker,
            output_tokens=payload.output_tokens,
            vector_store_monthly=payload.vector_store_monthly,
        ),
    )
    return ok(result)


@router.post(
    "/chunking-strategy", response_model=Envelope[ToolRunOut], name="run_chunking_strategy"
)
async def run_chunking_strategy(
    db: Db, identity: RunIdentity, payload: ChunkingStrategyIn
) -> dict[str, Any]:
    model = await catalog_service.get_model(db, payload.model_id) if payload.model_id else None
    result = await tool_service.run_tool(
        db,
        slug="chunking-strategy",
        workflow=WORKFLOW,
        payload=payload,
        identity=identity,
        compute=lambda: rag_service.chunking_strategy(
            document_type=payload.document_type,
            avg_tokens_per_document=payload.avg_tokens_per_document,
            query_pattern=payload.query_pattern,
            model=model,
        ),
    )
    return ok(result)


@router.post("/architecture", response_model=Envelope[ToolRunOut], name="run_rag_architecture")
async def run_rag_architecture(
    db: Db, identity: RunIdentity, payload: RagArchitectureIn
) -> dict[str, Any]:
    catalog = await catalog_service.list_tools(db)
    rerankers = await catalog_service.list_models(db, family="rerank")

    result = await tool_service.run_tool(
        db,
        slug="rag-architecture",
        workflow=WORKFLOW,
        payload=payload,
        identity=identity,
        compute=lambda: rag_architecture_service.design(
            use_case=payload.use_case,
            corpus_documents=payload.corpus_documents,
            sensitivity=payload.sensitivity,
            latency_target_ms=payload.latency_target_ms,
            scale=payload.scale,
            team_skill=payload.team_skill,
            catalog=catalog,
            rerank_models=rerankers,
        ),
        # Synthesis over the selection the engine already made. With no key,
        # no quota, or a failed call this returns None and the rule-written
        # summary ships unchanged, marked `rule_based` — D-06.
        enrich=ai_service.enrichment(
            db,
            purpose="rag_architecture",
            identity=identity,
            tool_slug="rag-architecture",
            variables=payload.model_dump(mode="json"),
            apply=_apply_rag_commentary,
        ),
    )
    return ok(result)


def _apply_rag_commentary(output: ToolOutput, data: dict[str, Any]) -> None:
    """Merge the written commentary into the rule result.

    The components, the diagram, and the constraints are untouched — only the
    summary is replaced and the commentary appended. A synthesis pass that
    could edit the engine's selection would be the thing D-06 exists to
    prevent.
    """
    if summary := str(data.get("summary") or "").strip():
        output.metrics["summary"] = summary
    if why := str(data.get("why") or "").strip():
        output.metrics["rationale"] = why
    for item in data.get("watch_out_for") or []:
        output.warnings.append(ToolWarning(level="info", message=str(item)))
    if measure := data.get("measure_first"):
        output.tables["measure_first"] = [{"step": str(item)} for item in measure]


@router.post("/pdf-tokens", response_model=Envelope[ToolRunOut], name="run_pdf_tokens")
async def run_pdf_tokens(
    db: Db,
    identity: RunIdentity,
    file: UploadFile = File(description="PDF, up to 25 MB. Never stored."),
    model_id: str = Form(default="text-embedding-3-small"),
    chunk_size: int = Form(default=512),
) -> dict[str, Any]:
    if file.content_type not in {"application/pdf", "application/x-pdf", None}:
        raise ValidationFailed.on_field("file", f"Expected a PDF, got {file.content_type}.")

    data = await _read_capped(file)
    model = await catalog_service.get_model(db, model_id)

    # pypdf is synchronous and CPU-bound. A 300-page document parsed on the
    # event loop blocks every other request for the duration.
    try:
        pages = await asyncio.to_thread(_extract_pages, data)
    except Exception as exc:
        logger.warning("rag.pdf_unreadable", error=str(exc), size=len(data))
        raise ValidationFailed.on_field(
            "file",
            "That file could not be read as a PDF. Encrypted files need their "
            "password removed first.",
        ) from exc

    if not pages:
        raise ValidationFailed.on_field("file", "That PDF has no pages.")

    filename = file.filename or "document.pdf"
    # The payload recorded against the run deliberately excludes the file. The
    # run log is not a place to accumulate copies of user documents.
    payload = _PdfRunPayload(
        filename=filename, model_id=model_id, chunk_size=chunk_size, size_bytes=len(data)
    )

    result = await tool_service.run_tool(
        db,
        slug="pdf-tokens",
        workflow=WORKFLOW,
        payload=payload,
        identity=identity,
        compute=lambda: rag_service.pdf_tokens(
            filename=filename,
            page_texts=pages,
            model=model,
            chunk_size=chunk_size,
            overlap=int(chunk_size * 0.15),
        ),
    )
    return ok(result)


async def _read_capped(file: UploadFile) -> bytes:
    """Read up to the cap, refusing anything larger.

    Checked while streaming rather than from `content-length`, which a client
    controls and can simply lie about. Buffering first and measuring after
    would mean a 2 GB upload is already in memory by the time it is rejected.
    """
    buffer = bytearray()
    while chunk := await file.read(UPLOAD_CHUNK):
        buffer.extend(chunk)
        if len(buffer) > MAX_UPLOAD_BYTES:
            raise PayloadTooLarge(
                f"That file is over the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit."
            )
    if not buffer:
        raise ValidationFailed.on_field("file", "The uploaded file is empty.")
    return bytes(buffer)


def _extract_pages(data: bytes) -> list[str]:
    """Page text, in order. Runs in a worker thread."""
    import io

    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    return [page.extract_text() or "" for page in reader.pages]
