"""RAG Planner request shapes (WF2)."""

from __future__ import annotations

from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, Field, model_validator

QueryType = Literal["factoid", "synthesis", "mixed"]
IndexType = Literal["flat", "ivf", "hnsw"]
Sensitivity = Literal["public", "internal", "internal-only", "restricted"]
Scale = Literal["small", "medium", "large", "xlarge"]


class ChunkEstimateIn(BaseModel):
    document_count: int = Field(ge=1, le=1_000_000_000)
    avg_tokens_per_document: int = Field(ge=1, le=10_000_000)
    chunk_size: int = Field(default=512, ge=16, le=32_000)
    overlap: int = Field(default=76, ge=0, le=16_000)
    query_type: QueryType = "mixed"
    model_id: str | None = None

    @model_validator(mode="after")
    def _overlap_below_chunk_size(self) -> ChunkEstimateIn:
        # Overlap >= chunk size is a splitter that never advances. Rejected
        # here rather than clamped, because clamping would silently produce a
        # chunk count for a configuration the user cannot actually run.
        if self.overlap >= self.chunk_size:
            raise ValueError("Overlap must be smaller than chunk size.")
        return self


class VectorDbEstimateIn(BaseModel):
    vector_count: int = Field(ge=1, le=10_000_000_000)
    dimensions: int = Field(ge=1, le=16_384)
    index_type: IndexType = "hnsw"
    metadata_bytes_per_vector: int = Field(default=200, ge=0, le=1_000_000)
    replicas: int = Field(default=1, ge=1, le=20)


class PipelineCostIn(BaseModel):
    document_count: int = Field(ge=1, le=1_000_000_000)
    avg_tokens_per_document: int = Field(ge=1, le=10_000_000)
    chunk_size: int = Field(default=512, ge=16, le=32_000)
    overlap: int = Field(default=76, ge=0, le=16_000)
    reindex_per_month: Decimal = Field(default=Decimal(1), ge=0, le=100)
    queries_per_day: int = Field(ge=1, le=100_000_000)
    chunks_retrieved: int = Field(default=5, ge=1, le=200)
    embedding_model_id: str
    generation_model_id: str
    rerank_model_id: str | None = None
    output_tokens: int = Field(default=500, ge=0, le=200_000)
    vector_store_monthly: Decimal = Field(default=Decimal(0), ge=0)

    @model_validator(mode="after")
    def _overlap_below_chunk_size(self) -> PipelineCostIn:
        if self.overlap >= self.chunk_size:
            raise ValueError("Overlap must be smaller than chunk size.")
        return self


class ChunkingStrategyIn(BaseModel):
    document_type: Literal[
        "articles", "docs", "code", "support", "logs", "policy", "research", "mixed"
    ]
    avg_tokens_per_document: int = Field(ge=1, le=10_000_000)
    query_pattern: QueryType = "mixed"
    model_id: str | None = None


class RagArchitectureIn(BaseModel):
    use_case: Literal["docs", "support", "code", "research", "policy", "mixed"] = "mixed"
    corpus_documents: int = Field(ge=1, le=1_000_000_000)
    sensitivity: Sensitivity = "internal"
    latency_target_ms: int = Field(default=2000, ge=50, le=60_000)
    scale: Scale = "medium"
    team_skill: Literal["beginner", "intermediate", "advanced"] = "intermediate"
