"""Cost Planner request shapes."""

from __future__ import annotations

from decimal import Decimal

from pydantic import BaseModel, Field


class LlmPricingIn(BaseModel):
    model_id: str = Field(description="Canonical model id, e.g. gpt-4o-mini")
    input_tokens: int = Field(ge=0, le=10_000_000, description="Input tokens per request")
    output_tokens: int = Field(ge=0, le=1_000_000, description="Output tokens per request")
    requests_per_day: int = Field(ge=0, le=100_000_000)
    cached_input_ratio: Decimal = Field(
        default=Decimal(0),
        ge=0,
        le=1,
        description="Fraction of input tokens served from the prompt cache (0-1).",
    )
    compare_provider: str | None = Field(
        default=None, description="Limit the alternatives table to one provider."
    )


class TokenCalculatorIn(BaseModel):
    text: str = Field(max_length=2_000_000)
    model_id: str
    output_tokens: int = Field(default=0, ge=0, le=1_000_000)


class EmbeddingCostIn(BaseModel):
    model_id: str
    document_count: int = Field(ge=1, le=1_000_000_000)
    avg_tokens_per_document: int = Field(ge=1, le=10_000_000)
    reembeds_per_month: int = Field(default=1, ge=0, le=1000)
    chunk_overlap_pct: Decimal = Field(default=Decimal(0), ge=0, le=90)


class WorkloadLineIn(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    model_id: str
    requests_per_day: int = Field(ge=0, le=100_000_000)
    input_tokens: int = Field(ge=0, le=10_000_000)
    output_tokens: int = Field(ge=0, le=1_000_000)


class BudgetEstimatorIn(BaseModel):
    lines: list[WorkloadLineIn] = Field(min_length=1, max_length=25)
    monthly_growth_pct: Decimal = Field(default=Decimal(0), ge=-50, le=200)
    infrastructure_monthly: Decimal = Field(default=Decimal(0), ge=0)
    embedding_monthly: Decimal = Field(default=Decimal(0), ge=0)
    user_count: int | None = Field(default=None, ge=1)
