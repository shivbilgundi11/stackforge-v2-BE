"""Catalog wire shapes.

Prices serialise as **decimal strings**, not JSON numbers. `0.00005` survives
`json.dumps` in Python and then loses precision in JavaScript's `Number`; a
string arrives intact and the frontend formats it with `Intl.NumberFormat`.
That is D-08 carried all the way to the wire.
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PlainSerializer, field_validator

# Serialise Decimal as a plain string, always.
Price = Annotated[Decimal, PlainSerializer(lambda v: format(v, "f"), return_type=str)]
OptPrice = Annotated[
    Decimal | None,
    PlainSerializer(lambda v: None if v is None else format(v, "f"), return_type=str | None),
]


class ProvenanceOut(BaseModel):
    """Attached to everything priced.

    `variant` is computed server-side so the freshness thresholds live in one
    place. A frontend that decides "is 8 days stale?" for itself will disagree
    with the backend the first time someone edits one and not the other.
    """

    last_verified_at: datetime
    age_days: int
    variant: Literal["fresh", "aging", "stale"]
    source_name: str
    source_url: str
    source_kind: str


class ModelOut(BaseModel):
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

    id: str
    provider: str
    model_id: str
    display_name: str
    family: str
    input_cost_per_1k: Price
    output_cost_per_1k: OptPrice = None
    cached_input_cost_per_1k: OptPrice = None
    context_window: int | None = None
    max_output_tokens: int | None = None
    dimensions: int | None = None
    capabilities: dict[str, Any] = Field(default_factory=dict)
    tokenizer: str | None = None
    # "tokens" everywhere except Cohere rerank, which is "searches" (D-18).
    # On the wire because a client comparing two rerankers has to know, and
    # inferring it from the provider name is how this went wrong the first
    # time.
    price_unit: str = "tokens"
    status: str
    status_reason: str | None = None
    provenance: ProvenanceOut


class GpuOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    provider: str
    instance_name: str
    gpu_model: str
    gpu_count: int
    vram_gb: int
    vram_total_gb: int
    vcpu: int | None = None
    ram_gb: int | None = None
    hourly_cost_usd: Price
    monthly_cost_usd: Price
    region: str
    spot: bool
    provenance: ProvenanceOut


class ToolOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    slug: str
    name: str
    category: str
    description: str
    status: str
    status_reason: str | None = None
    alternatives: list[str] = Field(default_factory=list)
    maturity_score: int
    license: str | None = None
    self_hostable: bool
    pricing_model: str | None = None
    docs_url: str | None = None
    pricing_url: str | None = None
    repo_url: str | None = None
    tags: list[str] = Field(default_factory=list)
    use_cases: list[str] = Field(default_factory=list)
    facts: dict[str, Any] = Field(default_factory=dict)
    last_reviewed_at: datetime


class GraveyardEntryOut(ToolOut):
    """A buried tool. Same shape, but the reason is guaranteed present."""

    status_reason: str
    alternative_tools: list[ToolOut] = Field(default_factory=list)


class ArchitectureOut(BaseModel):
    """A model's shape, for VRAM estimation.

    `uses_gqa` is computed rather than left to the client: the KV cache scales
    with `kv_heads`, and a caller that compares the wrong pair of fields gets
    an answer that is wrong by the group size.
    """

    key: str
    name: str
    family: str
    params_b: float
    layers: int
    hidden_size: int
    heads: int
    kv_heads: int
    head_dim: int
    max_context: int
    uses_gqa: bool


class CompatibilityPairOut(BaseModel):
    tool_a: str
    tool_b: str
    score: int
    dimensions: dict[str, int]
    notes: str | None = None
    warnings: list[str] = Field(default_factory=list)


class CompatibilityOut(BaseModel):
    """Pairwise scores plus one overall figure for the whole set.

    `overall` is the *minimum* pair score, not the mean. A stack is only as
    compatible as its worst pairing — averaging lets four good pairs hide one
    combination that does not work, which is precisely the thing this endpoint
    exists to surface.
    """

    tools: list[str]
    pairs: list[CompatibilityPairOut]
    overall: int
    weakest_pair: CompatibilityPairOut | None = None
    warnings: list[str] = Field(default_factory=list)
    missing_pairs: list[list[str]] = Field(
        default_factory=list,
        description="Pairs with no scored row. Reported rather than assumed compatible.",
    )


class PricingHistoryOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    entity_type: str
    entity_id: str
    field: str
    old_value: OptPrice = None
    new_value: OptPrice = None
    pct_change: OptPrice = None
    applied: bool
    detected_at: datetime


class FlagIn(BaseModel):
    entity_type: Literal["model", "gpu", "tool"]
    entity_id: str = Field(min_length=1, max_length=64)
    field: str | None = Field(default=None, max_length=60)
    suggested_value: str | None = Field(default=None, max_length=120)
    note: str | None = Field(default=None, max_length=2000)
    source_url: str | None = Field(default=None, max_length=500)

    @field_validator("note", "suggested_value", "source_url", mode="before")
    @classmethod
    def _blank_to_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value


class FlagOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    entity_type: str
    entity_id: str
    status: str
    created_at: datetime


class DriftEntryOut(BaseModel):
    entity_type: str
    entity_id: str
    label: str
    field: str
    old_value: Price
    new_value: Price
    pct_change: Price
    source_name: str
    detected_at: datetime


class RefreshResultOut(BaseModel):
    """The editorial queue.

    `sources_skipped` is the honest headline while prices are hand-verified:
    it is the count of sources no machine reads, which is all of them.
    """

    checked: int
    changes_detected: int
    sources_failed: int
    sources_skipped: int = 0
    stale_rows: int = 0
    entries: list[DriftEntryOut] = Field(default_factory=list)
    ran_at: datetime


class CatalogStatsOut(BaseModel):
    models: int
    gpus: int
    tools: int
    compatibility_pairs: int
    oldest_verification: date | None = None
    stale_rows: int
