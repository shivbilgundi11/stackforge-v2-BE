"""The seven-key tool response contract.

Seven keys, the same seven for all 28 tools:

    metrics · tables · series · artifacts · warnings · provenance · ai

The result renderer switches on block kind, not on tool identity, so adding
tool 29 changes no rendering code. That property is the entire point of fixing
the shape, and it only holds if nothing is ever added to this list for one
tool's convenience — the escape hatch (`result.component` on the frontend)
exists so that a tool with a genuinely bespoke display does not need to.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, PlainSerializer

MetricValue = Annotated[
    Decimal | int | str,
    PlainSerializer(lambda v: format(v, "f") if isinstance(v, Decimal) else v),
]

WarningLevel = Literal["info", "warning", "critical"]
RunSource = Literal["rule_based", "ai_synthesis", "hybrid"]


class ToolWarning(BaseModel):
    """Not `Warning` — that name is a builtin exception class."""

    level: WarningLevel = "warning"
    message: str
    field: str | None = None


class Artifact(BaseModel):
    type: str
    format: Literal["markdown", "json", "yaml", "csv", "mermaid", "text", "code"]
    filename: str
    content: str
    language: str | None = None


class ProvenanceSource(BaseModel):
    name: str
    url: str
    last_verified_at: datetime
    age_days: int
    variant: Literal["fresh", "aging", "stale"]


class Provenance(BaseModel):
    """The verification dates of the specific rows this run touched.

    Not a global "catalog last updated" date — that would be a number the user
    cannot act on. `oldest_verified_at` is what the chip shows, because the
    trustworthiness of a result is the trustworthiness of its worst input.
    """

    oldest_verified_at: datetime | None = None
    variant: Literal["fresh", "aging", "stale"] = "fresh"
    sources: list[ProvenanceSource] = Field(default_factory=list)


class AiMeta(BaseModel):
    model: str
    prompt_version: str
    input_tokens: int
    output_tokens: int
    cost_usd: Annotated[Decimal, PlainSerializer(lambda v: format(v, "f"), return_type=str)]
    latency_ms: int


class ToolOutput(BaseModel):
    """What a `compute` function returns.

    Deliberately *not* the wire shape: `run_id`, `source`, `provenance`, and
    `ai` are attached by `run_tool`, so a compute function stays a pure
    function of its inputs and the catalog rows it was handed. That constraint
    is what makes "assert an exact computed number" a one-line unit test.
    """

    metrics: dict[str, MetricValue] = Field(default_factory=dict)
    tables: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)
    series: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)
    artifacts: list[Artifact] = Field(default_factory=list)
    warnings: list[ToolWarning] = Field(default_factory=list)
    # Catalog row ids the computation read. `run_tool` turns these into the
    # provenance block, so a compute function never has to build one.
    sourced_from: list[str] = Field(default_factory=list, exclude=True)


class ToolRunOut(BaseModel):
    """The wire shape. `ToolOutput` plus what the engine attaches."""

    run_id: str
    tool_slug: str
    source: RunSource
    duration_ms: int
    created_at: datetime

    metrics: dict[str, MetricValue] = Field(default_factory=dict)
    tables: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)
    series: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)
    artifacts: list[Artifact] = Field(default_factory=list)
    warnings: list[ToolWarning] = Field(default_factory=list)
    provenance: Provenance = Field(default_factory=Provenance)
    ai: AiMeta | None = None


class QuotaOut(BaseModel):
    """Returned in the 402 body so the upgrade dialog shows real numbers.

    "You have hit your limit" with no figures is a dead end; "42 of 42 runs
    used, resets in 6 hours" tells the user whether to upgrade or wait.
    """

    metric: str
    limit: int
    used: int
    remaining: int
    period: str
    resets_at: datetime
    plan: str
