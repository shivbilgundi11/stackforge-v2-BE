"""The catalog: model pricing, GPU pricing, the tool catalog, compatibility.

Every priced row points at a `data_sources` entry and carries a
`last_verified_at`. Those two columns are the difference between a number a
senior engineer will plan against and a number they will ignore.

Money is `NUMERIC(14, 6)` throughout, never float. Per-1k-token prices need six
decimals: GPT-5 nano input is $0.00005 per 1k, and binary floating point cannot
represent that exactly.
"""

from __future__ import annotations

import enum
from datetime import datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base, TimestampMixin, new_id

# Every price column in this module. Six decimal places, exact.
Money = Numeric(14, 6)


class ModelFamily(str, enum.Enum):
    CHAT = "chat"
    EMBEDDING = "embedding"
    RERANK = "rerank"
    IMAGE = "image"
    AUDIO = "audio"


class LifecycleStatus(str, enum.Enum):
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    RETIRED = "retired"


class ToolCategory(str, enum.Enum):
    VECTOR_DB = "vector-db"
    LLM_PROVIDER = "llm-provider"
    AGENT_FRAMEWORK = "agent-framework"
    RAG_FRAMEWORK = "rag-framework"
    ORCHESTRATION = "orchestration"
    OBSERVABILITY = "observability"
    DEPLOYMENT = "deployment"
    DATABASE = "database"
    CACHE = "cache"


class ToolStatus(str, enum.Enum):
    RECOMMENDED = "recommended"
    STABLE = "stable"
    CAUTION = "caution"
    DEPRECATED = "deprecated"
    NOT_FOR_PRODUCTION = "not_for_production"


class SourceKind(str, enum.Enum):
    API = "api"
    SCRAPE = "scrape"
    MANUAL = "manual"


class PricedEntity(str, enum.Enum):
    MODEL = "model"
    GPU = "gpu"


class FlagStatus(str, enum.Enum):
    OPEN = "open"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class DataSource(Base, TimestampMixin):
    """Where a number came from.

    `failure_count` is the verification job's memory. Three consecutive
    failures means the page moved or the format changed, and a human needs to
    look — one failure is a flaky network.
    """

    __tablename__ = "data_sources"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("src"))
    slug: Mapped[str] = mapped_column(String(80), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(160), nullable=False)
    url: Mapped[str] = mapped_column(Text, nullable=False)
    kind: Mapped[SourceKind] = mapped_column(
        Enum(SourceKind, name="source_kind", values_callable=lambda e: [m.value for m in e]),
        nullable=False,
        default=SourceKind.MANUAL,
    )

    last_success_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_failure_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    failure_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)


class ModelPricing(Base, TimestampMixin):
    """One priced model.

    Prices are per 1k tokens, because that is the unit the calculators work in
    and converting at read time would mean every consumer repeats the same
    division. Providers publish per-1M; the seed divides once.
    """

    __tablename__ = "model_pricing"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("mdl"))
    provider: Mapped[str] = mapped_column(String(60), nullable=False)
    model_id: Mapped[str] = mapped_column(String(120), nullable=False)
    display_name: Mapped[str] = mapped_column(String(160), nullable=False)
    family: Mapped[ModelFamily] = mapped_column(
        Enum(ModelFamily, name="model_family", values_callable=lambda e: [m.value for m in e]),
        nullable=False,
    )

    input_cost_per_1k: Mapped[Decimal] = mapped_column(Money, nullable=False)
    output_cost_per_1k: Mapped[Decimal | None] = mapped_column(Money)
    cached_input_cost_per_1k: Mapped[Decimal | None] = mapped_column(Money)

    context_window: Mapped[int | None] = mapped_column(Integer)
    max_output_tokens: Mapped[int | None] = mapped_column(Integer)
    dimensions: Mapped[int | None] = mapped_column(Integer)

    capabilities: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    tokenizer: Mapped[str | None] = mapped_column(String(80))

    status: Mapped[LifecycleStatus] = mapped_column(
        Enum(
            LifecycleStatus,
            name="lifecycle_status",
            values_callable=lambda e: [m.value for m in e],
        ),
        nullable=False,
        default=LifecycleStatus.ACTIVE,
    )
    status_reason: Mapped[str | None] = mapped_column(Text)

    source_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("data_sources.id", ondelete="RESTRICT"), nullable=False
    )
    # Not nullable on purpose. A row with no verification date cannot render a
    # provenance chip, and a chip that sometimes disappears teaches people to
    # stop reading it.
    last_verified_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    __table_args__ = (
        UniqueConstraint("provider", "model_id", name="uq_model_pricing_provider_model_id"),
        Index("ix_model_pricing_family_status", "family", "status"),
        CheckConstraint("input_cost_per_1k >= 0", name="input_cost_non_negative"),
        CheckConstraint(
            "output_cost_per_1k IS NULL OR output_cost_per_1k >= 0",
            name="output_cost_non_negative",
        ),
    )

    @property
    def blended_cost_per_1k(self) -> Decimal:
        """A 3:1 input:output blend — the shape of a typical chat workload.

        Used only for ranking and display. Every actual estimate computes from
        the real token split the user supplied.
        """
        if self.output_cost_per_1k is None:
            return self.input_cost_per_1k
        return (self.input_cost_per_1k * 3 + self.output_cost_per_1k) / 4


class GpuPricing(Base, TimestampMixin):
    __tablename__ = "gpu_pricing"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("gpu"))
    provider: Mapped[str] = mapped_column(String(60), nullable=False)
    instance_name: Mapped[str] = mapped_column(String(120), nullable=False)
    gpu_model: Mapped[str] = mapped_column(String(80), nullable=False)
    gpu_count: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    vram_gb: Mapped[int] = mapped_column(Integer, nullable=False)
    vcpu: Mapped[int | None] = mapped_column(Integer)
    ram_gb: Mapped[int | None] = mapped_column(Integer)

    hourly_cost_usd: Mapped[Decimal] = mapped_column(Money, nullable=False)
    region: Mapped[str] = mapped_column(String(60), nullable=False)
    spot: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    source_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("data_sources.id", ondelete="RESTRICT"), nullable=False
    )
    last_verified_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    __table_args__ = (
        UniqueConstraint(
            "provider",
            "instance_name",
            "region",
            "spot",
            name="uq_gpu_pricing_provider_instance_name_region_spot",
        ),
        Index("ix_gpu_pricing_vram_gb", "vram_gb"),
        CheckConstraint("hourly_cost_usd >= 0", name="hourly_cost_non_negative"),
    )

    @property
    def vram_total_gb(self) -> int:
        return self.vram_gb * self.gpu_count


class Tool(Base, TimestampMixin):
    """A catalog entry.

    `status` + `status_reason` *is* the Tool Graveyard. Editorial staff change
    a status and write a sentence; the Graveyard page and every stack warning
    change on the next request, with no deploy. That is the requirement, and it
    is why status is a column rather than a constant in a Python file.
    """

    __tablename__ = "tool_catalog"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("tool"))
    slug: Mapped[str] = mapped_column(String(80), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(160), nullable=False)
    category: Mapped[ToolCategory] = mapped_column(
        Enum(ToolCategory, name="tool_category", values_callable=lambda e: [m.value for m in e]),
        nullable=False,
    )
    description: Mapped[str] = mapped_column(Text, nullable=False)

    status: Mapped[ToolStatus] = mapped_column(
        Enum(ToolStatus, name="tool_status", values_callable=lambda e: [m.value for m in e]),
        nullable=False,
        default=ToolStatus.STABLE,
    )
    status_reason: Mapped[str | None] = mapped_column(Text)
    alternatives: Mapped[list[str]] = mapped_column(ARRAY(String(80)), nullable=False, default=list)

    maturity_score: Mapped[int] = mapped_column(Integer, nullable=False, default=50)
    license: Mapped[str | None] = mapped_column(String(60))
    self_hostable: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    pricing_model: Mapped[str | None] = mapped_column(String(60))

    docs_url: Mapped[str | None] = mapped_column(Text)
    pricing_url: Mapped[str | None] = mapped_column(Text)
    repo_url: Mapped[str | None] = mapped_column(Text)

    tags: Mapped[list[str]] = mapped_column(ARRAY(String(60)), nullable=False, default=list)
    use_cases: Mapped[list[str]] = mapped_column(ARRAY(String(60)), nullable=False, default=list)
    # Free-form facts the comparison engine scores against — see
    # `app/data/compare_criteria.py`. Kept as JSONB so a new criterion is a data
    # change, not a migration.
    facts: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)

    last_reviewed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    reviewed_by: Mapped[str | None] = mapped_column(String(120))

    __table_args__ = (
        Index("ix_tool_catalog_category_status", "category", "status"),
        Index("ix_tool_catalog_tags", "tags", postgresql_using="gin"),
        Index("ix_tool_catalog_use_cases", "use_cases", postgresql_using="gin"),
        CheckConstraint(
            "maturity_score BETWEEN 0 AND 100", name="maturity_score_between_0_and_100"
        ),
    )

    @property
    def is_buried(self) -> bool:
        return self.status in (ToolStatus.DEPRECATED, ToolStatus.NOT_FOR_PRODUCTION)


class Compatibility(Base, TimestampMixin):
    """A scored pair.

    Stored once, with `tool_a_slug < tool_b_slug` enforced by a check
    constraint. A symmetric relation stored twice eventually disagrees with
    itself, and then the answer depends on argument order — which is exactly
    the bug nobody thinks to test for.
    """

    __tablename__ = "compatibility_matrix"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("cmp"))
    tool_a_slug: Mapped[str] = mapped_column(
        String(80), ForeignKey("tool_catalog.slug", ondelete="CASCADE"), nullable=False
    )
    tool_b_slug: Mapped[str] = mapped_column(
        String(80), ForeignKey("tool_catalog.slug", ondelete="CASCADE"), nullable=False
    )
    score: Mapped[int] = mapped_column(Integer, nullable=False)
    dimensions: Mapped[dict[str, int]] = mapped_column(JSONB, nullable=False, default=dict)
    notes: Mapped[str | None] = mapped_column(Text)
    warnings: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False, default=list)
    last_reviewed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    __table_args__ = (
        UniqueConstraint("tool_a_slug", "tool_b_slug", name="uq_compatibility_matrix_pair"),
        CheckConstraint("tool_a_slug < tool_b_slug", name="pair_ordered"),
        CheckConstraint("score BETWEEN 0 AND 100", name="score_between_0_and_100"),
        Index("ix_compatibility_matrix_tool_b_slug", "tool_b_slug"),
    )


class PricingHistory(Base):
    """An observed change. Append-only.

    Written by the verification job when a published price differs from the
    stored one, and by an editor accepting that change. The job never mutates
    a price — it records what it saw. See M07 §Verification job.
    """

    __tablename__ = "pricing_history"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("ph"))
    entity_type: Mapped[PricedEntity] = mapped_column(
        Enum(PricedEntity, name="priced_entity", values_callable=lambda e: [m.value for m in e]),
        nullable=False,
    )
    entity_id: Mapped[str] = mapped_column(String(64), nullable=False)
    field: Mapped[str] = mapped_column(String(60), nullable=False)
    old_value: Mapped[Decimal | None] = mapped_column(Money)
    new_value: Mapped[Decimal | None] = mapped_column(Money)
    pct_change: Mapped[Decimal | None] = mapped_column(Numeric(10, 4))
    applied: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    source_id: Mapped[str | None] = mapped_column(
        String(64), ForeignKey("data_sources.id", ondelete="SET NULL")
    )
    detected_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    __table_args__ = (
        Index("ix_pricing_history_entity_id_detected_at", "entity_id", "detected_at"),
        Index("ix_pricing_history_detected_at", "detected_at"),
    )


class CatalogFlag(Base):
    """ "This number looks wrong."

    Costs almost nothing to build and is the cheapest staleness detector
    available: the people using a pricing calculator are the people who notice
    a stale price first.
    """

    __tablename__ = "catalog_flags"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("flag"))
    entity_type: Mapped[str] = mapped_column(String(20), nullable=False)
    entity_id: Mapped[str] = mapped_column(String(64), nullable=False)
    field: Mapped[str | None] = mapped_column(String(60))
    suggested_value: Mapped[str | None] = mapped_column(String(120))
    note: Mapped[str | None] = mapped_column(Text)
    source_url: Mapped[str | None] = mapped_column(Text)

    reported_by_user_id: Mapped[str | None] = mapped_column(
        String(64), ForeignKey("users.id", ondelete="SET NULL")
    )
    status: Mapped[FlagStatus] = mapped_column(
        Enum(FlagStatus, name="flag_status", values_callable=lambda e: [m.value for m in e]),
        nullable=False,
        default=FlagStatus.OPEN,
    )
    resolution_note: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        Index("ix_catalog_flags_status_created_at", "status", "created_at"),
        Index("ix_catalog_flags_entity_id", "entity_id"),
    )


__all__ = [
    "CatalogFlag",
    "Compatibility",
    "DataSource",
    "FlagStatus",
    "GpuPricing",
    "LifecycleStatus",
    "ModelFamily",
    "ModelPricing",
    "PricedEntity",
    "PricingHistory",
    "SourceKind",
    "Tool",
    "ToolCategory",
    "ToolStatus",
]
