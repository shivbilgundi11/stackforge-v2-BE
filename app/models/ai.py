"""Every model call, logged.

One row per attempt — including the ones that failed. A table that only
records successes cannot answer the question the AI layer will actually be
asked, which is "how often does this not work, and what does it cost when it
does".

`prompt_version` is on the row rather than derivable, because a quality change
after a prompt edit is unattributable without it. That is the whole reason the
prompt registry versions itself.
"""

from __future__ import annotations

import enum
from datetime import datetime
from decimal import Decimal

from sqlalchemy import (
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base, new_id


class AiOutcome(str, enum.Enum):
    """Why a call ended.

    Every failure mode is named. `AiService` returns `None` for all of them,
    so this column is the only place the difference survives — and "the prompt
    is wrong" and "the model was unavailable" need very different responses.
    """

    SUCCESS = "success"
    DISABLED = "disabled"
    QUOTA_EXCEEDED = "quota_exceeded"
    TIMEOUT = "timeout"
    RATE_LIMITED = "rate_limited"
    API_ERROR = "api_error"
    REFUSAL = "refusal"
    INVALID_OUTPUT = "invalid_output"


class AiCall(Base):
    __tablename__ = "ai_calls"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("ai"))
    purpose: Mapped[str] = mapped_column(String(60), nullable=False)
    model: Mapped[str] = mapped_column(String(80), nullable=False)
    prompt_version: Mapped[str] = mapped_column(String(20), nullable=False)

    user_id: Mapped[str | None] = mapped_column(
        String(64), ForeignKey("users.id", ondelete="SET NULL")
    )
    tool_slug: Mapped[str | None] = mapped_column(String(80))

    input_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    # Recorded separately from `input_tokens`, which is the uncached remainder.
    # Total prompt size is the sum of the three — a caching change that is
    # working shows up here and nowhere else.
    cached_read_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    cached_write_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # Priced from `ai_pricing`, not from `model_pricing`. The catalog is
    # user-facing content that editorial staff edit; billing internal
    # accounting off it would tie the books to a content table.
    cost_usd: Mapped[Decimal] = mapped_column(Numeric(14, 6), nullable=False, default=Decimal(0))
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    outcome: Mapped[AiOutcome] = mapped_column(
        Enum(AiOutcome, name="ai_outcome", values_callable=lambda e: [m.value for m in e]),
        nullable=False,
    )
    error_detail: Mapped[str | None] = mapped_column(String(500))

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    __table_args__ = (
        Index("ix_ai_calls_created_at", "created_at"),
        Index("ix_ai_calls_purpose_outcome", "purpose", "outcome"),
        Index("ix_ai_calls_user_id", "user_id"),
    )
