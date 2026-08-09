"""price_unit on model_pricing

Records what a stored price is a price *of* (D-18). Everything is per-token
except Cohere's rerank endpoints, which publish per 1K searches; both units
were landing in `input_cost_per_1k` with nothing to distinguish them.

Revision ID: 8877676ca92e
Revises: 15f2c3f39e61
Create Date: 2026-08-09 13:24:38.434085+00:00
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "8877676ca92e"
down_revision: str | None = "15f2c3f39e61"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Named explicitly so the downgrade can drop it. An enum created implicitly by
# `add_column` is not removed by `drop_column`, which leaves the type behind
# and makes the next upgrade fail with "type already exists".
price_unit = sa.Enum("tokens", "searches", name="price_unit")


def upgrade() -> None:
    bind = op.get_bind()
    price_unit.create(bind, checkfirst=True)
    op.add_column(
        "model_pricing",
        sa.Column("price_unit", price_unit, server_default="tokens", nullable=False),
    )


def downgrade() -> None:
    op.drop_column("model_pricing", "price_unit")
    price_unit.drop(op.get_bind(), checkfirst=True)
