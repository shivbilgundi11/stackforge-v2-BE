"""Cursor pagination.

Offset pagination on a list the user is actively adding to skips and duplicates
rows. Cursors cost nothing extra against a UUIDv7 primary key, which is already
time-ordered.

One implementation, used by every list endpoint.
"""

from __future__ import annotations

import base64
import binascii
import json
from datetime import datetime
from typing import Any, NamedTuple

from sqlalchemy import Select, and_, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.errors import ValidationFailed
from app.core.responses import PageMeta

MAX_LIMIT = 100
DEFAULT_LIMIT = 25


class Cursor(NamedTuple):
    created_at: datetime
    id: str

    def encode(self) -> str:
        raw = json.dumps({"t": self.created_at.isoformat(), "i": self.id})
        return base64.urlsafe_b64encode(raw.encode()).decode().rstrip("=")

    @staticmethod
    def decode(value: str) -> Cursor:
        try:
            padded = value + "=" * (-len(value) % 4)
            payload = json.loads(base64.urlsafe_b64decode(padded).decode())
            return Cursor(datetime.fromisoformat(payload["t"]), payload["i"])
        except (ValueError, KeyError, TypeError, binascii.Error) as exc:
            raise ValidationFailed(
                "The pagination cursor is not valid.",
                details={"fields": [{"path": "cursor", "message": "Malformed cursor."}]},
            ) from exc


class Page[T](NamedTuple):
    items: list[T]
    meta: PageMeta


async def paginate(
    session: AsyncSession,
    statement: Select[Any],
    model: Any,
    *,
    cursor: str | None = None,
    limit: int = DEFAULT_LIMIT,
    with_total: bool = False,
) -> Page[Any]:
    """Newest first, keyed on `(created_at, id)`.

    The compound key matters: `created_at` alone is not unique, and two rows
    sharing a timestamp would make a page boundary drop or repeat one of them.
    """
    limit = max(1, min(limit, MAX_LIMIT))

    if cursor:
        point = Cursor.decode(cursor)
        statement = statement.where(
            or_(
                model.created_at < point.created_at,
                and_(model.created_at == point.created_at, model.id < point.id),
            )
        )

    total: int | None = None
    if with_total:
        count_stmt = select(func.count()).select_from(statement.order_by(None).subquery())
        total = (await session.execute(count_stmt)).scalar_one()

    # One extra row tells us whether a next page exists without a second query.
    statement = statement.order_by(model.created_at.desc(), model.id.desc()).limit(limit + 1)
    rows = list((await session.execute(statement)).scalars().all())

    has_more = len(rows) > limit
    items = rows[:limit]

    next_cursor = (
        Cursor(items[-1].created_at, items[-1].id).encode() if has_more and items else None
    )

    return Page(
        items=items,
        meta=PageMeta(cursor=cursor, next_cursor=next_cursor, limit=limit, total=total),
    )
