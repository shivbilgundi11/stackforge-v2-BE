"""Ownership scoping, enforced by construction.

**This is the security decision of M17.** Every read of user-scoped data goes
through a query that cannot express "any user's row" — the owner predicate is
applied when the query is built, not remembered by whoever writes the endpoint.

The alternative is an `if row.user_id != user.id: raise NotFound` in every
handler. That works until the fifteenth handler, and the fifteenth handler is
the one that ships. A helper that has no way to return someone else's row is a
much stronger guarantee than a check that has to be repeated correctly.

Two rules the callers inherit:

**Not-yours and does-not-exist are the same answer.** Both are `NotFound`.
Distinguishing them turns any id-taking endpoint into an oracle that confirms
which ids are real.

**Anonymous identity is scoped too.** A run made before signup belongs to an
anonymous session, and that session is as much an owner as a user id is.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import Select, false, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import DeclarativeBase

from app.api.deps import Identity
from app.core.errors import NotFound
from app.models.user import User


def owned_by[Model: DeclarativeBase](model: type[Model], user: User) -> Select[tuple[Model]]:
    """A select over `model` that can only ever return `user`'s rows.

    Every user-scoped listing starts here. Add filters and ordering onto the
    returned statement; there is no supported way to remove the owner
    predicate, which is the point.
    """
    return select(model).where(model.user_id == user.id)  # type: ignore[attr-defined]


async def get_owned[Model: DeclarativeBase](
    db: AsyncSession,
    model: type[Model],
    row_id: str,
    user: User,
    *,
    label: str = "record",
) -> Model:
    """Fetch one row, or raise `NotFound` — including when it exists but is
    someone else's."""
    row = await db.scalar(
        owned_by(model, user).where(model.id == row_id)  # type: ignore[attr-defined]
    )
    if row is None:
        raise NotFound(f"No {label} with that id.")
    return row


def visible_to[Model: DeclarativeBase](
    model: type[Model], identity: Identity
) -> Select[tuple[Model]]:
    """A select scoped to whoever is asking — user *or* anonymous session.

    Runs are the case this exists for: they are created before an account
    exists and claimed on signup, so "the owner" is one of two columns for the
    life of the row. An identity with neither matches nothing, which is the
    correct answer rather than an error.
    """
    statement = select(model)
    if identity.user is not None:
        return statement.where(model.user_id == identity.user.id)  # type: ignore[attr-defined]
    if identity.anonymous_id is not None:
        return statement.where(
            model.anonymous_session_id == identity.anonymous_id,  # type: ignore[attr-defined]
        )
    # No identity at all. An explicit `false()` rather than an unfiltered
    # query: the failure mode of getting this branch wrong is returning
    # everyone's rows, so it must be impossible to fall through to one.
    return statement.where(false())


async def get_visible[Model: DeclarativeBase](
    db: AsyncSession,
    model: type[Model],
    row_id: str,
    identity: Identity,
    *,
    label: str = "record",
) -> Model:
    row = await db.scalar(
        visible_to(model, identity).where(model.id == row_id)  # type: ignore[attr-defined]
    )
    if row is None:
        raise NotFound(f"No {label} with that id.")
    return row


def owner_columns(identity: Identity) -> dict[str, Any]:
    """The owner fields to write on a new row.

    Exactly one is set, which is what the `exactly_one_owner` check constraint
    on `tool_runs` enforces. Building them here means no caller can set both
    and produce a row that vanishes from every per-owner query.
    """
    if identity.user is not None:
        return {"user_id": identity.user.id, "anonymous_session_id": None}
    return {"user_id": None, "anonymous_session_id": identity.anonymous_id}
