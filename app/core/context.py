"""Request-scoped context.

A ContextVar rather than a parameter threaded through every signature. The
logger reads it, so every line emitted inside a request is correlated without
each call site having to remember.
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field


@dataclass
class RequestContext:
    request_id: str = ""
    user_id: str | None = None
    organization_id: str | None = None
    plan: str | None = None
    extra: dict[str, str] = field(default_factory=dict)


# Default is None, not a RequestContext instance. A shared mutable default
# would be the *same object* in every context that never called set_context —
# so `bind()` outside a request would write into global state that the next
# request then reads.
_ctx: ContextVar[RequestContext | None] = ContextVar("request_context", default=None)


def get_context() -> RequestContext:
    ctx = _ctx.get()
    if ctx is None:
        ctx = RequestContext()
        _ctx.set(ctx)
    return ctx


def set_context(ctx: RequestContext) -> None:
    _ctx.set(ctx)


def bind(**values: str | None) -> None:
    """Merge values into the current context.

    Used by the auth dependency once identity is resolved, so the closing log
    line carries the user even though the opening one could not.
    """
    ctx = get_context()
    for key, value in values.items():
        if hasattr(ctx, key):
            setattr(ctx, key, value)
        elif value is not None:
            ctx.extra[key] = value


def as_dict() -> dict[str, str]:
    ctx = get_context()
    out: dict[str, str] = {"request_id": ctx.request_id}
    if ctx.user_id:
        out["user_id"] = ctx.user_id
    if ctx.organization_id:
        out["organization_id"] = ctx.organization_id
    if ctx.plan:
        out["plan"] = ctx.plan
    out.update(ctx.extra)
    return out
