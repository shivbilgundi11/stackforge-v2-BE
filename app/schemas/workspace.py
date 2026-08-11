"""Workspace wire shapes (M17)."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

ItemType = Literal["run", "stack", "artifact", "template"]


class ProjectIn(BaseModel):
    name: str = Field(min_length=1, max_length=160)
    description: str | None = Field(default=None, max_length=2000)
    use_case: str | None = Field(default=None, max_length=60)


class ProjectPatch(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=160)
    description: str | None = Field(default=None, max_length=2000)
    use_case: str | None = Field(default=None, max_length=60)
    archived: bool | None = None


class ProjectOut(BaseModel):
    id: str
    name: str
    description: str | None
    use_case: str | None
    session_state: dict[str, Any]
    item_count: int = 0
    archived_at: datetime | None
    created_at: datetime
    updated_at: datetime


class ProjectItemIn(BaseModel):
    item_type: ItemType
    item_id: str = Field(min_length=1, max_length=64)
    note: str | None = Field(default=None, max_length=1000)
    pinned: bool = False


class ProjectItemOut(BaseModel):
    id: str
    item_type: str
    item_id: str
    position: int
    pinned: bool
    note: str | None
    #: Resolved from whichever table `item_type` names. `None` when the target
    #: has been deleted — the polymorphic join cannot be a foreign key, so a
    #: dangling row is possible and is reported rather than hidden.
    title: str | None = None
    subtitle: str | None = None
    href: str | None = None
    created_at: datetime


class ReorderIn(BaseModel):
    item_ids: list[str] = Field(min_length=1, max_length=200)


class PinIn(BaseModel):
    pinned: bool


class CarriedSessionPatch(BaseModel):
    """Values a tool is handing forward.

    Free-form because the carried set grows with the tools — pinning a schema
    here would mean a migration every time a workflow learns to hand something
    on.
    """

    values: dict[str, Any] = Field(default_factory=dict)


class CarriedSessionOut(BaseModel):
    """Named `Carried…` rather than `SessionOut`.

    Auth already owns `SessionOut` for a signed-in device session, and two
    schemas with one name collide in the generated TypeScript — both get
    mangled to fully-qualified aliases, and the existing auth client stops
    compiling. Two different things should not share a name anyway.
    """

    session_state: dict[str, Any]


# ── dashboard ────────────────────────────────────────────────────────────────


class HeadlineOut(BaseModel):
    key: str
    value: str


class RecentRunOut(BaseModel):
    run_id: str
    tool_slug: str
    workflow: str
    saved: bool
    source: str
    headline: HeadlineOut | None
    created_at: datetime


class SavedStackOut(BaseModel):
    id: str
    name: str
    components: int
    version: int
    deprecated: list[str]
    updated_at: datetime


class ProjectCardOut(BaseModel):
    id: str
    name: str
    use_case: str | None
    items: int
    updated_at: datetime


class StaleAlertOut(BaseModel):
    stack_id: str
    stack_name: str
    tool: str
    status: str
    reason: str
    alternatives: list[str]


class UsageOut(BaseModel):
    total: int
    saved: int
    today: int
    projects: int
    #: `null` is unlimited (M20). Not a large sentinel — a meter reading
    #: "3 of 999999" is a meter nobody believes.
    project_limit: int | None


class PlanOut(BaseModel):
    plan: str
    source: str


class DashboardOut(BaseModel):
    recent_runs: list[RecentRunOut]
    saved_stacks: list[SavedStackOut]
    projects: list[ProjectCardOut]
    stale_alerts: list[StaleAlertOut]
    quick_start: list[str]
    usage: UsageOut
    plan: PlanOut


class SearchHitOut(BaseModel):
    kind: Literal["project", "stack", "run"]
    id: str
    title: str
    subtitle: str
    href: str
    updated_at: datetime
