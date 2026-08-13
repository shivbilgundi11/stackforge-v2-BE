"""Stack Architect request and response shapes (M15)."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

UseCase = Literal["rag", "chat", "agents", "automation", "coding", "search", "analytics"]
ScaleTarget = Literal["small", "medium", "large", "xlarge"]
TeamSkill = Literal["beginner", "intermediate", "advanced"]
Sensitivity = Literal["public", "internal", "confidential", "restricted", "regulated"]
Deployment = Literal["any", "managed", "self-hosted", "hybrid"]


class RecommendIn(BaseModel):
    """The eight inputs from `PRD.md` §8.1.

    All eight have defaults so the form is runnable from the first screen —
    an anonymous visitor gets a real recommendation without filling anything
    in, which is the product's strongest demo.
    """

    use_case: UseCase = "rag"
    scale_target: ScaleTarget = "medium"
    monthly_budget: int = Field(default=2_000, ge=0, le=10_000_000)
    team_skill: TeamSkill = "intermediate"
    latency_ms: int = Field(default=2_000, ge=50, le=600_000)
    sensitivity: Sensitivity = "internal"
    deployment: Deployment = "any"
    capabilities: list[str] = Field(default_factory=list, max_length=12)


class ScoreIn(BaseModel):
    """Score a stack the user assembled themselves."""

    component_slugs: list[str] = Field(min_length=1, max_length=15)
    monthly_budget: int = Field(default=2_000, ge=0, le=10_000_000)
    scale_target: ScaleTarget = "medium"
    sensitivity: Sensitivity = "internal"


# ── stacks ───────────────────────────────────────────────────────────────────


class StackIn(BaseModel):
    name: str = Field(min_length=1, max_length=160)
    description: str | None = Field(default=None, max_length=2000)
    component_slugs: list[str] = Field(min_length=1, max_length=15)
    requirements: dict[str, Any] = Field(default_factory=dict)
    source_run_id: str | None = Field(default=None, max_length=64)
    #: `team` requires acting inside an organization (the `X-Organization-Id`
    #: header). Absent means private — personal work never becomes team work
    #: by default (M21).
    visibility: Literal["private", "team", "public"] | None = None


class StackPatch(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=160)
    description: str | None = Field(default=None, max_length=2000)
    component_slugs: list[str] | None = Field(default=None, min_length=1, max_length=15)
    requirements: dict[str, Any] | None = None
    visibility: Literal["private", "team", "public"] | None = None
    #: What changed and why. Written to the version row, because a version
    #: history of "updated" entries is a list nobody reads.
    change_summary: str | None = Field(default=None, max_length=500)


class StackOut(BaseModel):
    id: str
    name: str
    description: str | None
    component_slugs: list[str]
    requirements: dict[str, Any]
    current_version: int
    #: Computed at read time from the current catalog, never stored — which is
    #: what makes the deprecation flag below meaningful.
    score: str
    deprecated_components: list[str]
    visibility: str
    organization_id: str | None
    #: Whether the caller authored it — a team listing mixes their rows with
    #: teammates', and the UI gates edit affordances on this.
    is_yours: bool
    owner_name: str | None
    created_at: datetime
    updated_at: datetime


class StackVersionOut(BaseModel):
    version: int
    name: str
    component_slugs: list[str]
    requirements: dict[str, Any]
    change_summary: str | None
    created_at: datetime


class StackDiffOut(BaseModel):
    """What changed between two versions, and what it cost or bought.

    Score delta is computed by re-scoring both component sets against
    *today's* catalog. Comparing a stored v1 score against a fresh v2 score
    would attribute catalog drift to the user's edit.
    """

    from_version: int
    to_version: int
    added: list[str]
    removed: list[str]
    unchanged: list[str]
    score_from: str
    score_to: str
    score_delta: str
    change_summary: str | None
