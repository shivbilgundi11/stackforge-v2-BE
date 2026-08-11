"""Template wire shapes (M19).

Two output models rather than one, and the split is the security boundary.
`TemplateSummaryOut` carries no body at all, so a listing endpoint cannot leak
premium content by forgetting to gate — there is nowhere for it to go.
`TemplateDetailOut` carries a body that has already been through
`template_service.render`.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

TemplateCategoryName = Literal[
    "stack", "blueprint", "code-starter", "prompt", "config", "checklist", "business"
]
DifficultyName = Literal["beginner", "intermediate", "advanced"]


class TemplateFileOut(BaseModel):
    path: str
    language: str
    content: str


class TemplateSummaryOut(BaseModel):
    """A card in the grid. No body, by construction."""

    slug: str
    title: str
    category: TemplateCategoryName
    difficulty: DifficultyName
    summary: str
    use_cases: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    is_premium: bool
    #: Multi-file starters are worth flagging on the card — it is the
    #: difference between a snippet and something you can run.
    file_count: int = 0
    #: Present for stack templates. The card's "Use this stack" needs to know
    #: before the detail page is fetched.
    is_stack_template: bool = False
    view_count: int = 0
    copy_count: int = 0
    published_at: datetime | None = None


class TemplateDetailOut(TemplateSummaryOut):
    """The page. `content_markdown` is whatever the caller is entitled to."""

    content_markdown: str
    files: list[TemplateFileOut] = Field(default_factory=list)
    #: A premium body the caller cannot unlock. The upgrade card renders on
    #: this, not on `is_premium` — an entitled Pro user sees neither.
    locked: bool = False
    truncated: bool = False
    #: The Stack Architect payload, for `category = stack`. Typed loosely on
    #: purpose: it is validated against `RecommendIn` when it is *used*, and
    #: duplicating that schema here would be a second thing to keep in step.
    stack_input: dict[str, object] = Field(default_factory=dict)
    related_tools: list[str] = Field(default_factory=list)
    related: list[TemplateSummaryOut] = Field(default_factory=list)


class CategoryOut(BaseModel):
    key: TemplateCategoryName
    label: str
    description: str
    count: int


class LibraryOut(BaseModel):
    """The hub. One request — category counts, featured, and newest."""

    total: int
    categories: list[CategoryOut]
    featured: list[TemplateSummaryOut] = Field(default_factory=list)
    recent: list[TemplateSummaryOut] = Field(default_factory=list)


class FacetsOut(BaseModel):
    """Filter values that exist in the data.

    Served rather than hardcoded in the client, so a control never offers a
    value that returns nothing.
    """

    categories: list[str] = Field(default_factory=list)
    use_cases: list[str] = Field(default_factory=list)
    difficulties: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
