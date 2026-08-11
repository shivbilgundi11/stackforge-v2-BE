"""The template library (M19).

One table. Templates are authored as Markdown files with frontmatter in
`app/data/templates/` and seeded from there, so a template is version
controlled, reviewable in a pull request, and editable without touching code.
An admin UI for thirty rows maintained by the people who own the repository
would be premature, and it would put the content outside review.

Two columns carry the shapes that make this more than a blog:

`content_markdown` is the body of a document template. `files` is a real
directory tree for a multi-file code starter — `[{path, language, content}]` —
because a starter flattened into one Markdown file is a starter someone has to
take apart before they can run it.

`stack_input` is what makes a *stack* template part of the product rather than
adjacent to it. It holds a `RecommendIn` payload, so opening one loads its
constraints into the Stack Architect form and produces a real recommendation
against today's catalog. A stack template that were merely a description of a
stack would go stale the moment the catalog moved; this one cannot, because it
stores the question rather than the answer.
"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    DateTime,
    Enum,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base, TimestampMixin, new_id


class TemplateCategory(str, enum.Enum):
    STACK = "stack"
    BLUEPRINT = "blueprint"
    CODE_STARTER = "code-starter"
    PROMPT = "prompt"
    CONFIG = "config"
    CHECKLIST = "checklist"
    BUSINESS = "business"


class Difficulty(str, enum.Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


class Template(Base, TimestampMixin):
    __tablename__ = "templates"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("tpl"))
    #: The URL. Stable across re-seeds and the key the seeder matches on, so
    #: renaming a title does not orphan every link to it.
    slug: Mapped[str] = mapped_column(String(120), nullable=False, unique=True)

    title: Mapped[str] = mapped_column(String(200), nullable=False)
    category: Mapped[TemplateCategory] = mapped_column(
        Enum(
            TemplateCategory,
            name="template_category",
            values_callable=lambda e: [m.value for m in e],
        ),
        nullable=False,
    )
    difficulty: Mapped[Difficulty] = mapped_column(
        Enum(
            Difficulty, name="template_difficulty", values_callable=lambda e: [m.value for m in e]
        ),
        nullable=False,
        default=Difficulty.INTERMEDIATE,
    )

    summary: Mapped[str] = mapped_column(Text, nullable=False)
    content_markdown: Mapped[str] = mapped_column(Text, nullable=False, default="")
    #: `[{path, language, content}]`. Empty for a single-document template.
    files: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, nullable=False, default=list)
    #: A `RecommendIn` payload for `category = stack`, else empty.
    stack_input: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)

    use_cases: Mapped[list[str]] = mapped_column(ARRAY(String(40)), nullable=False, default=list)
    tags: Mapped[list[str]] = mapped_column(ARRAY(String(40)), nullable=False, default=list)
    #: Related tool slugs from the registry. Rendered as internal links, which
    #: is half of why the library is an SEO surface at all — a page with no
    #: outbound path into the product converts nobody.
    related_tools: Mapped[list[str]] = mapped_column(
        ARRAY(String(80)), nullable=False, default=list
    )

    #: Premium templates are *previewed*, never hidden. Hiding the row loses
    #: the indexable page, which is the acquisition channel that justifies half
    #: this module; the gate goes on the body.
    is_premium: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    #: Team-private templates. The column and the query scoping land in M19;
    #: the permission checks are M21's.
    organization_id: Mapped[str | None] = mapped_column(String(64))

    #: Which templates to write more of. The only reliable input to a content
    #: roadmap, and the reason `copy_count` is separate from `view_count`: a
    #: page people open and do not use is a different signal from one they take.
    view_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    copy_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # The full-text index over `title || summary || content_markdown` is
    # created in the migration rather than declared here, for the same reason
    # as the projects one: `to_tsvector('english', …)` needs a regconfig
    # literal that SQLAlchemy cannot render into an index definition, and
    # autogenerate raises on it.
    __table_args__ = (
        Index("ix_templates_category_published_at", "category", "published_at"),
        Index("ix_templates_tags", "tags", postgresql_using="gin"),
        Index("ix_templates_use_cases", "use_cases", postgresql_using="gin"),
        Index("ix_templates_organization_id", "organization_id"),
    )

    @property
    def is_multi_file(self) -> bool:
        return bool(self.files)


__all__ = ["Difficulty", "Template", "TemplateCategory"]
