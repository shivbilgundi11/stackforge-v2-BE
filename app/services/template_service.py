"""The template library: search, filtering, and the preview gate (M19).

Everything here is public. `PRD.md` §15 makes the library the product's best
organic acquisition surface, and an endpoint that requires a token cannot be
crawled, cannot be shared, and cannot be the thing that brings someone in. The
only authenticated behaviour in this module is *unlocking* — never reading.

**Premium templates are previewed, not hidden.** A hidden row loses the
indexable page, which is the acquisition channel that justifies half the
module. So the gate sits on `content_markdown` and `files`, and everything a
search engine or a browsing human needs — title, summary, category, tags, and
the first part of the body — is served to everyone.

The preview is cut on a **paragraph boundary**, not a character count. A gate
that stops mid-sentence reads as a bug in the page rather than as a deliberate
boundary, and the first impression of a paywall should not be that the site is
broken.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

from sqlalchemy import Select, func, literal_column, or_, select, update
from sqlalchemy.dialects.postgresql import TSVECTOR
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.elements import ColumnElement

from app.api.deps import Identity
from app.core.errors import NotFound
from app.core.logging import get_logger
from app.data.plans import Feature
from app.models.template import Difficulty, Template, TemplateCategory

logger = get_logger("templates")

#: Roughly a screen of prose. Enough to judge whether the rest is worth paying
#: for, which is the only job a preview has.
PREVIEW_CHARS: Final = 900

#: Display order for the hub. Deliberately not alphabetical: stack templates
#: are the category that leads into the product, so they go first.
CATEGORY_ORDER: Final[tuple[str, ...]] = (
    "stack",
    "blueprint",
    "code-starter",
    "prompt",
    "config",
    "checklist",
    "business",
)

CATEGORY_LABELS: Final[dict[str, str]] = {
    "stack": "Stack templates",
    "blueprint": "Architecture blueprints",
    "code-starter": "Code starters",
    "prompt": "Prompt templates",
    "config": "Config templates",
    "checklist": "Checklists",
    "business": "Business templates",
}

CATEGORY_BLURBS: Final[dict[str, str]] = {
    "stack": (
        "Pre-configured Stack Architect inputs. Open one and it scores against today's catalog."
    ),
    "blueprint": (
        "Reference architectures, the decision at each stage, and which of them are one-way doors."
    ),
    "code-starter": "Multi-file starters that run, not snippets to assemble.",
    "prompt": (
        "Prompts that change what an assistant produces, with the reasoning behind each rule."
    ),
    "config": (
        "The configuration files, with the mistakes that account for most failures called out."
    ),
    "checklist": ("What separates working from shippable, as things you either have or do not."),
    "business": (
        "Proposals, ROI reviews, and build-versus-buy, structured to survive a finance review."
    ),
}


def can_unlock(identity: Identity) -> bool:
    """Premium content needs Pro. Read access never does.

    The plan comparison moved to `FeatureService` in M20 — this module used to
    carry its own rank table, which is one of the five copies that made "what
    does Free get" a question with five answers.
    """
    from app.services import feature_service

    return feature_service.can(identity, Feature.PREMIUM_TEMPLATES).allowed


@dataclass(frozen=True)
class Filters:
    query: str | None = None
    category: str | None = None
    use_case: str | None = None
    difficulty: str | None = None
    #: `None` means both. The filter exists so someone on Free can find what
    #: they can use today, not so the library can hide what they cannot.
    premium: bool | None = None
    tag: str | None = None


def _visible(identity: Identity) -> Select[tuple[Template]]:
    """Templates this caller may see listed.

    Organization-scoped rows are excluded for everyone right now: M19 lands the
    column and the scoping, and M21 lands the membership table that could say
    whether a caller belongs to an organization. Excluding them is the safe
    direction — a team template leaking to the public library is a much worse
    failure than one not appearing until M21.
    """
    _ = identity
    return select(Template).where(Template.organization_id.is_(None))


def _apply(statement: Select[tuple[Template]], filters: Filters) -> Select[tuple[Template]]:
    if filters.category:
        statement = statement.where(Template.category == TemplateCategory(filters.category))
    if filters.difficulty:
        statement = statement.where(Template.difficulty == Difficulty(filters.difficulty))
    if filters.premium is not None:
        statement = statement.where(Template.is_premium.is_(filters.premium))
    # `.contains([value])` rather than `.any(value)`: both compile to an array
    # containment check, but only this one keeps the GIN index in play, and the
    # index is the reason the column is an array rather than a join table.
    if filters.use_case:
        statement = statement.where(Template.use_cases.contains([filters.use_case]))
    if filters.tag:
        statement = statement.where(Template.tags.contains([filters.tag]))
    return statement


def _weighted(column: Any, weight: str) -> ColumnElement[Any]:
    """One weighted field of the search document.

    The weight goes in as a `literal_column`, not a bound parameter.
    `setweight`'s second argument is Postgres's internal `"char"` type, and a
    bound string arrives as `varchar` — for which no overload exists, so the
    query fails at prepare time with `function setweight(tsvector, character
    varying) does not exist`. An inline literal is typed `unknown` and coerces.
    """
    return func.setweight(
        func.to_tsvector("english", func.coalesce(column, "")),
        literal_column(f"'{weight}'"),
    )


def search_vector() -> ColumnElement[Any]:
    """The weighted document, built exactly as the index builds it.

    Composed from the column expressions rather than written as raw SQL, so
    mypy can see it and Postgres renders it identically to the index
    definition in the migration. That identity is the whole point: a query
    whose vector differs from the index's by one field silently stops using
    the index and falls back to a sequential scan, which nothing fails on —
    it just gets slower as the library grows.

    Weighted A/B/C so a template whose *title* is "RAG Chatbot" outranks one
    that mentions RAG in paragraph eleven. Unweighted, the longest document
    wins every query.
    """
    return (
        _weighted(Template.title, "A")
        .op("||")(_weighted(Template.summary, "B"))
        .op("||", return_type=TSVECTOR)(_weighted(Template.content_markdown, "C"))
    )


async def search(
    db: AsyncSession,
    identity: Identity,
    *,
    filters: Filters,
    limit: int = 60,
) -> list[Template]:
    statement = _apply(_visible(identity), filters)

    if filters.query and filters.query.strip():
        # `websearch_to_tsquery` rather than `plainto_tsquery`: it understands
        # quoted phrases and `or`, which is what someone types into a search
        # box, and it does not raise on punctuation the way `to_tsquery` does.
        query = func.websearch_to_tsquery("english", filters.query.strip())
        vector = search_vector()
        statement = statement.where(vector.op("@@")(query)).order_by(
            func.ts_rank(vector, query).desc(), Template.title
        )
    else:
        # No query: newest first within the category ordering the hub uses, so
        # browsing and searching do not present the same rows in two orders.
        statement = statement.order_by(Template.published_at.desc().nulls_last(), Template.title)

    return list((await db.execute(statement.limit(limit))).scalars().all())


async def get(db: AsyncSession, slug: str, identity: Identity) -> Template:
    row = await db.scalar(_visible(identity).where(Template.slug == slug))
    if row is None:
        raise NotFound("No template with that slug.")
    return row


async def counts_by_category(db: AsyncSession, identity: Identity) -> dict[str, int]:
    rows = (
        await db.execute(
            _visible(identity)
            .with_only_columns(Template.category, func.count())
            .group_by(Template.category)
        )
    ).all()
    counted = {str(category.value): int(total) for category, total in rows}
    return {key: counted.get(key, 0) for key in CATEGORY_ORDER}


async def facets(db: AsyncSession, identity: Identity) -> dict[str, list[str]]:
    """The filter values that actually exist.

    Read from the data rather than hardcoded, so a filter never offers a use
    case that returns nothing — an empty result from a control the product
    itself offered reads as a broken search.
    """
    rows = (
        await db.execute(_visible(identity).with_only_columns(Template.use_cases, Template.tags))
    ).all()

    use_cases: set[str] = set()
    tags: set[str] = set()
    for row_use_cases, row_tags in rows:
        use_cases.update(row_use_cases or [])
        tags.update(row_tags or [])

    return {
        "categories": list(CATEGORY_ORDER),
        "use_cases": sorted(use_cases),
        "difficulties": [member.value for member in Difficulty],
        "tags": sorted(tags),
    }


# ── the gate ─────────────────────────────────────────────────────────────────


def preview_of(content: str, *, limit: int = PREVIEW_CHARS) -> str:
    """Cut on a paragraph boundary at or before `limit`.

    Never mid-sentence. A preview that stops halfway through a word reads as a
    rendering bug, and the first thing a paywall communicates should not be
    that the page is broken.
    """
    if len(content) <= limit:
        return content

    window = content[:limit]
    cut = window.rfind("\n\n")
    if cut < limit // 3:
        # No paragraph break in a sensible place — fall back to a sentence, and
        # then to the raw window rather than returning almost nothing.
        cut = max(window.rfind(". "), window.rfind("\n"))
        if cut < limit // 3:
            cut = limit
        else:
            cut += 1
    return content[:cut].rstrip()


@dataclass(frozen=True)
class Rendered:
    """A template as the caller may see it."""

    content_markdown: str
    files: list[dict[str, Any]]
    locked: bool
    #: True when a body was cut. Distinct from `locked`, because the frontend
    #: shows an upgrade card for one and nothing for the other.
    truncated: bool


def render(template: Template, identity: Identity) -> Rendered:
    if not template.is_premium or can_unlock(identity):
        return Rendered(
            content_markdown=template.content_markdown,
            files=list(template.files),
            locked=False,
            truncated=False,
        )

    preview = preview_of(template.content_markdown)
    return Rendered(
        content_markdown=preview,
        # Files are withheld entirely rather than previewed. Half a source file
        # is not a preview of anything — it is a file that does not compile,
        # and someone will paste it in before noticing.
        files=[],
        locked=True,
        truncated=len(preview) < len(template.content_markdown),
    )


# ── counters ─────────────────────────────────────────────────────────────────


async def record_view(db: AsyncSession, template: Template) -> None:
    await _increment(db, template, "view_count")


async def record_copy(db: AsyncSession, template: Template) -> None:
    """Counted on copy or download, never on view.

    They are separate signals and conflating them destroys the useful one: a
    template people open and leave is telling you something different from one
    they take, and only the second is a reason to write more like it.
    """
    await _increment(db, template, "copy_count")
    logger.info("templates.copied", slug=template.slug)


async def _increment(db: AsyncSession, template: Template, column: str) -> None:
    """An atomic UPDATE, not a read-modify-write.

    `template.view_count += 1` on a loaded row loses counts under any
    concurrency at all, which on the most-visited page in the library is every
    view that matters.
    """
    await db.execute(
        update(Template)
        .where(Template.id == template.id)
        .values(**{column: getattr(Template, column) + 1})
    )


async def related(db: AsyncSession, template: Template, *, limit: int = 4) -> list[Template]:
    """Other templates worth reading next.

    Matched on shared tags or use cases, in the same category first. Falling
    back to "anything recent" rather than returning nothing: an empty related
    block is a dead end on the page most likely to be someone's entry point.
    """
    overlap = or_(
        Template.tags.overlap(template.tags or []),
        Template.use_cases.overlap(template.use_cases or []),
    )
    statement = (
        _visible(Identity(user=None, anonymous_id=None, session_id=None))
        .where(Template.id != template.id)
        .where(overlap)
        .order_by((Template.category == template.category).desc(), Template.published_at.desc())
        .limit(limit)
    )
    rows = list((await db.execute(statement)).scalars().all())

    if len(rows) < limit:
        seen = {row.id for row in rows} | {template.id}
        filler = (
            (
                await db.execute(
                    _visible(Identity(user=None, anonymous_id=None, session_id=None))
                    .where(Template.id.notin_(seen))
                    .order_by(Template.published_at.desc())
                    .limit(limit - len(rows))
                )
            )
            .scalars()
            .all()
        )
        rows.extend(filler)

    return rows
