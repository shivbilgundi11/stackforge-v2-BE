"""What an artifact is generated *from*.

A generator takes one of these and returns content. It never takes a session,
a request, or an identity — every database read happens here, once, so a
generator stays a pure function of its input. That is what makes "generate
twice, assert byte equality" a unit test rather than an integration one, and
FR-11's idempotency requirement is exactly that assertion.

Two source shapes, matching the two things a user can own:

  * `StackSource` — a saved stack, re-resolved against today's catalog. The
    score and the compatibility matrix are recomputed here for the same reason
    `stacks.py` recomputes them on read (D-27): an exported document that
    quotes a score frozen at save time is a document that disagrees with the
    screen it was exported from.
  * `RunSource` — a stored `tool_runs` row, rebuilt into the wire shape the
    result page renders. Artifacts the tool already emitted come back
    unchanged; nothing is regenerated, so an exported `docker-compose.yml` is
    byte-for-byte the one the user saw.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import Identity
from app.core.errors import NotFound
from app.models.export import SourceType
from app.models.stack import Stack
from app.models.tool_run import ToolRun
from app.schemas.catalog import CompatibilityOut, ToolOut
from app.schemas.tools import Artifact, ToolRunOut
from app.services import catalog_service, stack_architect_service, stack_score_service
from app.services.stack_architect_service import Requirements

#: Defaults for a stack saved without a full requirements bag. They match
#: `RecommendIn`'s defaults, so a stack created from the Architect and one
#: created by hand describe themselves the same way.
REQUIREMENT_DEFAULTS: dict[str, Any] = {
    "use_case": "rag",
    "scale_target": "medium",
    "monthly_budget": 2_000,
    "team_skill": "intermediate",
    "latency_ms": 2_000,
    "sensitivity": "internal",
    "deployment": "any",
    "capabilities": (),
}


def requirements_of(raw: dict[str, Any] | None) -> Requirements:
    values = {**REQUIREMENT_DEFAULTS, **(raw or {})}
    return Requirements(
        use_case=str(values["use_case"]),
        scale_target=str(values["scale_target"]),
        monthly_budget=int(values["monthly_budget"] or 0),
        team_skill=str(values["team_skill"]),
        latency_ms=int(values["latency_ms"] or 0),
        sensitivity=str(values["sensitivity"]),
        deployment=str(values["deployment"]),
        capabilities=tuple(values["capabilities"] or ()),
    )


@dataclass(frozen=True)
class Narrative:
    """The written half of an architecture document.

    A separate object rather than three optional strings on the source,
    because the three arrive together or not at all: they are one model call,
    and a document with an overview but no operations section would be a
    partial answer presented as a whole one.
    """

    overview: str
    decisions: str
    operations: str


@dataclass(frozen=True)
class StackSource:
    kind = SourceType.STACK

    id: str
    title: str
    description: str | None
    slug_basis: str
    components: list[ToolOut]
    #: Slugs the stack names that the catalog no longer carries. Reported
    #: rather than dropped: a plan whose vector store vanished from the catalog
    #: is a plan with a hole in it, and an export that quietly omits the row
    #: makes the hole invisible.
    missing_slugs: list[str]
    deprecated: list[ToolOut]
    requirements: Requirements
    score: stack_score_service.StackScore
    compatibility: CompatibilityOut | None
    version: int
    #: When the thing this describes last changed — *not* when the export ran.
    #:
    #: FR-11 requires that re-exporting produces byte-identical output, and a
    #: `generated_at: now()` in an export envelope breaks that on the second
    #: request. Stamping the source's own timestamp keeps the property and is
    #: the more useful figure anyway: a reader of a six-month-old JSON file
    #: needs to know how old the *stack* is, not how old the download is.
    updated_at: datetime
    #: Written commentary for the architecture document, when a model
    #: produced some. Carried on the source rather than fetched by the
    #: generator, so the generator stays a pure function and FR-11's
    #: "export twice, get the same bytes" stays a unit test: the same source
    #: — narrative included or not — renders the same document.
    narrative: Narrative | None = None


@dataclass(frozen=True)
class RunSource:
    kind = SourceType.RUN

    id: str
    title: str
    slug_basis: str
    tool_slug: str
    workflow: str
    input: dict[str, Any]
    output: ToolRunOut

    @property
    def updated_at(self) -> datetime:
        """A run is immutable, so its creation is also its last change."""
        return self.output.created_at

    @property
    def artifacts(self) -> list[Artifact]:
        return list(self.output.artifacts)


Source = StackSource | RunSource


async def resolve(
    db: AsyncSession,
    *,
    source_type: SourceType,
    source_id: str,
    identity: Identity,
) -> Source:
    """Load a source the caller owns, or raise `NotFound`.

    Ownership is checked here rather than in the router because every entry
    point into M18 — export, share mint, bundle — needs the same check, and a
    gate that has to be repeated at three call sites is a gate that will be
    missing from the fourth.
    """
    if source_type is SourceType.STACK:
        return await stack_source(db, source_id, identity)
    return await run_source(db, source_id, identity)


async def run_source(db: AsyncSession, run_id: str, identity: Identity) -> RunSource:
    from app.services import tool_service

    run = await tool_service.get_run(db, run_id, identity)
    if run is None:
        raise NotFound("No such run.")
    return run_source_of(run)


def run_source_of(run: ToolRun) -> RunSource:
    """Rebuild the wire shape from a stored row.

    The identifying fields come from the row rather than the blob, matching
    `runs.py`: a stored `run_id` that disagrees with the row it lives on is a
    bug the export must not carry forward.
    """
    blob = dict(run.output or {})
    for key in ("run_id", "tool_slug", "source", "duration_ms", "created_at"):
        blob.pop(key, None)

    output = ToolRunOut(
        run_id=run.id,
        tool_slug=run.tool_slug,
        source=run.source.value if hasattr(run.source, "value") else str(run.source),
        duration_ms=run.duration_ms,
        created_at=run.created_at,
        **blob,
    )
    return RunSource(
        id=run.id,
        title=run.tool_slug.replace("-", " ").title(),
        slug_basis=run.tool_slug,
        tool_slug=run.tool_slug,
        workflow=run.workflow,
        input=dict(run.input or {}),
        output=output,
    )


async def stack_source(db: AsyncSession, stack_id: str, identity: Identity) -> StackSource:
    stack = await db.get(Stack, stack_id)
    # Anonymous callers cannot own a stack — saving requires an account — so
    # a caller who does not own the row gets the same answer as a wrong id.
    if stack is None or identity.user is None or stack.user_id != identity.user.id:
        raise NotFound("No stack with that id.")
    return await stack_source_of(db, stack)


async def stack_source_of(db: AsyncSession, stack: Stack) -> StackSource:
    catalog = await catalog_service.list_tools(db)
    by_slug = {tool.slug: tool for tool in catalog}

    components = [by_slug[slug] for slug in stack.component_slugs if slug in by_slug]
    missing = [slug for slug in stack.component_slugs if slug not in by_slug]
    requirements = requirements_of(stack.requirements)

    compatibility = (
        await catalog_service.get_compatibility(db, [tool.slug for tool in components])
        if len(components) > 1
        else None
    )
    score = stack_score_service.score(
        components,
        monthly_budget=requirements.monthly_budget,
        scale_target=requirements.scale_target,
        sensitivity=requirements.sensitivity,
        compatibility=compatibility,
    )

    return StackSource(
        id=stack.id,
        title=stack.name,
        description=stack.description,
        slug_basis=stack.name,
        components=components,
        missing_slugs=missing,
        deprecated=[
            tool for tool in components if tool.status not in stack_architect_service.RECOMMENDABLE
        ],
        requirements=requirements,
        score=score,
        compatibility=compatibility,
        version=stack.current_version,
        updated_at=stack.updated_at,
    )


__all__ = [
    "RunSource",
    "Source",
    "StackSource",
    "requirements_of",
    "resolve",
    "run_source_of",
    "stack_source_of",
]
