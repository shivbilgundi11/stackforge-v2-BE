"""The dashboard, and the search behind ⌘K (M17).

M05 built the shell and its empty states; this fills them with real rows.

The panel worth the most is the stale-data alert. `PRD.md` §24 makes catalog
drift a retention mechanic, and a mechanic that only exists as an email is one
the user meets in their inbox and never again. Scoring stacks on read (D-27)
is what lets this be computed rather than stored: a tool buried this morning
shows up in this panel this afternoon, with no job in between.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Final

from sqlalchemy import func, literal, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.models.project import Project, ProjectItem
from app.models.stack import Stack
from app.models.tool_run import ToolRun
from app.models.user import User
from app.repositories import scoped
from app.services import catalog_service, project_service, run_service
from app.services.stack_architect_service import RECOMMENDABLE

logger = get_logger("dashboard")

RECENT_RUNS: Final = 10
RECENT_STACKS: Final = 5

#: Metric keys worth showing beside a run in the activity feed, in preference
#: order. A feed row that says only "you ran the cost planner" is a timestamp
#: with extra steps — the number is the reason to click.
HEADLINE_KEYS: Final[tuple[str, ...]] = (
    "monthly_cost",
    "cost_per_month",
    "total_cost",
    "score",
    "total_vram_gb",
    "monthly_total",
    "total_gb",
    "tokens",
    "binding_constraint",
    "topology",
    "winner_name",
)


def headline_of(run: ToolRun) -> dict[str, str] | None:
    """The one figure worth putting next to a run in the feed."""
    metrics = (run.output or {}).get("metrics") or {}
    for key in HEADLINE_KEYS:
        if key in metrics and metrics[key] not in (None, ""):
            return {"key": key, "value": str(metrics[key])}
    # Nothing recognised: show the first metric rather than nothing, because a
    # tool added later should not silently produce a blank row.
    for key, value in metrics.items():
        if value not in (None, ""):
            return {"key": key, "value": str(value)}
    return None


async def recent_runs(db: AsyncSession, user: User) -> list[dict[str, Any]]:
    rows = (
        (
            await db.execute(
                scoped.owned_by(ToolRun, user)
                .order_by(ToolRun.created_at.desc())
                .limit(RECENT_RUNS)
            )
        )
        .scalars()
        .all()
    )
    return [
        {
            "run_id": run.id,
            "tool_slug": run.tool_slug,
            "workflow": run.workflow,
            "saved": run.saved,
            "source": run.source.value,
            "headline": headline_of(run),
            "created_at": run.created_at,
        }
        for run in rows
    ]


async def saved_stacks(db: AsyncSession, user: User) -> list[dict[str, Any]]:
    """Recent stacks, each re-scored against the catalog as it is now."""
    rows = (
        (
            await db.execute(
                scoped.owned_by(Stack, user).order_by(Stack.updated_at.desc()).limit(RECENT_STACKS)
            )
        )
        .scalars()
        .all()
    )
    if not rows:
        return []

    catalog = {tool.slug: tool for tool in await catalog_service.list_tools(db)}
    return [
        {
            "id": stack.id,
            "name": stack.name,
            "components": len(stack.component_slugs),
            "version": stack.current_version,
            "deprecated": [
                slug
                for slug in stack.component_slugs
                if slug in catalog and catalog[slug].status not in RECOMMENDABLE
            ],
            "updated_at": stack.updated_at,
        }
        for stack in rows
    ]


async def projects(db: AsyncSession, user: User) -> list[dict[str, Any]]:
    rows = await project_service.list_for(db, user)
    if not rows:
        return []

    counted = (
        await db.execute(
            select(ProjectItem.project_id, func.count())
            .where(ProjectItem.project_id.in_([project.id for project in rows]))
            .group_by(ProjectItem.project_id)
        )
    ).all()
    counts: dict[str, int] = {row[0]: int(row[1]) for row in counted}
    return [
        {
            "id": project.id,
            "name": project.name,
            "use_case": project.use_case,
            "items": counts.get(project.id, 0),
            "updated_at": project.updated_at,
        }
        for project in rows
    ]


async def stale_alerts(db: AsyncSession, user: User) -> list[dict[str, Any]]:
    """Saved stacks holding a tool that has since been buried.

    The retention mechanic from `PRD.md` §24, as a surface rather than an
    email. Computable at all only because the score is not stored (D-27).
    """
    rows = (await db.execute(scoped.owned_by(Stack, user))).scalars().all()
    if not rows:
        return []

    catalog = {tool.slug: tool for tool in await catalog_service.list_tools(db)}
    alerts: list[dict[str, Any]] = []

    for stack in rows:
        for slug in stack.component_slugs:
            tool = catalog.get(slug)
            if tool is None:
                alerts.append(
                    {
                        "stack_id": stack.id,
                        "stack_name": stack.name,
                        "tool": slug,
                        "status": "removed",
                        "reason": "This tool is no longer in the catalog.",
                        "alternatives": [],
                    }
                )
            elif tool.status not in RECOMMENDABLE:
                alerts.append(
                    {
                        "stack_id": stack.id,
                        "stack_name": stack.name,
                        "tool": tool.name,
                        "status": tool.status,
                        "reason": tool.status_reason or "No reason recorded.",
                        "alternatives": tool.alternatives,
                    }
                )
    return alerts


async def quick_start(db: AsyncSession, user: User) -> list[str]:
    """The tools this user actually reaches for, or nothing for a new one.

    An empty list rather than invented defaults: the frontend already knows a
    sensible starting six, and inventing a "most used" list for someone with
    no history would be a personalised surface that is not personalised.
    """
    rows = (
        await db.execute(
            select(ToolRun.tool_slug, func.count().label("runs"))
            .where(ToolRun.user_id == user.id)
            .group_by(ToolRun.tool_slug)
            .order_by(func.count().desc())
            .limit(6)
        )
    ).all()
    return [row.tool_slug for row in rows]


async def overview(db: AsyncSession, user: User) -> dict[str, Any]:
    return {
        "recent_runs": await recent_runs(db, user),
        "saved_stacks": await saved_stacks(db, user),
        "projects": await projects(db, user),
        "stale_alerts": await stale_alerts(db, user),
        "quick_start": await quick_start(db, user),
        "usage": {
            **await run_service.counts_for(db, user),
            "projects": await project_service.count_for(db, user),
            "project_limit": project_service.limit_for(user),
        },
        "plan": {"plan": user.plan.value, "source": user.plan_source.value},
    }


# ── search ───────────────────────────────────────────────────────────────────


class SearchHit(dict[str, Any]):
    pass


async def search(
    db: AsyncSession, user: User, query: str, *, limit: int = 20
) -> list[dict[str, Any]]:
    """One search across saved work — projects, stacks, and runs.

    Postgres full-text over the fields a person would actually type, plus a
    prefix match on tool slug so "vram" finds the VRAM runs. At the volume of
    one user's saved work a dedicated search service would be infrastructure
    without a benefit.

    Every branch is owner-scoped through the repository, so a search cannot
    become a way to enumerate other people's work — which is exactly what an
    unscoped search would be.
    """
    term = query.strip()
    if not term:
        return []

    pattern = f"%{term.lower()}%"
    tsquery = func.websearch_to_tsquery("english", term)
    hits: list[dict[str, Any]] = []

    project_rows = (
        (
            await db.execute(
                scoped.owned_by(Project, user)
                .where(
                    or_(
                        func.to_tsvector(
                            "english",
                            func.coalesce(Project.name, "")
                            + " "
                            + func.coalesce(Project.description, ""),
                        ).op("@@")(tsquery),
                        func.lower(Project.name).like(pattern),
                    )
                )
                .limit(limit)
            )
        )
        .scalars()
        .all()
    )
    hits += [
        {
            "kind": "project",
            "id": project.id,
            "title": project.name,
            "subtitle": project.use_case or "Project",
            "href": f"/projects/{project.id}",
            "updated_at": project.updated_at,
        }
        for project in project_rows
    ]

    stack_rows = (
        (
            await db.execute(
                scoped.owned_by(Stack, user)
                .where(
                    or_(
                        func.lower(Stack.name).like(pattern),
                        # A stack is findable by what is in it, not only by
                        # what it was named — "qdrant" should find the stack
                        # that uses Qdrant.
                        Stack.component_slugs.any(literal(term.lower())),
                    )
                )
                .limit(limit)
            )
        )
        .scalars()
        .all()
    )
    hits += [
        {
            "kind": "stack",
            "id": stack.id,
            "title": stack.name,
            "subtitle": ", ".join(stack.component_slugs[:4]),
            "href": f"/stack-architect/my-stacks?stack={stack.id}",
            "updated_at": stack.updated_at,
        }
        for stack in stack_rows
    ]

    run_rows = (
        (
            await db.execute(
                scoped.owned_by(ToolRun, user)
                .where(
                    ToolRun.saved.is_(True),
                    or_(
                        ToolRun.tool_slug.like(f"%{term.lower()}%"),
                        ToolRun.workflow.like(f"%{term.lower()}%"),
                    ),
                )
                .order_by(ToolRun.created_at.desc())
                .limit(limit)
            )
        )
        .scalars()
        .all()
    )
    hits += [
        {
            "kind": "run",
            "id": run.id,
            "title": run.tool_slug,
            "subtitle": (headline_of(run) or {}).get("value", run.workflow),
            "href": f"/{run.workflow}?run={run.id}",
            "updated_at": run.created_at,
        }
        for run in run_rows
    ]

    hits.sort(key=lambda hit: _sort_key(hit["updated_at"]), reverse=True)
    return hits[:limit]


def _sort_key(value: datetime) -> float:
    return value.timestamp()
