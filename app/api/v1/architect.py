"""Stack Architect endpoints (M15).

`/recommend` is **public**. An anonymous visitor gets one full recommendation
per the anonymous tool quota, because the product's strongest demo is the
product working — gating it behind signup trades the best conversion moment
for a smaller one. Saving requires an account, and that is the conversion
moment.

Compatibility and the graveyard are not re-exposed here: `/catalog/compatibility`
and `/catalog/graveyard` already serve exactly this contract from M07, and a
second path to the same data is a second thing to keep in step.
"""

from __future__ import annotations

import asyncio
from decimal import Decimal
from typing import Any

from fastapi import APIRouter

from app.api.deps import CurrentMembership, Db, RunIdentity
from app.core.errors import NotFound
from app.core.responses import Envelope, ok
from app.models.organization import Organization
from app.schemas.architect import RecommendIn, ScoreIn
from app.schemas.tools import Artifact, ToolOutput, ToolRunOut, ToolWarning
from app.services import (
    ai_service,
    catalog_service,
    stack_architect_service,
    stack_diagram_service,
    stack_score_service,
    tool_service,
)
from app.services.stack_architect_service import Requirements

router = APIRouter(tags=["architect"])

WORKFLOW = "architect"


def _requirements(payload: RecommendIn) -> Requirements:
    return Requirements(
        use_case=payload.use_case,
        scale_target=payload.scale_target,
        monthly_budget=payload.monthly_budget,
        team_skill=payload.team_skill,
        latency_ms=payload.latency_ms,
        sensitivity=payload.sensitivity,
        deployment=payload.deployment,
        capabilities=tuple(payload.capabilities),
    )


@router.post("/recommend", response_model=Envelope[ToolRunOut], name="run_recommend_stack")
async def run_recommend_stack(
    db: Db, identity: RunIdentity, member: CurrentMembership, payload: RecommendIn
) -> dict[str, Any]:
    requirements = _requirements(payload)
    catalog = await catalog_service.list_tools(db)

    # The organization's approved-tool list, when the caller is acting inside
    # one (M21). It prefers and flags — `eliminate` never sees it, because a
    # policy must not silently exclude the best answer.
    approved: frozenset[str] = frozenset()
    if member is not None:
        org = await db.get(Organization, member.organization_id)
        if org is not None:
            from app.services import organization_service

            approved = frozenset(organization_service.settings_of(org).approved_tools)

    survivors, eliminations = stack_architect_service.eliminate(catalog, requirements)
    candidates = stack_architect_service.assemble(survivors, requirements)

    # Compatibility is a database read, so it happens here rather than inside
    # the pure scoring path.
    #
    # Concurrently, not in a loop. These are five independent reads, and
    # awaiting them one at a time made a 30-second request on a machine whose
    # Redis was down — each lookup paid the connection timeout in series. The
    # cache failing is meant to be slower, not five times slower.
    scored = [
        (index, components) for index, components in enumerate(candidates) if len(components) > 1
    ]
    resolved = await asyncio.gather(
        *(
            catalog_service.get_compatibility(db, [tool.slug for tool in components])
            for _, components in scored
        )
    )
    compatibilities = {index: result for (index, _), result in zip(scored, resolved, strict=True)}

    def compute() -> ToolOutput:
        ranked = stack_architect_service.prefer_approved(
            stack_architect_service.rank(candidates, requirements, compatibilities),
            approved,
        )
        if not ranked:
            return ToolOutput(
                metrics={"score": 0, "components": 0, "candidates": 0},
                tables={"exclusions": _exclusion_rows(eliminations)},
                warnings=[
                    ToolWarning(
                        level="critical",
                        message=(
                            "No stack satisfies every constraint at once. The exclusions "
                            "table shows which constraint removed what — relaxing the "
                            "tightest one is the fastest way to a result."
                        ),
                    )
                ],
            )

        winner = ranked[0]
        rows = stack_architect_service.component_rows(winner, requirements, approved=approved)
        diagram = stack_diagram_service.mermaid(winner.components, requirements)
        summary = stack_architect_service.rule_summary(winner, requirements)

        return ToolOutput(
            metrics={
                "score": winner.score.total,
                "components": len(winner.components),
                "candidates": len(candidates),
                "excluded": len(eliminations),
                "compatibility": (
                    winner.compatibility.overall if winner.compatibility else "unknown"
                ),
                "deprecated_components": len(winner.deprecated),
                "summary": summary,
            },
            tables={
                "components": rows,
                "score_breakdown": winner.score.breakdown(),
                "alternatives": _alternative_rows(ranked[1:], requirements),
                "compatibility": _compatibility_rows(winner),
                "exclusions": _exclusion_rows(eliminations),
                "roadmap": [],
            },
            artifacts=[
                Artifact(
                    type="diagram",
                    format="mermaid",
                    filename="stack-architecture.mmd",
                    content=diagram,
                ),
                Artifact(
                    type="architecture",
                    format="markdown",
                    filename="stack-architecture.md",
                    content=stack_diagram_service.document(
                        components=winner.components,
                        requirements=requirements,
                        summary=summary,
                        diagram=diagram,
                        score_rows=winner.score.breakdown(),
                        component_rows=rows,
                        roadmap=[],
                    ),
                ),
            ],
            warnings=(
                stack_architect_service.warnings_for(winner, requirements, eliminations)
                + stack_architect_service.approved_flags(winner, approved)
            ),
            sourced_from=[tool.slug for tool in winner.components],
        )

    result = await tool_service.run_tool(
        db,
        slug="stack-architect",
        workflow=WORKFLOW,
        payload=payload,
        identity=identity,
        compute=compute,
        # The model picks among the candidates the engine produced and writes
        # the rationale. With no key the ranked result ships unchanged, marked
        # `rule_based` — the flagship degrades to a complete answer, never to
        # an error page (D-06).
        enrich=ai_service.enrichment(
            db,
            purpose="stack_synthesis",
            identity=identity,
            tool_slug="stack-architect",
            variables=payload.model_dump(mode="json"),
            apply=_apply_synthesis,
            generate=ai_service.generate_gemini_json,
        ),
    )
    return ok(result)


def _apply_synthesis(output: ToolOutput, data: dict[str, Any]) -> None:
    """Merge Gemini's grounded assessment and written analysis."""
    _apply_gemini_scores(output, data.get("score_breakdown"))
    if summary := str(data.get("summary") or "").strip():
        output.metrics["summary"] = summary
    if why := str(data.get("why") or "").strip():
        output.metrics["rationale"] = why
    if confidence := str(data.get("confidence") or "").strip():
        output.metrics["confidence"] = confidence

    rationale: list[dict[str, Any]] = []
    rationale += [{"kind": "tradeoff", "text": str(item)} for item in data.get("trade_offs") or []]
    rationale += [
        {"kind": "switch_when", "text": str(item)} for item in data.get("switch_when") or []
    ]
    if rationale:
        output.tables["rationale"] = rationale

    for risk in data.get("risks") or []:
        level = {"high": "critical", "medium": "warning"}.get(str(risk.get("severity")), "info")
        output.warnings.append(
            ToolWarning(level=level, message=f"{risk.get('risk')} — {risk.get('mitigation')}")
        )


def _apply_gemini_scores(output: ToolOutput, raw: object) -> None:
    """Replace every visible score row together, or leave the rule score intact."""
    if not isinstance(raw, list):
        return
    scores: dict[str, Decimal] = {}
    for item in raw:
        if not isinstance(item, dict):
            return
        key, value = item.get("key"), item.get("score")
        if (
            not isinstance(key, str)
            or not isinstance(value, (int, float))
            or isinstance(value, bool)
        ):
            return
        score = Decimal(str(value))
        if not Decimal(0) <= score <= Decimal(10):
            return
        scores[key] = score

    rows = output.tables.get("score_breakdown")
    if not isinstance(rows, list) or set(scores) != set(stack_score_service.BY_KEY):
        return

    total = Decimal(0)
    for row in rows:
        key = row.get("key")
        if not isinstance(key, str) or key not in scores:
            return
        score = scores[key].quantize(Decimal("0.1"))
        contribution = (score * Decimal(str(row["weight_pct"])) / 10).quantize(Decimal("0.1"))
        row["score"] = str(score)
        row["contribution"] = str(contribution)
        total += contribution
    output.metrics["score"] = total.quantize(Decimal("0.1"))


def _alternative_rows(
    others: list[stack_architect_service.Candidate], requirements: Requirements
) -> list[dict[str, Any]]:
    """Ranks 2 and 3, and what each one trades.

    The trade is spelled out per dimension rather than left as a score gap: a
    reader deciding between 87 and 84 needs to know *which* three points.
    """
    rows: list[dict[str, Any]] = []
    for candidate in others:
        rows.append(
            {
                "rank": candidate.rank,
                "score": str(candidate.score.total),
                "components": ", ".join(tool.name for tool in candidate.components),
                "strongest": _extreme(candidate, best=True),
                "weakest": _extreme(candidate, best=False),
            }
        )
    return rows


def _extreme(candidate: stack_architect_service.Candidate, *, best: bool) -> str:
    scores = candidate.score.dimensions
    if not scores:
        return "—"
    key = (max if best else min)(scores, key=lambda name: scores[name])
    return f"{stack_score_service.BY_KEY[key].label} ({scores[key]}/10)"


def _compatibility_rows(candidate: stack_architect_service.Candidate) -> list[dict[str, Any]]:
    if candidate.compatibility is None:
        return []
    return [
        {
            "pair": f"{pair.tool_a} + {pair.tool_b}",
            "score": pair.score,
            "status": _status_of(pair.score),
            "notes": pair.notes or "",
        }
        for pair in candidate.compatibility.pairs
    ]


def _status_of(score: int) -> str:
    """Value plus a word, never colour alone (M04)."""
    if score >= 85:
        return "Excellent"
    if score >= 70:
        return "Good"
    if score >= 50:
        return "Caution"
    return "Incompatible"


def _exclusion_rows(
    eliminations: list[stack_architect_service.Elimination],
) -> list[dict[str, Any]]:
    """Every tool a hard constraint removed, and which constraint did it.

    Shown rather than logged: a recommendation missing the tool the user
    expected reads as a broken engine unless it says why.
    """
    return [
        {
            "tool": elimination.name,
            "slug": elimination.slug,
            "constraint": elimination.constraint.replace("_", " "),
            "reason": elimination.reason,
        }
        for elimination in eliminations
    ]


@router.post("/score", response_model=Envelope[ToolRunOut], name="run_score_stack")
async def run_score_stack(db: Db, identity: RunIdentity, payload: ScoreIn) -> dict[str, Any]:
    """Score a stack the user assembled themselves."""
    catalog = await catalog_service.list_tools(db)
    by_slug = {tool.slug: tool for tool in catalog}

    missing = [slug for slug in payload.component_slugs if slug not in by_slug]
    if missing:
        raise NotFound(f"No catalog entry for: {', '.join(sorted(missing))}.")

    components = [by_slug[slug] for slug in payload.component_slugs]
    # No pairs to score in a one-component stack; asking anyway raises.
    compatibility = (
        await catalog_service.get_compatibility(db, payload.component_slugs)
        if len(payload.component_slugs) > 1
        else None
    )

    def compute() -> ToolOutput:
        stack_score = stack_score_service.score(
            components,
            monthly_budget=payload.monthly_budget,
            scale_target=payload.scale_target,
            sensitivity=payload.sensitivity,
            compatibility=compatibility,
        )
        recommendable = stack_architect_service.RECOMMENDABLE
        buried = [tool for tool in components if tool.status not in recommendable]

        warnings = [
            ToolWarning(
                level="critical",
                message=(
                    f"{tool.name} is marked {tool.status}: "
                    f"{tool.status_reason or 'no reason recorded'}."
                ),
            )
            for tool in buried
        ]

        return ToolOutput(
            metrics={
                "score": stack_score.total,
                "components": len(components),
                "compatibility": compatibility.overall if compatibility else "unknown",
                "deprecated_components": len(buried),
            },
            tables={
                "score_breakdown": stack_score.breakdown(),
                "compatibility": [
                    {
                        "pair": f"{pair.tool_a} + {pair.tool_b}",
                        "score": pair.score,
                        "status": _status_of(pair.score),
                        "notes": pair.notes or "",
                    }
                    for pair in (compatibility.pairs if compatibility else [])
                ],
            },
            warnings=warnings,
            sourced_from=payload.component_slugs,
        )

    result = await tool_service.run_tool(
        db,
        slug="stack-score",
        workflow=WORKFLOW,
        payload=payload,
        identity=identity,
        compute=compute,
    )
    return ok(result)
