"""Stack Architect endpoints (M15).

`/recommend` needs an account like every other tool route. A new free account
gets a full recommendation inside its daily allowance, because the product's
strongest demo is the
product working — gating it behind signup trades the best conversion moment
for a smaller one. Saving requires an account, and that is the conversion
moment.

Compatibility and the graveyard are not re-exposed here: `/catalog/compatibility`
and `/catalog/graveyard` already serve exactly this contract from M07, and a
second path to the same data is a second thing to keep in step.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any, Final

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
        model_hosting=payload.model_hosting,
        workload=payload.workload,
        traffic=payload.traffic,
        residency=payload.residency,
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

    # Filled by `compute`, read by the two passes that run after it: synthesis
    # may re-point the result at another candidate, and the roadmap pass
    # rebuilds the document around its steps. Held here rather than in closure
    # variables because each half is written and read in a different function.
    document_inputs: dict[str, Any] = {}
    ranked_stacks: list[stack_architect_service.Candidate] = []

    def build(winner: stack_architect_service.Candidate) -> ToolOutput:
        """The complete result for one candidate.

        Called by `compute` for the engine's leader, and called again by the
        synthesis pass when the model picks a different one — M15 layer 2, "the
        model chooses among options the engine produced". One builder rather
        than two: every table, the diagram, the alternatives and the exported
        document all describe *a* stack, and a swap that rebuilt some of them
        would put one stack's picture over another's numbers.
        """
        rows = stack_architect_service.component_rows(winner, requirements, approved=approved)
        diagram = stack_diagram_service.mermaid(winner.components, requirements)
        summary = stack_architect_service.rule_summary(winner, requirements)
        others = [candidate for candidate in ranked_stacks if candidate.rank != winner.rank]

        # The compute vendor's representative instance, so the result page can
        # open `gpu-cost` on a real machine rather than on its own defaults
        # (M25). Absent on the stacks that have no compute layer, which is
        # most of them, and the handoff hides itself accordingly.
        compute = next((tool for tool in winner.components if tool.category == "gpu-cloud"), None)

        # What the exported document is built from, kept so the roadmap pass
        # can rebuild it with the steps in place. Re-rendering the whole file
        # is what keeps the download and the screen agreeing; patching the
        # placeholder line out of the finished markdown would not.
        document_inputs.update(
            components=winner.components,
            requirements=requirements,
            summary=summary,
            diagram=diagram,
            score_rows=winner.score.breakdown(),
            component_rows=rows,
        )

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
                **(
                    {"compute_gpu": str((compute.facts or {}).get("gpu_slug", ""))}
                    if compute is not None and (compute.facts or {}).get("gpu_slug")
                    else {}
                ),
            },
            tables={
                "components": rows,
                "score_breakdown": winner.score.breakdown(),
                "alternatives": _alternative_rows(others, requirements),
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
                        **document_inputs,
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

        ranked_stacks[:] = ranked
        return build(ranked[0])

    result = await tool_service.run_tool(
        db,
        slug="stack-architect",
        workflow=WORKFLOW,
        payload=payload,
        identity=identity,
        compute=compute,
        # Two passes, because they are two different jobs. The assessment
        # picks among the candidates the engine produced and writes the
        # rationale; the roadmap turns the chosen stack into an ordered build
        # plan. Splitting them keeps each prompt answering one question, which
        # is also what lets the roadmap survive an assessment that failed.
        #
        # With no key at all the ranked result ships unchanged, marked
        # `rule_based` — the flagship degrades to a complete answer, never to
        # an error page (D-06).
        enrich=ai_service.chain(
            ai_service.enrichment(
                db,
                purpose="stack_synthesis",
                identity=identity,
                tool_slug="stack-architect",
                variables=payload.model_dump(mode="json"),
                apply=_synthesis_applier(build, ranked_stacks),
            ),
            ai_service.enrichment(
                db,
                purpose="roadmap",
                identity=identity,
                tool_slug="stack-architect",
                variables=payload.model_dump(mode="json"),
                apply=_roadmap_applier(document_inputs),
                grounding=_roadmap_grounding,
            ),
        ),
    )
    return ok(result)


ROADMAP_KEYS: Final = ("title", "detail", "effort", "depends_on", "done_when")


def _roadmap_grounding(output: ToolOutput) -> dict[str, Any]:
    """The chosen stack, and nothing else.

    The default grounding is the whole result — scores, alternatives,
    exclusions, the compatibility matrix. None of that tells anyone what to
    build first, and all of it competes for the same input budget as the part
    that does. Handing over the components and the summary keeps the steps
    about the stack that won rather than about the ones that lost.
    """
    return {
        "summary": str(output.metrics.get("summary", "")),
        "components": output.tables.get("components", []),
        "compatibility": output.tables.get("compatibility", []),
    }


def _roadmap_applier(
    document_inputs: dict[str, Any],
) -> Callable[[ToolOutput, dict[str, Any]], None]:
    """Fill the roadmap table, and rebuild the document around it.

    Two places render the roadmap — the result page reads `tables.roadmap`,
    the exported markdown embeds it — and they are built at different moments.
    Updating only the first is how a page and its own download come to
    disagree, which is the one failure this artifact exists to avoid.
    """

    def apply(output: ToolOutput, data: dict[str, Any]) -> None:
        steps: list[dict[str, Any]] = [
            {key: str(step.get(key) or "") for key in ROADMAP_KEYS}
            for step in data.get("steps") or []
            if isinstance(step, dict)
        ]
        if not steps:
            return

        output.tables["roadmap"] = steps
        if not document_inputs:  # pragma: no cover — set whenever a stack ranked
            return
        for artifact in output.artifacts:
            if artifact.type == "architecture":
                artifact.content = stack_diagram_service.document(
                    **document_inputs, roadmap=list(steps)
                )

    return apply


def _synthesis_applier(
    build: Callable[[stack_architect_service.Candidate], ToolOutput],
    ranked: list[stack_architect_service.Candidate],
) -> Callable[[ToolOutput, dict[str, Any]], None]:
    """Merge the model's selection, assessment and written analysis.

    A closure over the builder and the ranking, because the model's first
    answer is *which* stack — and honouring that means rebuilding the result,
    not editing the one already in hand.
    """

    def apply(output: ToolOutput, data: dict[str, Any]) -> None:
        _apply_choice(output, data.get("recommended_rank"), build, ranked)
        if summary := str(data.get("summary") or "").strip():
            output.metrics["summary"] = summary
        if why := str(data.get("why") or "").strip():
            output.metrics["rationale"] = why
        if confidence := str(data.get("confidence") or "").strip():
            output.metrics["confidence"] = confidence

        rationale: list[dict[str, Any]] = []
        rationale += [
            {"kind": "tradeoff", "text": str(item)} for item in data.get("trade_offs") or []
        ]
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

    return apply


def _apply_choice(
    output: ToolOutput,
    raw: object,
    build: Callable[[stack_architect_service.Candidate], ToolOutput],
    ranked: list[stack_architect_service.Candidate],
) -> None:
    """Re-point the whole result at the candidate the model picked.

    The engine ranks and the model selects among what it ranked (M15 layer 2).
    Selecting was the one half of that contract the schema asked for and
    nothing read: the model named a stack, the page shipped the engine's
    leader, and a rationale arguing for the runner-up sat above the winner's
    component table.

    Everything is replaced together, from the builder, for the same reason the
    builder exists at all. `warnings` can be assigned rather than merged
    because nothing has appended to it yet — the risks below are the first,
    and `run_tool` only touches it if `enrich` raises.

    A rank the engine never offered leaves the leader in place: that is a
    malformed answer, and the fallback for a malformed answer is the
    deterministic result (D-06).
    """
    try:
        chosen = int(str(raw))
    except (TypeError, ValueError):
        return

    winner = next((candidate for candidate in ranked if candidate.rank == chosen), None)
    if winner is None or winner is ranked[0]:
        return

    rebuilt = build(winner)
    output.metrics = rebuilt.metrics
    output.tables = rebuilt.tables
    output.series = rebuilt.series
    output.artifacts = rebuilt.artifacts
    output.warnings = rebuilt.warnings
    output.sourced_from = rebuilt.sourced_from


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
        # The matrix is the answer; what it *costs* is the question people
        # actually arrive with. A 6/10 pairing is a number until someone says
        # it means writing your own sync job. Scores, statuses, and notes are
        # the engine's and stay untouched.
        enrich=ai_service.enrichment(
            db,
            purpose="compatibility_rationale",
            identity=identity,
            tool_slug="stack-score",
            variables=payload.model_dump(mode="json"),
            apply=_apply_compatibility_rationale,
        ),
    )
    return ok(result)


def _apply_compatibility_rationale(output: ToolOutput, data: dict[str, Any]) -> None:
    """Commentary only. The pairwise scores are the engine's."""
    if summary := str(data.get("summary") or "").strip():
        output.metrics["summary"] = summary
    if impact := str(data.get("weakest_pair_impact") or "").strip():
        output.metrics["weakest_pair_impact"] = impact
