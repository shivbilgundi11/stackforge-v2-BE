"""Cost Planner endpoints (WF1).

Each one is the three lines M08 promised: fetch what the compute needs, call
`run_tool`, return. Quota, run logging, provenance, and AI enrichment all
happen inside `run_tool` and cannot be forgotten here.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from app.api.deps import Db, RunIdentity
from app.core.errors import NotFound, ValidationFailed
from app.core.responses import Envelope, ok
from app.schemas.cost import (
    BudgetEstimatorIn,
    EmbeddingCostIn,
    LlmPricingIn,
    TokenCalculatorIn,
)
from app.schemas.tools import ToolOutput, ToolRunOut
from app.services import (
    ai_service,
    catalog_service,
    cost_service,
    tokenizer_service,
    tool_service,
)
from app.services.cost_service import WorkloadLine

router = APIRouter(tags=["cost"])

WORKFLOW = "cost"


@router.post("/llm-pricing", response_model=Envelope[ToolRunOut], name="run_llm_pricing")
async def run_llm_pricing(db: Db, identity: RunIdentity, payload: LlmPricingIn) -> dict[str, Any]:
    model = await catalog_service.get_model(db, payload.model_id)
    alternatives = await catalog_service.list_models(
        db, family="chat", provider=payload.compare_provider
    )

    result = await tool_service.run_tool(
        db,
        slug="llm-pricing",
        workflow=WORKFLOW,
        payload=payload,
        identity=identity,
        compute=lambda: cost_service.llm_pricing(
            model=model,
            input_tokens=payload.input_tokens,
            output_tokens=payload.output_tokens,
            requests_per_day=payload.requests_per_day,
            cached_input_ratio=payload.cached_input_ratio,
            alternatives=alternatives,
        ),
    )
    return ok(result)


@router.post("/token-calculator", response_model=Envelope[ToolRunOut], name="run_token_calculator")
async def run_token_calculator(
    db: Db, identity: RunIdentity, payload: TokenCalculatorIn
) -> dict[str, Any]:
    model = await catalog_service.get_model(db, payload.model_id)
    candidates = await catalog_service.list_models(db, family="chat")

    # Counted before `run_tool`, because counting a Claude model is an API
    # call and the compute function is pure by contract. `method` rides the
    # response so the user knows whether the number was measured.
    counted = await tokenizer_service.count(payload.text, model=model)

    result = await tool_service.run_tool(
        db,
        slug="token-calculator",
        workflow=WORKFLOW,
        payload=payload,
        identity=identity,
        compute=lambda: cost_service.token_calculator(
            text=payload.text,
            model=model,
            candidates=candidates,
            output_tokens=payload.output_tokens,
            counted=(counted.tokens, counted.method),
        ),
    )
    return ok(result)


@router.post("/embedding-cost", response_model=Envelope[ToolRunOut], name="run_embedding_cost")
async def run_embedding_cost(
    db: Db, identity: RunIdentity, payload: EmbeddingCostIn
) -> dict[str, Any]:
    model = await catalog_service.get_model(db, payload.model_id)
    if model.family != "embedding":
        raise ValidationFailed.on_field(
            "model_id",
            "Choose an embedding model.",
            summary=f"{model.display_name} is a {model.family} model, not an embedding model.",
        )
    alternatives = [
        candidate
        for candidate in await catalog_service.list_models(db, family="embedding")
        if candidate.model_id != model.model_id
    ]

    result = await tool_service.run_tool(
        db,
        slug="embedding-cost",
        workflow=WORKFLOW,
        payload=payload,
        identity=identity,
        compute=lambda: cost_service.embedding_cost(
            model=model,
            document_count=payload.document_count,
            avg_tokens_per_document=payload.avg_tokens_per_document,
            reembeds_per_month=payload.reembeds_per_month,
            chunk_overlap_pct=payload.chunk_overlap_pct,
            alternatives=alternatives,
        ),
    )
    return ok(result)


@router.post("/budget-estimator", response_model=Envelope[ToolRunOut], name="run_budget_estimator")
async def run_budget_estimator(
    db: Db, identity: RunIdentity, payload: BudgetEstimatorIn
) -> dict[str, Any]:
    wanted = [line.model_id for line in payload.lines]
    models = await catalog_service.get_models_by_ids(db, wanted)

    missing = [model_id for model_id in wanted if model_id not in models]
    if missing:
        raise NotFound(f"No pricing for: {', '.join(sorted(set(missing)))}.")

    lines = [
        WorkloadLine(
            name=line.name,
            model=models[line.model_id],
            requests_per_day=line.requests_per_day,
            input_tokens=line.input_tokens,
            output_tokens=line.output_tokens,
        )
        for line in payload.lines
    ]

    result = await tool_service.run_tool(
        db,
        slug="budget-estimator",
        workflow=WORKFLOW,
        payload=payload,
        identity=identity,
        compute=lambda: cost_service.budget_estimator(
            lines=lines,
            monthly_growth_pct=payload.monthly_growth_pct,
            infrastructure_monthly=payload.infrastructure_monthly,
            embedding_monthly=payload.embedding_monthly,
            user_count=payload.user_count,
        ),
        # The estimator is the synthesis tool of the cost workflow — the point
        # where several workloads become one number someone has to defend.
        # The engine already names the mechanical savings it can compute
        # exactly, caching and batching; the model reads the shape of the bill
        # and names what to change about the workload itself.
        enrich=ai_service.enrichment(
            db,
            purpose="cost_optimization",
            identity=identity,
            tool_slug="budget-estimator",
            variables=payload.model_dump(mode="json"),
            apply=_apply_optimisations,
            grounding=_optimisation_grounding,
        ),
    )
    return ok(result)


def _optimisation_grounding(output: ToolOutput) -> dict[str, Any]:
    """The bill, not the projection.

    Twelve months of compounded growth is twelve rows of the same arithmetic,
    and none of them change which line to cut. The per-workload breakdown is
    what a suggestion has to be supported by, so the projection stays out.
    """
    return {
        "metrics": {key: str(value) for key, value in output.metrics.items()},
        "breakdown": output.tables.get("breakdown", []),
        # The engine's own suggestions, labelled so the model extends them
        # rather than restating them. Handed over under a neutral name they
        # came back as the same two changes with worse arithmetic.
        "already_suggested_do_not_repeat": [
            row.get("detail") for row in output.tables.get("recommendations", [])
        ],
    }


def _apply_optimisations(output: ToolOutput, data: dict[str, Any]) -> None:
    """Append, never replace.

    The engine's rows carry a computed dollar saving; the model's carry a
    judgement about what the change costs in quality or effort. Dropping the
    first for the second would trade a number for an opinion, so both are
    shown and the `kind` column says which is which.
    """
    rows = [dict(row) for row in output.tables.get("recommendations", [])]
    for suggestion in data.get("suggestions") or []:
        if not isinstance(suggestion, dict):
            continue
        change = str(suggestion.get("change") or "").strip()
        if not change:
            continue
        costs_you = str(suggestion.get("costs_you") or "").strip()

        # A suggestion the engine already made arrives as an annotation on
        # that row, not as a second row saying the same thing. The engine
        # computed the saving and cannot judge what the change costs in
        # quality; the model is the other way round. Merged, the row carries
        # both — which is more than either produced alone, and is why this is
        # a merge rather than a duplicate filter.
        matched = next((row for row in rows if ai_service.echoes(change, str(row["detail"]))), None)
        if matched is not None:
            if costs_you:
                matched["trade_off"] = costs_you
            continue

        rows.append(
            {
                "line": "workload shape",
                "kind": "ai_suggestion",
                "detail": change,
                "monthly_saving": str(suggestion.get("saves") or "").strip() or "not costed",
                "trade_off": costs_you or "not stated",
            }
        )

    # Every row carries the column or none does: a table where half the cells
    # in a column are missing renders as a column of blanks.
    if any("trade_off" in row for row in rows):
        for row in rows:
            row.setdefault("trade_off", "—")
    output.tables["recommendations"] = rows
