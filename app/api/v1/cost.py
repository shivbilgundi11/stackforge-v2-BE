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
from app.schemas.tools import ToolRunOut
from app.services import catalog_service, cost_service, tool_service
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
        ),
    )
    return ok(result)


@router.post("/embedding-cost", response_model=Envelope[ToolRunOut], name="run_embedding_cost")
async def run_embedding_cost(
    db: Db, identity: RunIdentity, payload: EmbeddingCostIn
) -> dict[str, Any]:
    model = await catalog_service.get_model(db, payload.model_id)
    if model.family != "embedding":
        raise ValidationFailed(
            f"{model.display_name} is a {model.family} model, not an embedding model.",
            details={"fields": [{"field": "model_id", "message": "Choose an embedding model."}]},
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
    )
    return ok(result)
