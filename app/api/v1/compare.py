"""Compare Center endpoints.

Four comparisons, one output contract, one renderer. Every one accepts a
`priority` that reweights the criteria — see `app/data/compare_criteria.py`.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.api.deps import Db, RunIdentity
from app.core.errors import NotFound, ValidationFailed
from app.core.responses import Envelope, ok
from app.data.compare_criteria import PRIORITIES, STACK_ARCHETYPES, Priority
from app.schemas.tools import ToolRunOut
from app.services import catalog_service, compare_service, tool_service

router = APIRouter(tags=["compare"])

WORKFLOW = "compare"


class CompareModelsIn(BaseModel):
    model_ids: list[str] = Field(min_length=2, max_length=4)
    input_tokens: int = Field(default=2000, ge=0, le=10_000_000)
    output_tokens: int = Field(default=500, ge=0, le=1_000_000)
    requests_per_day: int = Field(default=1000, ge=0, le=100_000_000)
    cached_input_ratio: Decimal = Field(default=Decimal(0), ge=0, le=1)
    priority: Priority = "balanced"


class CompareVectorDbIn(BaseModel):
    tool_slugs: list[str] = Field(min_length=2, max_length=6)
    vector_count: int = Field(default=1_000_000, ge=1, le=10_000_000_000)
    dimensions: int = Field(default=1536, ge=1, le=16_384)
    priority: Priority = "balanced"


class CompareStacksIn(BaseModel):
    archetypes: list[str] = Field(min_length=2, max_length=5)
    monthly_model_spend: Decimal = Field(default=Decimal(500), ge=0)
    blended_hourly_rate: Decimal = Field(default=Decimal(120), ge=1, le=1000)
    priority: Priority = "balanced"


class CompareBuildVsBuyIn(BaseModel):
    build_hours: int = Field(ge=1, le=100_000)
    blended_hourly_rate: Decimal = Field(default=Decimal(120), ge=1, le=1000)
    build_infra_monthly: Decimal = Field(default=Decimal(0), ge=0)
    maintenance_hours_per_month: Decimal = Field(default=Decimal(0), ge=0, le=1000)
    vendor_monthly: Decimal = Field(ge=0)
    vendor_integration_hours: int = Field(default=0, ge=0, le=10_000)
    priority: Priority = "balanced"


@router.post("/models", response_model=Envelope[ToolRunOut], name="run_compare_models")
async def run_compare_models(
    db: Db, identity: RunIdentity, payload: CompareModelsIn
) -> dict[str, Any]:
    found = await catalog_service.get_models_by_ids(db, payload.model_ids)
    missing = [model_id for model_id in payload.model_ids if model_id not in found]
    if missing:
        raise NotFound(f"No pricing for: {', '.join(sorted(set(missing)))}.")

    # Preserve the caller's order so the matrix columns match what they asked
    # for, not whatever order Postgres returned.
    models = [found[model_id] for model_id in payload.model_ids]

    result = await tool_service.run_tool(
        db,
        slug="compare-models",
        workflow=WORKFLOW,
        payload=payload,
        identity=identity,
        compute=lambda: compare_service.compare_models(
            models=models,
            input_tokens=payload.input_tokens,
            output_tokens=payload.output_tokens,
            requests_per_day=payload.requests_per_day,
            cached_input_ratio=payload.cached_input_ratio,
            priority=payload.priority,
        ),
    )
    return ok(result)


@router.post("/vector-db", response_model=Envelope[ToolRunOut], name="run_compare_vector_db")
async def run_compare_vector_db(
    db: Db, identity: RunIdentity, payload: CompareVectorDbIn
) -> dict[str, Any]:
    tools = []
    for slug in payload.tool_slugs:
        tool = await catalog_service.get_tool(db, slug)
        if tool.category != "vector-db":
            raise ValidationFailed.on_field(
                "tool_slugs",
                f"{tool.name} is a {tool.category}, not a vector database.",
            )
        tools.append(tool)

    result = await tool_service.run_tool(
        db,
        slug="compare-vector-db",
        workflow=WORKFLOW,
        payload=payload,
        identity=identity,
        compute=lambda: compare_service.compare_vector_db(
            tools=tools,
            vector_count=payload.vector_count,
            dimensions=payload.dimensions,
            priority=payload.priority,
        ),
    )
    return ok(result)


@router.post("/stacks", response_model=Envelope[ToolRunOut], name="run_compare_stacks")
async def run_compare_stacks(
    db: Db, identity: RunIdentity, payload: CompareStacksIn
) -> dict[str, Any]:
    archetypes = compare_service.resolve_archetypes(payload.archetypes)
    if len(archetypes) < 2:
        known = ", ".join(a.key for a in STACK_ARCHETYPES)
        raise ValidationFailed(f"Pick at least two known archetypes from: {known}.")

    result = await tool_service.run_tool(
        db,
        slug="compare-stacks",
        workflow=WORKFLOW,
        payload=payload,
        identity=identity,
        compute=lambda: compare_service.compare_stacks(
            archetypes=archetypes,
            monthly_model_spend=payload.monthly_model_spend,
            blended_hourly_rate=payload.blended_hourly_rate,
            priority=payload.priority,
        ),
    )
    return ok(result)


@router.post("/build-vs-buy", response_model=Envelope[ToolRunOut], name="run_compare_build_vs_buy")
async def run_compare_build_vs_buy(
    db: Db, identity: RunIdentity, payload: CompareBuildVsBuyIn
) -> dict[str, Any]:
    result = await tool_service.run_tool(
        db,
        slug="compare-build-vs-buy",
        workflow=WORKFLOW,
        payload=payload,
        identity=identity,
        compute=lambda: compare_service.compare_build_vs_buy(
            build_hours=payload.build_hours,
            blended_hourly_rate=payload.blended_hourly_rate,
            build_infra_monthly=payload.build_infra_monthly,
            maintenance_hours_per_month=payload.maintenance_hours_per_month,
            vendor_monthly=payload.vendor_monthly,
            vendor_integration_hours=payload.vendor_integration_hours,
            priority=payload.priority,
        ),
    )
    return ok(result)


class ComparePriorityOut(BaseModel):
    key: str
    label: str
    description: str


class StackArchetypeOut(BaseModel):
    key: str
    name: str
    description: str
    components: list[str]


class CompareMetaOut(BaseModel):
    priorities: list[ComparePriorityOut]
    stack_archetypes: list[StackArchetypeOut]


PRIORITY_LABELS: dict[str, tuple[str, str]] = {
    "balanced": ("Balanced", "No axis favoured. A reasonable default."),
    "cost": ("Cost", "Weight spend heavily; accept more operational work to save money."),
    "scale": ("Scale", "Weight headroom; assume this has to survive 10x growth."),
    "speed": ("Speed", "Weight latency and time to ship over long-run cost."),
    "simplicity": ("Simplicity", "Weight low operational burden; prefer managed."),
    "control": ("Control", "Weight portability and self-hosting; avoid lock-in."),
}


@router.get("/meta", response_model=Envelope[CompareMetaOut], name="get_compare_meta")
async def get_compare_meta() -> dict[str, Any]:
    """Priorities and stack archetypes, so the frontend does not hardcode them."""
    return ok(
        CompareMetaOut(
            priorities=[
                ComparePriorityOut(
                    key=key,
                    label=PRIORITY_LABELS[key][0],
                    description=PRIORITY_LABELS[key][1],
                )
                for key in PRIORITIES
            ],
            stack_archetypes=[
                StackArchetypeOut(
                    key=archetype.key,
                    name=archetype.name,
                    description=archetype.description,
                    components=list(archetype.components),
                )
                for archetype in STACK_ARCHETYPES
            ],
        )
    )
