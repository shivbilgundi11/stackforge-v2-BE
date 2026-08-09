"""ROI and business-case endpoints (WF5).

No catalog reads: every input is the user's own number, or one carried in from
WF1 or WF4 by the workflow session on the frontend. That is why these four
have no `sourced_from` and therefore no provenance chip - there is no vendor
price behind them to be stale, and showing a freshness chip on arithmetic over
the user's own figures would be theatre.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from app.api.deps import Db, RunIdentity
from app.core.responses import Envelope, ok
from app.schemas.roi import (
    HoursSavedIn,
    ImplementationCostIn,
    ModelRoiIn,
    RoiBuildVsBuyIn,
)
from app.schemas.tools import ToolRunOut
from app.services import roi_service, tool_service

router = APIRouter(tags=["roi"])

WORKFLOW = "roi"


@router.post("/hours-saved", response_model=Envelope[ToolRunOut], name="run_hours_saved")
async def run_hours_saved(db: Db, identity: RunIdentity, payload: HoursSavedIn) -> dict[str, Any]:
    result = await tool_service.run_tool(
        db,
        slug="hours-saved",
        workflow=WORKFLOW,
        payload=payload,
        identity=identity,
        compute=lambda: roi_service.hours_saved(
            affected_users=payload.affected_users,
            hours_saved_per_user_per_week=payload.hours_saved_per_user_per_week,
            fully_loaded_hourly_cost=payload.fully_loaded_hourly_cost,
            adoption_rate_pct=payload.adoption_rate_pct,
            error_rate_reduction_pct=payload.error_rate_reduction_pct,
            rework_hours_per_month=payload.rework_hours_per_month,
        ),
    )
    return ok(result)


@router.post("/model-roi", response_model=Envelope[ToolRunOut], name="run_model_roi")
async def run_model_roi(db: Db, identity: RunIdentity, payload: ModelRoiIn) -> dict[str, Any]:
    result = await tool_service.run_tool(
        db,
        slug="model-roi",
        workflow=WORKFLOW,
        payload=payload,
        identity=identity,
        compute=lambda: roi_service.model_roi(
            current_monthly_cost=payload.current_monthly_cost,
            ai_monthly_cost=payload.ai_monthly_cost,
            implementation_cost=payload.implementation_cost,
            adoption_ramp_months=payload.adoption_ramp_months,
            horizon_months=payload.horizon_months,
            discount_rate_pct=payload.discount_rate_pct,
        ),
    )
    return ok(result)


@router.post(
    "/implementation-cost",
    response_model=Envelope[ToolRunOut],
    name="run_implementation_cost",
)
async def run_implementation_cost(
    db: Db, identity: RunIdentity, payload: ImplementationCostIn
) -> dict[str, Any]:
    result = await tool_service.run_tool(
        db,
        slug="implementation-cost",
        workflow=WORKFLOW,
        payload=payload,
        identity=identity,
        compute=lambda: roi_service.implementation_cost(
            roles=[role.model_dump() for role in payload.roles],
            duration_months=payload.duration_months,
            infrastructure_setup=payload.infrastructure_setup,
            licences=payload.licences,
            training=payload.training,
            contingency_pct=payload.contingency_pct,
            ongoing_monthly=payload.ongoing_monthly,
        ),
    )
    return ok(result)


@router.post("/build-vs-buy", response_model=Envelope[ToolRunOut], name="run_roi_build_vs_buy")
async def run_roi_build_vs_buy(
    db: Db, identity: RunIdentity, payload: RoiBuildVsBuyIn
) -> dict[str, Any]:
    result = await tool_service.run_tool(
        db,
        slug="roi-build-vs-buy",
        workflow=WORKFLOW,
        payload=payload,
        identity=identity,
        compute=lambda: roi_service.roi_build_vs_buy(
            build_hours=payload.build_hours,
            blended_hourly_rate=payload.blended_hourly_rate,
            build_infra_monthly=payload.build_infra_monthly,
            maintenance_pct_of_build_annual=payload.maintenance_pct_of_build_annual,
            vendor_monthly=payload.vendor_monthly,
            vendor_integration_hours=payload.vendor_integration_hours,
            vendor_escalation_pct_annual=payload.vendor_escalation_pct_annual,
            build_months_to_value=payload.build_months_to_value,
            buy_months_to_value=payload.buy_months_to_value,
        ),
    )
    return ok(result)
