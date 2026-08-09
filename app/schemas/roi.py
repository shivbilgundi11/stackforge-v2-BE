"""ROI request shapes (WF5).

`adoption_ramp_months` has no default on `model-roi`. Every other optional
number here defaults to something harmless, but a ramp defaulting to 1 would
silently produce the instant-adoption case that M14 exists to stop people
publishing.
"""

from __future__ import annotations

from decimal import Decimal

from pydantic import BaseModel, Field


class HoursSavedIn(BaseModel):
    affected_users: int = Field(ge=1, le=1_000_000)
    hours_saved_per_user_per_week: Decimal = Field(gt=0, le=168)
    fully_loaded_hourly_cost: Decimal = Field(gt=0, le=10_000)
    adoption_rate_pct: Decimal = Field(default=Decimal(100), ge=1, le=100)
    error_rate_reduction_pct: Decimal = Field(default=Decimal(0), ge=0, le=100)
    rework_hours_per_month: Decimal = Field(default=Decimal(0), ge=0, le=100_000)


class ModelRoiIn(BaseModel):
    current_monthly_cost: Decimal = Field(ge=0, le=100_000_000)
    ai_monthly_cost: Decimal = Field(ge=0, le=100_000_000)
    implementation_cost: Decimal = Field(ge=0, le=100_000_000)
    adoption_ramp_months: int = Field(
        ge=1, le=36, description="Months to reach full adoption. 1 means instant."
    )
    horizon_months: int = Field(default=36, ge=6, le=120)
    discount_rate_pct: Decimal = Field(default=Decimal(10), ge=0, le=50)


class RoleCostIn(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    hours: int = Field(ge=0, le=200_000)
    hourly_rate: Decimal = Field(gt=0, le=10_000)


class ImplementationCostIn(BaseModel):
    roles: list[RoleCostIn] = Field(min_length=1, max_length=15)
    duration_months: Decimal = Field(default=Decimal(3), ge=1, le=60)
    infrastructure_setup: Decimal = Field(default=Decimal(0), ge=0)
    licences: Decimal = Field(default=Decimal(0), ge=0)
    training: Decimal = Field(default=Decimal(0), ge=0)
    contingency_pct: Decimal = Field(default=Decimal(15), ge=0, le=100)
    ongoing_monthly: Decimal = Field(default=Decimal(0), ge=0)


class RoiBuildVsBuyIn(BaseModel):
    build_hours: int = Field(ge=1, le=200_000)
    blended_hourly_rate: Decimal = Field(gt=0, le=10_000)
    build_infra_monthly: Decimal = Field(default=Decimal(0), ge=0)
    maintenance_pct_of_build_annual: Decimal = Field(
        default=Decimal(20),
        ge=0,
        le=200,
        description="Annual maintenance as a percentage of the original build cost.",
    )
    vendor_monthly: Decimal = Field(ge=0, le=10_000_000)
    vendor_integration_hours: int = Field(default=0, ge=0, le=50_000)
    vendor_escalation_pct_annual: Decimal = Field(default=Decimal(0), ge=0, le=50)
    build_months_to_value: int = Field(default=6, ge=0, le=60)
    buy_months_to_value: int = Field(default=1, ge=0, le=60)
