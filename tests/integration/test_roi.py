"""WF5 endpoints through the shared engine.

The arithmetic is unit-tested; what matters here is that these four tools get
the same treatment as every other tool - quota, run logging, the seven-key
envelope - without the endpoint having to remember any of it.
"""

from __future__ import annotations

from decimal import Decimal

import pytest
from httpx import AsyncClient

pytestmark = pytest.mark.usefixtures("seeded_catalog")

HOURS = "/api/v1/tools/roi/hours-saved"
MODEL_ROI = "/api/v1/tools/roi/model-roi"
IMPL = "/api/v1/tools/roi/implementation-cost"
BUILD_BUY = "/api/v1/tools/roi/build-vs-buy"


async def test_hours_saved_end_to_end(client: AsyncClient) -> None:
    response = await client.post(
        HOURS,
        json={
            "affected_users": 5,
            "hours_saved_per_user_per_week": "2",
            "fully_loaded_hourly_cost": "100",
        },
    )
    assert response.status_code == 200

    data = response.json()["data"]
    assert data["metrics"]["monthly_value"] == "4348.21"
    assert data["source"] == "rule_based"


async def test_model_roi_returns_payback_roi_and_npv(client: AsyncClient) -> None:
    response = await client.post(
        MODEL_ROI,
        json={
            "current_monthly_cost": "10000",
            "ai_monthly_cost": "2000",
            "implementation_cost": "40000",
            "adoption_ramp_months": 6,
        },
    )
    assert response.status_code == 200

    metrics = response.json()["data"]["metrics"]
    assert metrics["payback_months"] == 9
    assert "roi_12m_pct" in metrics
    assert "npv" in metrics


async def test_the_cash_flow_series_shows_the_crossover(client: AsyncClient) -> None:
    response = await client.post(
        MODEL_ROI,
        json={
            "current_monthly_cost": "10000",
            "ai_monthly_cost": "2000",
            "implementation_cost": "40000",
            "adoption_ramp_months": 6,
            "horizon_months": 12,
        },
    )
    series = response.json()["data"]["series"]["cash_flow"]

    assert len(series) == 12
    # Cumulative starts negative and ends positive: that sign change is the
    # crossover the chart exists to show.
    assert Decimal(series[0]["cumulative"]) < 0
    assert Decimal(series[-1]["cumulative"]) > 0


async def test_a_ramp_of_zero_is_rejected_rather_than_coerced(client: AsyncClient) -> None:
    response = await client.post(
        MODEL_ROI,
        json={
            "current_monthly_cost": "10000",
            "ai_monthly_cost": "2000",
            "implementation_cost": "40000",
            "adoption_ramp_months": 0,
        },
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


async def test_implementation_cost_end_to_end(client: AsyncClient) -> None:
    response = await client.post(
        IMPL,
        json={
            "roles": [{"name": "Engineer", "hours": 400, "hourly_rate": "100"}],
            "duration_months": "4",
            "contingency_pct": "20",
        },
    )
    assert response.status_code == 200
    assert response.json()["data"]["metrics"]["total_cost"] == "48000.00"


async def test_build_vs_buy_end_to_end(client: AsyncClient) -> None:
    response = await client.post(
        BUILD_BUY,
        json={
            "build_hours": 600,
            "blended_hourly_rate": "120",
            "build_infra_monthly": "300",
            "maintenance_pct_of_build_annual": "15",
            "vendor_monthly": "2600",
            "vendor_integration_hours": 80,
            "vendor_escalation_pct_annual": "15",
        },
    )
    assert response.status_code == 200

    data = response.json()["data"]
    assert data["metrics"]["recommendation"] == "build"
    assert len(data["tables"]["tco"]) == 3
    assert len(data["series"]["crossover"]) == 36


async def test_roi_tools_claim_no_provenance_they_do_not_have(client: AsyncClient) -> None:
    """These read no catalog rows, so there is no verification date to show.

    An empty provenance block is the honest answer. Rendering a freshness chip
    over arithmetic on the user's own numbers would be theatre.
    """
    response = await client.post(
        HOURS,
        json={
            "affected_users": 3,
            "hours_saved_per_user_per_week": "1",
            "fully_loaded_hourly_cost": "80",
        },
    )
    assert response.json()["data"]["provenance"]["sources"] == []
