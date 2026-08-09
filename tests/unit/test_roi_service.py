"""ROI arithmetic, asserted against hand-computed values.

Every expected number below was worked out independently of the
implementation. A test that recomputes the function's own formula proves the
formula is stable, not that it is right, and for a workflow whose output goes
in front of a CFO that distinction is the entire point.
"""

from __future__ import annotations

from decimal import Decimal

from app.services.roi_service import (
    WEEKS_PER_MONTH,
    adoption_curve,
    fully_loaded_rate,
    hours_saved,
    implementation_cost,
    model_roi,
    npv,
    payback_month,
    roi_build_vs_buy,
)

# ── hours-saved ──────────────────────────────────────────────────────────────


def test_hours_saved_prices_reclaimed_time_at_the_loaded_rate() -> None:
    # 5 users x 2 h/week = 10 h/week.
    # A month is 365.25/12/7 = 4.348214285714... weeks (WF1's month, divided
    # by 7), so 43.48214... h/month at $100 = $4,348.21.
    #
    # M14's worked example says $4,330, which comes from a 4.33-week month
    # (52/12, a 364-day year). Deviating deliberately: the cost tool that
    # feeds this one uses 30.4375 days, and a month cannot be two lengths.
    result = hours_saved(
        affected_users=5,
        hours_saved_per_user_per_week=Decimal(2),
        fully_loaded_hourly_cost=Decimal(100),
    )

    assert result.metrics["monthly_hours"] == Decimal("43.48")
    assert result.metrics["monthly_value"] == Decimal("4348.21")
    assert result.metrics["annual_value"] == Decimal("52178.57")


def test_a_months_worth_of_weeks_matches_the_cost_planners_month() -> None:
    """The two workflows must agree, because WF1 output feeds WF5."""
    from app.services.cost_service import DAYS_PER_MONTH

    assert WEEKS_PER_MONTH * Decimal(7) == DAYS_PER_MONTH


def test_partial_adoption_scales_the_value_linearly() -> None:
    full = hours_saved(
        affected_users=5,
        hours_saved_per_user_per_week=Decimal(2),
        fully_loaded_hourly_cost=Decimal(100),
    )
    partial = hours_saved(
        affected_users=5,
        hours_saved_per_user_per_week=Decimal(2),
        fully_loaded_hourly_cost=Decimal(100),
        adoption_rate_pct=Decimal(60),
    )

    # 60% of $4,348.214... = $2,608.93.
    assert partial.metrics["monthly_value"] == Decimal("2608.93")
    assert Decimal(str(partial.metrics["monthly_value"])) < Decimal(
        str(full.metrics["monthly_value"])
    )


def test_fte_equivalent_is_against_a_forty_hour_week() -> None:
    result = hours_saved(
        affected_users=20,
        hours_saved_per_user_per_week=Decimal(2),
        fully_loaded_hourly_cost=Decimal(100),
    )
    # 40 h/week reclaimed is exactly one FTE.
    assert result.metrics["fte_equivalent"] == Decimal("1.00")


def test_a_base_salary_rate_is_flagged_rather_than_silently_used() -> None:
    result = hours_saved(
        affected_users=5,
        hours_saved_per_user_per_week=Decimal(2),
        fully_loaded_hourly_cost=Decimal(25),
    )
    assert any(w.field == "fully_loaded_hourly_cost" for w in result.warnings)


def test_the_loaded_rate_helper_applies_the_multiplier() -> None:
    assert fully_loaded_rate(base_hourly=Decimal(50)) == Decimal("65.00")
    assert fully_loaded_rate(base_hourly=Decimal(50), multiplier=Decimal("1.25")) == Decimal(
        "62.50"
    )


# ── adoption and discounting ─────────────────────────────────────────────────


def test_the_ramp_reaches_full_adoption_on_its_last_month_and_stays() -> None:
    curve = adoption_curve(months=8, ramp_months=4)
    assert curve[0] == Decimal("0.25")
    assert curve[3] == Decimal(1)
    assert curve[7] == Decimal(1)


def test_a_one_month_ramp_is_full_adoption_throughout() -> None:
    assert adoption_curve(months=3, ramp_months=1) == [Decimal(1)] * 3


def test_npv_at_zero_percent_is_the_undiscounted_sum() -> None:
    flows = [Decimal(100), Decimal(200), Decimal(300)]
    assert npv(flows, annual_discount_rate=Decimal(0)) == Decimal(600)


def test_npv_discounts_the_first_month_by_one_period() -> None:
    # 12% annual = 1% monthly. One flow of 101 discounted once = 100.
    result = npv([Decimal("101")], annual_discount_rate=Decimal(12))
    assert result.quantize(Decimal("0.01")) == Decimal("100.00")


def test_payback_is_none_rather_than_zero_when_it_never_arrives() -> None:
    assert payback_month([Decimal(-10), Decimal(-5), Decimal(-1)]) is None
    assert payback_month([Decimal(-10), Decimal(0)]) == 2


# ── model-roi ────────────────────────────────────────────────────────────────


def test_model_roi_pays_back_later_with_a_ramp_than_without() -> None:
    """The whole reason the ramp is a required input."""
    common = {
        "current_monthly_cost": Decimal(10_000),
        "ai_monthly_cost": Decimal(2_000),
        "implementation_cost": Decimal(40_000),
    }

    instant = model_roi(**common, adoption_ramp_months=1)  # type: ignore[arg-type]
    ramped = model_roi(**common, adoption_ramp_months=6)  # type: ignore[arg-type]

    # Instant: $8,000/mo net against $40,000 → month 5.
    assert instant.metrics["payback_months"] == 5

    # Ramped over 6 months the monthly net is
    # 10000*(m/6) - 2000 → -333.33, 1333.33, 3000, 4666.67, 6333.33, 8000.
    # Cumulative from -40,000 crosses zero in month 9.
    assert ramped.metrics["payback_months"] == 9
    assert ramped.metrics["payback_months"] > instant.metrics["payback_months"]


def test_an_instant_ramp_is_warned_about() -> None:
    result = model_roi(
        current_monthly_cost=Decimal(10_000),
        ai_monthly_cost=Decimal(2_000),
        implementation_cost=Decimal(40_000),
        adoption_ramp_months=1,
    )
    assert any(w.field == "adoption_ramp_months" for w in result.warnings)


def test_model_roi_npv_at_zero_discount_equals_the_undiscounted_net() -> None:
    result = model_roi(
        current_monthly_cost=Decimal(10_000),
        ai_monthly_cost=Decimal(2_000),
        implementation_cost=Decimal(40_000),
        adoption_ramp_months=1,
        horizon_months=12,
        discount_rate_pct=Decimal(0),
    )
    # 12 months x $8,000 net, less $40,000 implementation = $56,000.
    assert result.metrics["npv"] == Decimal("56000.00")
    assert result.metrics["year_one_net"] == Decimal("96000.00")


def test_twelve_month_roi_is_against_the_implementation_cost() -> None:
    result = model_roi(
        current_monthly_cost=Decimal(10_000),
        ai_monthly_cost=Decimal(2_000),
        implementation_cost=Decimal(40_000),
        adoption_ramp_months=1,
        horizon_months=12,
    )
    # (96,000 - 40,000) / 40,000 = 140%.
    assert result.metrics["roi_12m_pct"] == Decimal("140.0")


def test_no_saving_is_reported_as_critical_not_as_a_negative_payback() -> None:
    result = model_roi(
        current_monthly_cost=Decimal(1_000),
        ai_monthly_cost=Decimal(3_000),
        implementation_cost=Decimal(10_000),
        adoption_ramp_months=3,
    )
    assert result.metrics["payback_months"] == "never"
    assert any(w.level == "critical" for w in result.warnings)


def test_the_business_case_artifact_states_its_assumptions() -> None:
    result = model_roi(
        current_monthly_cost=Decimal(10_000),
        ai_monthly_cost=Decimal(2_000),
        implementation_cost=Decimal(40_000),
        adoption_ramp_months=6,
    )
    artifact = result.artifacts[0]
    assert artifact.format == "markdown"
    assert "Stated assumptions" in artifact.content
    # A case whose assumptions are invisible gets discarded in the meeting.
    assert "6 months" in artifact.content
    assert "$10,000.00" in artifact.content

    assumptions = {row["assumption"] for row in result.tables["assumptions"]}
    assert "Adoption ramp" in assumptions
    assert "Discount rate" in assumptions


# ── implementation-cost ──────────────────────────────────────────────────────


def test_contingency_adds_exactly_its_percentage_to_the_subtotal() -> None:
    result = implementation_cost(
        roles=[{"name": "Engineer", "hours": 400, "hourly_rate": Decimal(100)}],
        duration_months=Decimal(4),
        contingency_pct=Decimal(20),
    )
    # 400 x $100 = $40,000 subtotal; 20% = $8,000; total $48,000.
    assert result.metrics["labour_cost"] == Decimal("40000.00")
    assert result.metrics["contingency"] == Decimal("8000.00")
    assert result.metrics["total_cost"] == Decimal("48000.00")


def test_every_cost_line_reaches_the_subtotal() -> None:
    result = implementation_cost(
        roles=[{"name": "Engineer", "hours": 100, "hourly_rate": Decimal(100)}],
        duration_months=Decimal(2),
        infrastructure_setup=Decimal(5_000),
        licences=Decimal(2_000),
        training=Decimal(3_000),
        contingency_pct=Decimal(0),
    )
    # 10,000 + 5,000 + 2,000 + 3,000 = 20,000, no contingency.
    assert result.metrics["total_cost"] == Decimal("20000.00")


def test_the_burn_curve_sums_to_the_total() -> None:
    result = implementation_cost(
        roles=[{"name": "Engineer", "hours": 300, "hourly_rate": Decimal(100)}],
        duration_months=Decimal(3),
        contingency_pct=Decimal(0),
    )
    burn = result.series["burn"]
    assert len(burn) == 3
    assert Decimal(burn[-1]["cumulative"]) == Decimal(str(result.metrics["total_cost"]))


def test_thin_contingency_is_flagged() -> None:
    result = implementation_cost(
        roles=[{"name": "Engineer", "hours": 100, "hourly_rate": Decimal(100)}],
        duration_months=Decimal(1),
        contingency_pct=Decimal(5),
    )
    assert any(w.field == "contingency_pct" for w in result.warnings)


# ── roi-build-vs-buy ─────────────────────────────────────────────────────────


def test_vendor_escalation_can_flip_the_thirty_six_month_winner() -> None:
    """The documented case M14 asks for.

    Buy is cheaper at 36 months with a flat subscription, and dearer once the
    contract's annual uplift is applied. A 12-month comparison shows neither.
    """
    common = {
        "build_hours": 600,
        "blended_hourly_rate": Decimal(120),
        "build_infra_monthly": Decimal(300),
        "maintenance_pct_of_build_annual": Decimal(15),
        "vendor_monthly": Decimal(2_600),
        "vendor_integration_hours": 80,
    }

    flat = roi_build_vs_buy(**common, vendor_escalation_pct_annual=Decimal(0))  # type: ignore[arg-type]
    escalating = roi_build_vs_buy(**common, vendor_escalation_pct_annual=Decimal(15))  # type: ignore[arg-type]

    assert flat.metrics["recommendation"] == "buy"
    assert escalating.metrics["recommendation"] == "build"


def test_maintenance_compounds_on_the_build_side() -> None:
    result = roi_build_vs_buy(
        build_hours=1000,
        blended_hourly_rate=Decimal(100),
        build_infra_monthly=Decimal(0),
        maintenance_pct_of_build_annual=Decimal(24),
        vendor_monthly=Decimal(1),
        vendor_integration_hours=0,
    )
    # $100,000 build, 24% annually = $24,000/yr = $2,000/mo.
    assert result.metrics["maintenance_monthly"] == Decimal("2000.00")
    assert result.metrics["build_upfront"] == Decimal("100000.00")


def test_tco_is_reported_at_all_three_horizons() -> None:
    result = roi_build_vs_buy(
        build_hours=500,
        blended_hourly_rate=Decimal(120),
        build_infra_monthly=Decimal(200),
        maintenance_pct_of_build_annual=Decimal(20),
        vendor_monthly=Decimal(2_000),
        vendor_integration_hours=40,
    )
    horizons = [row["horizon"] for row in result.tables["tco"]]
    assert horizons == ["12 months", "24 months", "36 months"]


def test_a_flat_subscription_assumption_is_flagged() -> None:
    result = roi_build_vs_buy(
        build_hours=500,
        blended_hourly_rate=Decimal(120),
        build_infra_monthly=Decimal(0),
        maintenance_pct_of_build_annual=Decimal(20),
        vendor_monthly=Decimal(2_000),
        vendor_integration_hours=0,
        vendor_escalation_pct_annual=Decimal(0),
    )
    assert any(w.field == "vendor_escalation_pct_annual" for w in result.warnings)


def test_sensitivity_covers_the_two_softest_inputs() -> None:
    result = roi_build_vs_buy(
        build_hours=500,
        blended_hourly_rate=Decimal(120),
        build_infra_monthly=Decimal(200),
        maintenance_pct_of_build_annual=Decimal(20),
        vendor_monthly=Decimal(2_000),
        vendor_integration_hours=40,
    )
    scenarios = {row["scenario"] for row in result.tables["sensitivity"]}
    assert "Build takes 50% longer" in scenarios
    assert "Rate 25% higher" in scenarios


def test_delayed_time_to_value_is_surfaced_even_though_it_is_not_priced() -> None:
    result = roi_build_vs_buy(
        build_hours=500,
        blended_hourly_rate=Decimal(120),
        build_infra_monthly=Decimal(0),
        maintenance_pct_of_build_annual=Decimal(20),
        vendor_monthly=Decimal(2_000),
        vendor_integration_hours=0,
        build_months_to_value=9,
        buy_months_to_value=1,
    )
    assert any("8 months later" in w.message for w in result.warnings)


# ── precision ────────────────────────────────────────────────────────────────


def test_fractional_cent_inputs_do_not_round_to_zero_through_the_chain() -> None:
    """D-08 all the way through WF5."""
    result = hours_saved(
        affected_users=1,
        hours_saved_per_user_per_week=Decimal("0.01"),
        fully_loaded_hourly_cost=Decimal("0.05"),
    )
    # 0.01 h/week x 4.348214... x $0.05 = $0.00217... which floors to $0.00 at
    # cent precision, but the hours must survive.
    assert Decimal(str(result.metrics["monthly_hours"])) > 0
