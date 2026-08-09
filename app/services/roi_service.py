"""ROI and business-case arithmetic (WF5). Four pure functions.

Every figure here is `Decimal`. This is the one workflow whose output leaves
the building - it goes into a deck, in front of someone whose job is to find
the flaw in it - so the arithmetic has to survive a reviewer with a
spreadsheet.

Two modelling choices carry most of the credibility:

**Adoption is ramped, not instant.** A model deployed in month 1 is not at
full utilisation until month 4-6. Assuming otherwise overstates year-one
return by a wide margin, and it is the specific flaw that teaches finance
reviewers to discount AI business cases on sight. The ramp is a required
input, not an option.

**Savings are discounted.** A 36-month total with no discount rate is not a
number a CFO recognises. NPV costs one extra input and is the difference
between a case that gets argued with and one that gets dismissed.
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Final

from app.schemas.tools import Artifact, ToolOutput, ToolWarning
from app.services.cost_service import DAYS_PER_MONTH

# Derived from WF1's month, not restated. WF1 costs flow into this workflow by
# handoff, so two different month lengths would make the two disagree by a
# percent for no reason - and typing the quotient by hand is how that happens
# (365.25/12/7 is 4.34821..., which is easy to fat-finger as 4.348125).
#
# This is a deliberate deviation from M14's worked example, which uses the
# conventional 4.33 and expects $4,330 from 5 users x 2h x $100. 4.33 is
# 52/12, a 364-day year. Adopting it would make a month in the ROI tool three
# days shorter than a month in the cost tool that feeds it, which is a worse
# error than disagreeing with the spec's rounded constant.
WEEKS_PER_MONTH: Final = DAYS_PER_MONTH / Decimal(7)
MONTHS_PER_YEAR: Final = Decimal(12)

CENTS: Final = Decimal("0.01")
PERCENT: Final = Decimal("0.1")


def _money(value: Decimal) -> Decimal:
    return value.quantize(CENTS, rounding=ROUND_HALF_UP)


def _pct(value: Decimal) -> Decimal:
    return value.quantize(PERCENT, rounding=ROUND_HALF_UP)


def _usd(value: Decimal) -> str:
    """Grouped, with the sign outside the symbol: `-$414.27`, not `$-414.27`."""
    amount = f"{abs(value):,.2f}"
    return f"-${amount}" if value < 0 else f"${amount}"


def adoption_curve(months: int, ramp_months: int) -> list[Decimal]:
    """Fraction of full benefit realised in each month, 1-indexed.

    Linear to full adoption over `ramp_months`, then flat. Month 1 of a
    6-month ramp is at 1/6, not 0: a rollout that delivers literally nothing
    in its first month describes a project that has not started, and users
    setting a ramp mean "it builds up", not "it is switched off".
    """
    if ramp_months <= 1:
        return [Decimal(1) for _ in range(months)]

    curve: list[Decimal] = []
    for month in range(1, months + 1):
        share = Decimal(min(month, ramp_months)) / Decimal(ramp_months)
        curve.append(share)
    return curve


def npv(cash_flows: list[Decimal], *, annual_discount_rate: Decimal) -> Decimal:
    """Net present value of monthly cash flows, discounted monthly.

    `cash_flows[0]` is month 1 and is discounted one period. Treating the
    first month as t=0 would quietly inflate every result by one month of
    discounting.
    """
    if annual_discount_rate == 0:
        return sum(cash_flows, Decimal(0))

    monthly_rate = annual_discount_rate / Decimal(100) / MONTHS_PER_YEAR
    total = Decimal(0)
    for period, flow in enumerate(cash_flows, start=1):
        total += flow / ((Decimal(1) + monthly_rate) ** period)
    return total


def payback_month(cumulative: list[Decimal]) -> int | None:
    """First 1-indexed month where cumulative cash flow turns non-negative.

    `None` rather than 0 or -1 when it never does, so the caller has to decide
    what to render. "Payback: 0 months" for a project that never pays back is
    the kind of output that destroys trust in everything beside it.
    """
    for index, value in enumerate(cumulative, start=1):
        if value >= 0:
            return index
    return None


# ── hours-saved ──────────────────────────────────────────────────────────────


def hours_saved(
    *,
    affected_users: int,
    hours_saved_per_user_per_week: Decimal,
    fully_loaded_hourly_cost: Decimal,
    adoption_rate_pct: Decimal = Decimal(100),
    error_rate_reduction_pct: Decimal = Decimal(0),
    rework_hours_per_month: Decimal = Decimal(0),
) -> ToolOutput:
    """Time reclaimed, priced at fully-loaded cost.

    Fully-loaded means salary plus benefits, overhead, and equipment -
    typically 1.25-1.4x base. A case built on the base salary rate understates
    its own value and is trivially dismissed by anyone who knows the
    difference, which in a room where this gets presented is everyone.
    """
    adoption = adoption_rate_pct / Decimal(100)

    weekly_hours = Decimal(affected_users) * hours_saved_per_user_per_week * adoption
    monthly_hours = weekly_hours * WEEKS_PER_MONTH
    annual_hours = monthly_hours * MONTHS_PER_YEAR

    monthly_value = _money(monthly_hours * fully_loaded_hourly_cost)
    annual_value = _money(annual_hours * fully_loaded_hourly_cost)

    # 1 FTE is 40h/week. Expressed because "0.8 FTE" lands with a reader in a
    # way that "139 hours a month" does not.
    fte_equivalent = (weekly_hours / Decimal(40)).quantize(Decimal("0.01"))

    rework_hours_saved = rework_hours_per_month * (error_rate_reduction_pct / Decimal(100))
    rework_value = _money(rework_hours_saved * fully_loaded_hourly_cost)

    warnings: list[ToolWarning] = []
    if adoption_rate_pct >= 95:
        warnings.append(
            ToolWarning(
                level="warning",
                field="adoption_rate_pct",
                message=(
                    "Near-universal adoption is rare in year one. A reviewer will "
                    "discount this figure unless the rollout is mandatory and enforced."
                ),
            )
        )
    if fully_loaded_hourly_cost < 40:
        warnings.append(
            ToolWarning(
                level="info",
                field="fully_loaded_hourly_cost",
                message=(
                    "That reads like a base salary rate. Fully-loaded cost is "
                    "usually 1.25-1.4x base once benefits and overhead are counted."
                ),
            )
        )

    breakdown = [
        {
            "line": "Time reclaimed",
            "hours_per_month": str(monthly_hours.quantize(CENTS)),
            "value": _usd(monthly_value),
        },
        {
            "line": "Rework avoided",
            "hours_per_month": str(rework_hours_saved.quantize(CENTS)),
            "value": _usd(rework_value),
        },
        {
            "line": "Total",
            "hours_per_month": str((monthly_hours + rework_hours_saved).quantize(CENTS)),
            "value": _usd(monthly_value + rework_value),
        },
    ]

    return ToolOutput(
        metrics={
            "monthly_hours": monthly_hours.quantize(CENTS),
            "annual_hours": annual_hours.quantize(CENTS),
            "monthly_value": monthly_value,
            "annual_value": annual_value,
            "fte_equivalent": fte_equivalent,
            "rework_value_monthly": rework_value,
            "total_monthly_value": _money(monthly_value + rework_value),
        },
        tables={"breakdown": breakdown},
        warnings=warnings,
    )


def fully_loaded_rate(*, base_hourly: Decimal, multiplier: Decimal = Decimal("1.3")) -> Decimal:
    """Helper for the UI: most people know the salary, not the loaded figure."""
    return _money(base_hourly * multiplier)


# ── model-roi ────────────────────────────────────────────────────────────────


def model_roi(
    *,
    current_monthly_cost: Decimal,
    ai_monthly_cost: Decimal,
    implementation_cost: Decimal,
    adoption_ramp_months: int,
    horizon_months: int = 36,
    discount_rate_pct: Decimal = Decimal(10),
) -> ToolOutput:
    """Payback, ROI, and NPV against a ramped adoption curve.

    The saving in month N is the *realised* saving: the gross monthly saving
    scaled by how much of the benefit is live that month. The AI running cost
    is not scaled, because you pay for the platform from day one whether or
    not the organisation has finished adopting it. Scaling both would hide the
    ramp entirely, which is the thing this tool exists to show.
    """
    gross_monthly_saving = current_monthly_cost - ai_monthly_cost
    curve = adoption_curve(horizon_months, adoption_ramp_months)

    flows: list[Decimal] = []
    cumulative: list[Decimal] = []
    running = -implementation_cost
    projection: list[dict[str, Any]] = []

    for month, share in enumerate(curve, start=1):
        realised = current_monthly_cost * share
        net = realised - ai_monthly_cost
        flows.append(net)
        running += net
        cumulative.append(running)
        projection.append(
            {
                "month": month,
                "net_saving": str(_money(net)),
                "cumulative": str(_money(running)),
                "adoption_pct": str(_pct(share * Decimal(100))),
            }
        )

    payback = payback_month(cumulative)
    year_one_net = sum(flows[:12], Decimal(0))
    roi_12m = (
        _pct((year_one_net - implementation_cost) / implementation_cost * Decimal(100))
        if implementation_cost > 0
        else Decimal(0)
    )
    discounted = npv(flows, annual_discount_rate=discount_rate_pct) - implementation_cost

    warnings: list[ToolWarning] = []
    if gross_monthly_saving <= 0:
        warnings.append(
            ToolWarning(
                level="critical",
                message=(
                    "The AI running cost is at or above the current process cost, so "
                    "there is no saving to pay back the implementation. The case has "
                    "to rest on something other than cost."
                ),
            )
        )
    if payback is None and gross_monthly_saving > 0:
        warnings.append(
            ToolWarning(
                level="warning",
                message=(
                    f"No payback within {horizon_months} months at this ramp. "
                    "Shortening the ramp or reducing implementation cost is what moves it."
                ),
            )
        )
    if adoption_ramp_months <= 1:
        warnings.append(
            ToolWarning(
                level="warning",
                field="adoption_ramp_months",
                message=(
                    "Instant adoption is the most common flaw in an AI business case. "
                    "Four to six months is the usual reality, and stating it makes the "
                    "rest of the case more credible, not less."
                ),
            )
        )

    return ToolOutput(
        metrics={
            "monthly_saving_at_full_adoption": _money(gross_monthly_saving),
            "year_one_net": _money(year_one_net),
            "implementation_cost": _money(implementation_cost),
            "payback_months": payback if payback is not None else "never",
            "roi_12m_pct": roi_12m,
            "npv": _money(discounted),
            "discount_rate_pct": _pct(discount_rate_pct),
        },
        series={"cash_flow": projection},
        tables={
            "assumptions": _assumptions(
                {
                    "Current process cost": _usd(current_monthly_cost) + "/mo",
                    "AI running cost": _usd(ai_monthly_cost) + "/mo",
                    "Implementation cost": _usd(implementation_cost),
                    "Adoption ramp": f"{adoption_ramp_months} months to full",
                    "Horizon": f"{horizon_months} months",
                    "Discount rate": f"{_pct(discount_rate_pct)}% annual",
                }
            )
        },
        artifacts=[
            _business_case(
                payback=payback,
                roi_12m=roi_12m,
                discounted=discounted,
                year_one_net=year_one_net,
                implementation_cost=implementation_cost,
                current_monthly_cost=current_monthly_cost,
                ai_monthly_cost=ai_monthly_cost,
                adoption_ramp_months=adoption_ramp_months,
                horizon_months=horizon_months,
                discount_rate_pct=discount_rate_pct,
            )
        ],
        warnings=warnings,
    )


def _assumptions(values: dict[str, str]) -> list[dict[str, Any]]:
    """Every ROI result carries its inputs back out.

    A business case whose assumptions are invisible gets challenged and then
    discarded. Shipping them in the result is what makes the output usable in
    a meeting rather than just correct.
    """
    return [{"assumption": key, "value": value} for key, value in values.items()]


def _business_case(
    *,
    payback: int | None,
    roi_12m: Decimal,
    discounted: Decimal,
    year_one_net: Decimal,
    implementation_cost: Decimal,
    current_monthly_cost: Decimal,
    ai_monthly_cost: Decimal,
    adoption_ramp_months: int,
    horizon_months: int,
    discount_rate_pct: Decimal,
) -> Artifact:
    payback_text = f"{payback} months" if payback else f"not within {horizon_months} months"

    body = f"""# AI investment business case

## Summary

| Measure | Value |
| --- | --- |
| Payback period | {payback_text} |
| 12-month ROI | {_pct(roi_12m)}% |
| NPV over {horizon_months} months | {_usd(_money(discounted))} |
| Year-one net saving | {_usd(_money(year_one_net))} |

## Stated assumptions

- Current process cost: **{_usd(current_monthly_cost)} per month**
- AI running cost: **{_usd(ai_monthly_cost)} per month**, incurred in full from month 1
- One-off implementation cost: **{_usd(implementation_cost)}**
- Adoption ramps linearly to full over **{adoption_ramp_months} months**
- Cash flows discounted at **{_pct(discount_rate_pct)}% annually**, applied monthly

The saving is scaled by adoption each month; the running cost is not, because
the platform is paid for from day one whether or not rollout has finished.

## How to read this

Payback is the month cumulative cash flow first turns positive, counting the
implementation cost as month-zero outflow. NPV discounts every monthly flow
back to today, so it is lower than the undiscounted total by design.
"""
    return Artifact(
        type="business-case",
        format="markdown",
        filename="business-case.md",
        content=body,
    )


# ── implementation-cost ──────────────────────────────────────────────────────


def implementation_cost(
    *,
    roles: list[dict[str, Any]],
    duration_months: Decimal,
    infrastructure_setup: Decimal = Decimal(0),
    licences: Decimal = Decimal(0),
    training: Decimal = Decimal(0),
    contingency_pct: Decimal = Decimal(15),
    ongoing_monthly: Decimal = Decimal(0),
) -> ToolOutput:
    """Effort, infrastructure, and the contingency that always gets used.

    `roles` is `[{name, hours, hourly_rate}]`. Labour is the dominant line in
    almost every case here, which is why it is entered per role rather than as
    one blended number - a reviewer will want to see the mix.
    """
    labour_rows: list[dict[str, Any]] = []
    labour_total = Decimal(0)

    for role in roles:
        hours = Decimal(str(role["hours"]))
        rate = Decimal(str(role["hourly_rate"]))
        cost = hours * rate
        labour_total += cost
        labour_rows.append(
            {
                "role": role["name"],
                "hours": int(hours),
                "hourly_rate": _usd(rate),
                "cost": _usd(_money(cost)),
            }
        )

    subtotal = labour_total + infrastructure_setup + licences + training
    contingency = subtotal * (contingency_pct / Decimal(100))
    total = subtotal + contingency

    phases = [
        {"phase": "Labour", "cost": _usd(_money(labour_total))},
        {"phase": "Infrastructure setup", "cost": _usd(_money(infrastructure_setup))},
        {"phase": "Licences", "cost": _usd(_money(licences))},
        {"phase": "Training", "cost": _usd(_money(training))},
        {"phase": "Subtotal", "cost": _usd(_money(subtotal))},
        {"phase": f"Contingency ({_pct(contingency_pct)}%)", "cost": _usd(_money(contingency))},
        {"phase": "Total", "cost": _usd(_money(total))},
    ]

    # Straight-line burn. Uneven phasing is a planning exercise this tool does
    # not have the inputs for, and a fabricated S-curve would look more precise
    # than it is.
    months = max(1, int(duration_months))
    monthly_burn = total / Decimal(months)
    burn = [
        {
            "month": month,
            "spend": str(_money(monthly_burn)),
            "cumulative": str(_money(monthly_burn * Decimal(month))),
        }
        for month in range(1, months + 1)
    ]

    warnings: list[ToolWarning] = []
    if contingency_pct < 10:
        warnings.append(
            ToolWarning(
                level="warning",
                field="contingency_pct",
                message=(
                    "Under 10% contingency on an AI project is optimistic. Integration "
                    "and data work are where these overrun."
                ),
            )
        )

    return ToolOutput(
        metrics={
            "total_cost": _money(total),
            "labour_cost": _money(labour_total),
            "contingency": _money(contingency),
            "monthly_burn": _money(monthly_burn),
            "ongoing_monthly": _money(ongoing_monthly),
            "duration_months": months,
        },
        tables={"phases": phases, "roles": labour_rows},
        series={"burn": burn},
        warnings=warnings,
    )


# ── roi-build-vs-buy ─────────────────────────────────────────────────────────


def roi_build_vs_buy(
    *,
    build_hours: int,
    blended_hourly_rate: Decimal,
    build_infra_monthly: Decimal,
    maintenance_pct_of_build_annual: Decimal,
    vendor_monthly: Decimal,
    vendor_integration_hours: int,
    vendor_escalation_pct_annual: Decimal = Decimal(0),
    build_months_to_value: int = 6,
    buy_months_to_value: int = 1,
    horizons: tuple[int, ...] = (12, 24, 36),
) -> ToolOutput:
    """Full TCO with maintenance drag, vendor escalation, and delayed value.

    Distinct from M10's `compare-build-vs-buy`, which is a fast scored
    comparison. This is the financial model: maintenance compounds on the
    build side, the subscription escalates on the buy side, and the two
    crossing over at 26 months is exactly the kind of finding a 12-month
    comparison hides.
    """
    build_upfront = Decimal(build_hours) * blended_hourly_rate
    buy_upfront = Decimal(vendor_integration_hours) * blended_hourly_rate
    maintenance_monthly = (
        build_upfront * (maintenance_pct_of_build_annual / Decimal(100)) / MONTHS_PER_YEAR
    )

    max_horizon = max(horizons)
    build_cum: list[Decimal] = []
    buy_cum: list[Decimal] = []
    build_running = build_upfront
    buy_running = buy_upfront
    crossover: list[dict[str, Any]] = []

    for month in range(1, max_horizon + 1):
        build_running += maintenance_monthly + build_infra_monthly
        # Escalation applies per completed year, so month 13 is the first at
        # the new rate.
        years_elapsed = (month - 1) // 12
        escalated = vendor_monthly * (
            (Decimal(1) + vendor_escalation_pct_annual / Decimal(100)) ** years_elapsed
        )
        buy_running += escalated

        build_cum.append(build_running)
        buy_cum.append(buy_running)
        crossover.append(
            {
                "month": month,
                "build": str(_money(build_running)),
                "buy": str(_money(buy_running)),
            }
        )

    tco_rows = [
        {
            "horizon": f"{h} months",
            "build": _usd(_money(build_cum[h - 1])),
            "buy": _usd(_money(buy_cum[h - 1])),
            "difference": _usd(_money(buy_cum[h - 1] - build_cum[h - 1])),
            "cheaper": "build" if build_cum[h - 1] < buy_cum[h - 1] else "buy",
        }
        for h in horizons
        if h <= max_horizon
    ]

    break_even = next(
        (m for m, (b, v) in enumerate(zip(build_cum, buy_cum, strict=True), start=1) if b < v),
        None,
    )

    final_build, final_buy = build_cum[-1], buy_cum[-1]
    winner = "build" if final_build < final_buy else "buy"

    sensitivity = _sensitivity(
        build_hours=build_hours,
        blended_hourly_rate=blended_hourly_rate,
        maintenance_pct=maintenance_pct_of_build_annual,
        build_infra_monthly=build_infra_monthly,
        vendor_monthly=vendor_monthly,
        vendor_escalation_pct=vendor_escalation_pct_annual,
        buy_upfront=buy_upfront,
        horizon=max_horizon,
    )

    warnings: list[ToolWarning] = []
    delay = build_months_to_value - buy_months_to_value
    if delay > 0:
        warnings.append(
            ToolWarning(
                level="info",
                message=(
                    f"Building reaches value {delay} months later. That delay is a real "
                    "cost this model does not price - if the capability earns or saves "
                    "anything per month, multiply it by "
                    f"{delay} and add it to the build column."
                ),
            )
        )
    if vendor_escalation_pct_annual == 0:
        warnings.append(
            ToolWarning(
                level="warning",
                field="vendor_escalation_pct_annual",
                message=(
                    "Zero price escalation assumes the vendor never raises prices over "
                    f"{max_horizon} months. 3-7% annually is the usual contract term."
                ),
            )
        )

    return ToolOutput(
        metrics={
            "build_upfront": _money(build_upfront),
            "buy_upfront": _money(buy_upfront),
            "maintenance_monthly": _money(maintenance_monthly),
            f"build_tco_{max_horizon}m": _money(final_build),
            f"buy_tco_{max_horizon}m": _money(final_buy),
            "break_even_month": break_even if break_even is not None else "never",
            "recommendation": winner,
        },
        tables={"tco": tco_rows, "sensitivity": sensitivity},
        series={"crossover": crossover},
        warnings=warnings,
    )


def _sensitivity(
    *,
    build_hours: int,
    blended_hourly_rate: Decimal,
    maintenance_pct: Decimal,
    build_infra_monthly: Decimal,
    vendor_monthly: Decimal,
    vendor_escalation_pct: Decimal,
    buy_upfront: Decimal,
    horizon: int,
) -> list[dict[str, Any]]:
    """How the answer moves when the two softest inputs are wrong.

    Build-hour estimates and blended rates are guesses, and the recommendation
    often flips inside their plausible range. A reviewer who can see that
    trusts the analysis more than one handed a single confident answer.
    """
    rows: list[dict[str, Any]] = []

    for label, hours_mult, rate_mult in (
        ("Build takes 50% longer", Decimal("1.5"), Decimal(1)),
        ("Build as estimated", Decimal(1), Decimal(1)),
        ("Rate 25% higher", Decimal(1), Decimal("1.25")),
        ("Build 25% faster", Decimal("0.75"), Decimal(1)),
    ):
        upfront = Decimal(build_hours) * hours_mult * blended_hourly_rate * rate_mult
        maintenance = upfront * (maintenance_pct / Decimal(100)) / MONTHS_PER_YEAR
        build_total = upfront + (maintenance + build_infra_monthly) * Decimal(horizon)

        buy_total = buy_upfront
        for month in range(1, horizon + 1):
            buy_total += vendor_monthly * (
                (Decimal(1) + vendor_escalation_pct / Decimal(100)) ** ((month - 1) // 12)
            )

        rows.append(
            {
                "scenario": label,
                "build": _usd(_money(build_total)),
                "buy": _usd(_money(buy_total)),
                "cheaper": "build" if build_total < buy_total else "buy",
            }
        )

    return rows
