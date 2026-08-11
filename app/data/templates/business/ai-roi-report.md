---
title: AI ROI Report
category: business
difficulty: intermediate
summary: >
  A post-launch report on what an AI investment actually returned, structured so
  the honest version is still a good one to present.
use_cases: [automation]
tags: [roi, reporting, measurement, finance]
related_tools: [model-roi, hours-saved, budget-estimator]
premium: true
---

The report nobody writes, which is why the next proposal is argued from
anecdote. Its job is to compare what was projected against what happened, and
to be worth reading when the answer is "less than we said".

## The template

    # [PROJECT] — ROI review, [PERIOD]

    ## Headline

    Projected payback was [N] months. Actual is **[N] months**.
    [ONE SENTENCE: ahead, behind, or on track, and the single largest reason.]

    ## Projected against actual

    | | Projected | Actual | Variance |
    | --- | --- | --- | --- |
    | Build cost | [$N] | [$N] | [+/-N%] |
    | Monthly run cost | [$N] | [$N] | [+/-N%] |
    | Monthly saving | [$N] | [$N] | [+/-N%] |
    | Adoption at [N] months | [N]% | [N]% | [+/-N pts] |

    ## Where the variance came from

    **[LARGEST VARIANCE]** — [WHY. A cause, not a restatement of the number.]

    **[SECOND]** — [WHY.]

    ## What we learned about the estimates

    - [WHICH ASSUMPTION WAS WRONG, AND IN WHICH DIRECTION]
    - [WHAT WE WOULD ESTIMATE DIFFERENTLY NEXT TIME]

    ## What is not in these numbers

    - [E.g. maintenance attention, which is real and unbilled]
    - [E.g. quality effects we can see but cannot price]

    ## Recommendation

    [CONTINUE / EXPAND / CHANGE / STOP], because [REASON TIED TO A NUMBER
    ABOVE].

## The sections that matter

**Variance with a cause.** "Run cost was 40% over" is a fact. "Run cost was 40%
over because retries were not counted in the original estimate" is a lesson,
and the second one improves the next forecast.

**What we learned about the estimates.** This is the compounding part. A team
that knows it habitually underestimates adoption ramp by two months estimates
better forever; a team that only reports outcomes does not.

**What is not in the numbers.** Maintenance attention is the usual omission —
real, unbilled, and invisible in every ROI model including this one. Naming it
protects the credibility of everything else in the report.

**A recommendation that could be "stop".** A report that can only conclude
"continue" is not a measurement, and everyone reading it knows.

## Getting the numbers

Re-run Model ROI with the actual figures and compare against the run you saved
at proposal time. The assumptions come back with the result, which is what makes
the variance table fillable rather than reconstructed from memory.
