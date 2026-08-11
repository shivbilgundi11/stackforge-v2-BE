---
title: Build vs Buy Comparison
category: business
difficulty: intermediate
summary: >
  A decision framework that counts the costs both sides usually omit —
  maintenance on the build side, lock-in and per-seat growth on the buy side.
use_cases: [automation]
tags: [build-vs-buy, decision, vendor, tco]
related_tools: [build-vs-buy, implementation-cost, model-roi]
---

Build-versus-buy arguments are usually won by whoever counted fewer costs.
This structure forces both sides to count the same things.

## The template

    # [CAPABILITY] — build or buy

    ## What we need it to do

    [THREE BULLETS. Capabilities, not features. If a vendor cannot do one of
    these, it is out — that is a hard constraint, not a scoring dimension.]

    ## Build

    | | |
    | --- | --- |
    | Initial | [N] weeks × [N] people = [$N] |
    | Maintenance | [N]% of one engineer, ongoing = [$N]/year |
    | Infrastructure | [$N]/month |
    | Time to first value | [N] weeks |
    | **Three-year total** | **[$N]** |

    ## Buy

    | | |
    | --- | --- |
    | Licence | [$N]/month at [N] seats |
    | Growth | [N] seats/year at [$N] = [$N]/year by year three |
    | Integration | [N] weeks × [N] people = [$N] |
    | Time to first value | [N] weeks |
    | **Three-year total** | **[$N]** |

    ## What each side gives up

    **Building** costs [N] weeks of the team not doing [THE THING YOU ACTUALLY
    SELL], and it commits you to maintaining it after everyone who built it has
    moved on.

    **Buying** puts [WHAT] outside your control, on their roadmap and their
    pricing. Exit cost is [$N] and [N] weeks.

    ## The deciding question

    Is [CAPABILITY] something customers choose us for?

    If yes, build it — differentiators do not get outsourced.
    If no, buy it, and spend the [N] weeks on something that is.

    ## Recommendation

    [BUILD / BUY], because [THE DECIDING QUESTION'S ANSWER], and the cost
    difference of [$N] over three years [DOES / DOES NOT] change that.

## The two costs that get omitted

**Maintenance, on the build side.** Twenty percent of an engineer forever is
the usual figure and it is almost never in the first version of the comparison.
Over three years it frequently exceeds the initial build.

**Seat growth, on the buy side.** Per-seat pricing at today's headcount is not
the number to compare against a three-year build. Model the headcount you
expect, not the one you have.

## Why the deciding question comes last

Because the arithmetic almost never settles it. The totals usually land within
noise of each other, and the decision is made on whether the capability is
something you differentiate on. Doing the numbers first and the question last
stops the question being answered by whoever preferred one answer.

Run the Build vs Buy tool for the three-year totals — it counts maintenance and
seat growth by default, which is most of the argument.
