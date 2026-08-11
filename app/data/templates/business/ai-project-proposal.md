---
title: AI Project Proposal
category: business
difficulty: beginner
summary: >
  A one-page proposal that survives contact with a finance review: the problem
  in the current cost, the assumptions visible, and what would make you stop.
use_cases: [automation]
tags: [proposal, business-case, stakeholder]
related_tools: [hours-saved, implementation-cost, model-roi]
premium: true
---

Most AI proposals are rejected for the same reason: they describe a capability
and leave the reader to work out what it is worth. This structure states the
cost of the status quo first, which is the only number that makes the ask
comparable to anything else on the list.

Keep it to one page. Everything below fits.

## The template

    # [PROJECT NAME]

    ## The problem, in money

    [WHO] currently spends [N hours/week] on [TASK]. At a fully loaded rate of
    [$X/hour] that is **[$Y/month]**. [ONE SENTENCE ON WHAT ELSE IT COSTS —
    delay, error rate, the work not being done at all.]

    ## What we would build

    [TWO SENTENCES. The behaviour, not the architecture. No model names.]

    ## What it would cost

    | | |
    | --- | --- |
    | Build | [$N] one-off — [N] weeks of [N] people |
    | Run | [$N]/month at expected volume |
    | Payback | [N] months |

    ## What we are assuming

    - Adoption reaches [N]% within [N] months
    - [TASK] volume stays around [N]/month
    - The saving is [N] hours/week, not the full [N] the task takes today
    - [THE ASSUMPTION YOU ARE LEAST SURE OF]

    Change any of these and the payback moves. They are listed so you can
    argue with them rather than with the conclusion.

    ## What would make us stop

    - If [METRIC] is not [THRESHOLD] by [DATE], this does not work and we stop.
    - [THE OTHER KILL CRITERION]

    ## What we need

    [THE ASK: people, budget, access, a decision by a date.]

## Why it is shaped this way

**Money first, capability second.** A reader comparing this to four other
proposals needs a number in the first paragraph. "We would use RAG to improve
knowledge retrieval" is not a number.

**Assumptions listed, not buried.** A business case whose assumptions are
invisible gets challenged and then discarded. One that lists them gets its
assumptions argued with, which is a conversation you can win.

**A stop condition.** Naming one is the strongest credibility signal in the
document. Proposals without one read as advocacy; a proposal that says what
failure looks like reads as an experiment, and experiments get approved.

**The saving is not the full task.** Claiming a task disappears entirely is the
fastest way to lose a finance reviewer. Partial and defensible beats total and
challenged.

## Getting the numbers

Run Hours Saved for the current cost, Implementation Cost for the build, and
Model ROI for payback and NPV. Each returns its assumptions with it, so the
table above is filled in from something checkable.
