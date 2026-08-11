---
title: Low-Cost MVP
category: stack
difficulty: beginner
summary: >
  The cheapest stack that is still honest. Open-source where the licence cost
  dominates, managed where the operator cost does, and nothing in it that needs
  someone to run it full time.
use_cases: [rag, chat]
tags: [mvp, budget, open-source, self-hosted]
related_tools: [llm-pricing, budget-estimator, build-vs-buy]
stack_input:
  use_case: rag
  scale_target: small
  monthly_budget: 200
  team_skill: beginner
  latency_ms: 3000
  sensitivity: internal
  deployment: any
  capabilities: []
---

A stack for proving the idea, not for carrying the company. The constraints
below are what make it cheap, and each one is a real trade rather than a
corner cut.

## What the constraints do

The **$200/month** budget is the load-bearing one. Below roughly $500 the Stack
Score weights licence cost heavily, so open-source and self-hostable components
score above managed ones. Above it that inverts, because the cost that actually
bites at scale is engineer-hours rather than subscriptions.

**Beginner** team skill eliminates anything with an operational burden of 4 or
5 outright. It does not rank them down — a component that needs a full-time
operator is not a cheap component for a team that does not have one, whatever
its licence says.

The relaxed **3,000 ms** latency budget is what keeps the cheap options in play.
Tightening it removes the components that are cheap precisely because they are
not fast.

## What you are trading away

Scale headroom, mostly. This stack is credible at small and starts creaking at
medium, which is the correct trade for an MVP and the wrong one for a launch
you expect to work. Re-run the Architect at a large scale target before you
commit to anything here long-term, and expect a different answer.

You are also trading away someone else's on-call rota. Self-hosting is cheaper
in dollars and more expensive in attention, and attention is the scarcer
resource on a small team.
