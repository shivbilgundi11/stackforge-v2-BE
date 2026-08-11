---
title: Enterprise RAG
category: stack
difficulty: advanced
summary: >
  Retrieval where the data cannot leave the network. Self-hostable throughout,
  because regulated data makes that a hard constraint rather than a preference.
use_cases: [rag, search]
tags: [enterprise, compliance, self-hosted, regulated]
related_tools: [vram-estimate, gpu-cost, readiness-checklist, k8s-estimate]
premium: true
stack_input:
  use_case: rag
  scale_target: large
  monthly_budget: 15000
  team_skill: advanced
  latency_ms: 1000
  sensitivity: regulated
  deployment: self-hosted
  capabilities: [hybrid-search, audit-logging]
---

Retrieval for data that is not allowed to leave your network. The interesting
part of this template is not what it recommends but what it **eliminates**.

## Hard constraints eliminate, they do not penalise

A regulated sensitivity removes every component that cannot be self-hosted —
not ranked down, removed. A managed vector store is not a low-scoring option
for regulated data; it is not an option. Softening that into a score is how a
compliance-violating stack ends up ranked third, and third is a position a
tired reader still picks.

The exclusions table on the result is worth reading rather than skipping. It
names every tool that was removed and which constraint did it, so a stack
missing the component you expected reads as a correct engine rather than a
broken one.

## What this costs

**A GPU budget.** Self-hosted generation means you are paying for accelerators
whether or not anyone is asking questions, which is a completely different cost
shape from per-token pricing. Run the VRAM Estimate and GPU Cost tools before
you accept the $15,000 figure — it is a constraint here, not an estimate.

**An operator.** Every component in this recommendation is something your team
runs. The advanced team-skill setting is not aspirational in this template; at
beginner it would eliminate most of the stack.

**Latency headroom.** The 1,000 ms target is achievable self-hosted but it is
not free — it assumes warm weights and a model sized to your hardware rather
than the largest one that fits.

## Before you build

Work through the AI Production Readiness checklist. On a regulated deployment
the gap between "it works" and "it can go live" is mostly audit logging, data
retention, and an answer to what happens when a model returns something it
should not have.
