---
title: AI Production Readiness Checklist
category: checklist
difficulty: intermediate
summary: >
  The gap between "it works in the demo" and "it can go live", as things you
  either have or do not. Ordered by how expensive each is to retrofit.
use_cases: [rag, chat, agents]
tags: [production, readiness, operations, launch]
related_tools: [readiness-checklist, rate-limits, budget-estimator]
---

Ordered so the expensive-to-retrofit items come first. Everything here is
observable — you can point at the thing or you cannot.

## Cost control

- [ ] A **hard spend cap** enforced server-side, not a billing alert. An alert
      tells you after the money is gone.
- [ ] A **per-user or per-tenant limit** separate from the global cap. The cap
      protects the month; the limit protects the next ten minutes.
- [ ] **Token counts recorded per request**, in the same transaction as the
      work. A counter incremented after the response misses everything that
      crashed — which is the expensive tail.
- [ ] **Max tokens set explicitly** on every call. The default is the largest
      the model allows.
- [ ] Someone can answer **"what did last month cost, broken down by feature"**.

## Failure behaviour

- [ ] A **timeout on every model call**, shorter than your request timeout.
- [ ] A **defined behaviour when the provider is down** — degrade, queue, or
      fail with a real message. Not a 500 with a stack trace.
- [ ] **Retries are bounded and backed off**, and non-idempotent operations are
      not retried at all.
- [ ] Rate-limit responses are **handled as a signal, not an error**. Read the
      published limits with the Rate Limits tool and check yours against them.
- [ ] The system **works with the AI layer disabled**. If the deterministic
      path is the product, prove it still runs.

## Data and safety

- [ ] You can say **what user data reaches the provider**, and it matches what
      you told users.
- [ ] **Prompt injection has been considered** for anything that reads
      untrusted text. Not solved — considered, with the blast radius written
      down.
- [ ] **Output is not executed or rendered as HTML** without escaping.
- [ ] **A retention policy exists** for prompts and completions, and it is
      implemented rather than described.
- [ ] **PII is redacted before logging.** Traces are the most common accidental
      PII store in an AI system.

## Observability

- [ ] **Every call is traced** with model, token counts, latency, and cost.
- [ ] **Retrieval quality and generation quality are measured separately.** A
      single end-to-end score hides which half is broken.
- [ ] **An evaluation set exists** — at least twenty real inputs with expected
      outputs — and it runs on every prompt change.
- [ ] **Prompt versions are recorded on each call**, so a regression can be
      traced to the change that caused it.
- [ ] You are alerted on **cost per request** moving, not just error rate. The
      expensive failure is usually a successful one.

## Operations

- [ ] **A rollback path for a prompt change** that does not need a deploy.
- [ ] **Model version is pinned.** A provider-side default moving under you is
      a silent behaviour change.
- [ ] **A deploy and a rollback have both been exercised**, not just the deploy.
- [ ] **Someone is on call** and knows what a model outage looks like from the
      dashboard.

## The four that get skipped

The evaluation set, the spend cap, the prompt version on each call, and the
exercised rollback. Each takes under a day before launch and each is unpleasant
to add during an incident.
