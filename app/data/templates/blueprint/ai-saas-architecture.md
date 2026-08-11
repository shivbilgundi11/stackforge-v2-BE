---
title: AI SaaS Blueprint
category: blueprint
difficulty: advanced
summary: >
  Multi-tenant AI as a product: tenant isolation in the vector store, usage
  metering that survives a crash, and the cost controls that stop one customer
  spending your margin.
use_cases: [rag, chat, agents]
tags: [saas, multi-tenant, metering, billing]
related_tools: [budget-estimator, model-roi, build-vs-buy, cloud-cost]
premium: true
---

Selling AI to many customers adds three problems that a single-tenant system
never has: isolation, metering, and one customer's usage becoming everyone's
cost.

## Tenant isolation in the vector store

| Approach | Isolation | Cost | When |
| --- | --- | --- | --- |
| Namespace per tenant | Logical | One index | The default. Scales to thousands. |
| Metadata filter | Logical, weakest | One index | Only if your store lacks namespaces. |
| Index per tenant | Physical | One index each | Regulatory demand, or very few large tenants. |

Namespaces are almost always right. Index-per-tenant hits account limits and
carries a per-index minimum charge, which turns a thousand small customers into
a thousand minimum charges.

A metadata filter alone is the risky one: it is one forgotten `WHERE` from a
cross-tenant leak. If you must use it, apply the filter in a repository layer
that has no method to omit it — a check that has to be remembered will be
forgotten at the fifteenth call site.

## Metering

**Record usage at the point it is incurred, in the same transaction as the
work.** A counter incremented after the response is a counter that misses
everything that crashed mid-flight, and those are exactly the expensive calls.

Meter tokens, not requests. A request is not a unit of cost, and a pricing model
built on requests either loses money on the heavy users or overcharges the light
ones until they leave.

## Cost controls

**A per-tenant cap, enforced server-side.** One customer looping an agent
against your account is not hypothetical, and the first you hear of it is the
invoice.

**A per-tenant rate limit, separate from the cap.** The cap protects the month;
the limit protects the next ten minutes.

**Model choice as a plan feature.** The largest model is where the margin goes.
Tiering by model is the cleanest cost control available, and it is legible to
the customer in a way a token quota is not.

## Margin

Run the Monthly Budget Estimator with your actual per-tenant volumes rather than
an average. AI margin is dominated by the tail: the mean tenant is cheap, and
the ninety-fifth percentile decides whether the plan price works.
