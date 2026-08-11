---
title: LLM Cost Optimisation Checklist
category: checklist
difficulty: intermediate
summary: >
  Ordered by how much each lever moves the bill, which is not the order most
  teams work in. Caching and model choice dominate; prompt trimming is noise.
use_cases: [rag, chat, agents]
tags: [cost, optimisation, caching, tokens]
related_tools: [llm-pricing, token-calculator, budget-estimator, embedding-cost]
---

Ordered by impact, which is roughly the reverse of the order teams usually
attempt. Measure before and after each change — several of these interact.

## Where the money actually is

- [ ] **You know the split between input and output tokens.** Output is
      typically four to five times the price of input, so a workload with long
      answers has a completely different optimisation than one with long
      prompts.
- [ ] **You know your cost per request at the 95th percentile**, not the mean.
      The tail is where the bill is.
- [ ] **Cost is attributed per feature.** "The AI costs $9,000" is not
      actionable; "summarisation is $7,000 of it" is.

## The two biggest levers

- [ ] **Prompt caching is on** for every stable prefix — system prompt, tool
      definitions, few-shot examples. Cached input is a fraction of the price
      and it is the single largest lever on most chat workloads. Model this
      with the LLM Pricing tool before and after.
- [ ] **The right model per task.** Classification, routing, extraction, and
      summarisation rarely need the frontier model. Tiering by task usually
      halves the bill; tiering by *plan* also makes the cost legible to
      customers.

## Next

- [ ] **A semantic or exact-match cache** in front of repeated questions. Every
      support workload has a head of near-identical queries.
- [ ] **`max_tokens` set deliberately** per endpoint. Output is the expensive
      half and the default is the model's ceiling.
- [ ] **Batch API used for anything not user-facing.** Typically half price for
      work that can wait.
- [ ] **Retrieval `top_k` tuned down.** Every extra chunk is input tokens on
      every call, and more context is not reliably better.

## Embeddings

- [ ] **Embeddings are not recomputed** for unchanged content. Hash the chunk
      and skip.
- [ ] **The embedding model is sized to the job.** The largest one is rarely
      worth its price for retrieval, and the dimension it forces costs storage
      forever.
- [ ] **Ingest cost was estimated before the run**, not discovered after. Use
      the Embedding Cost tool.

## Not worth your time

- **Shaving words off prompts.** A 10% shorter prompt on a cached prefix saves
  almost nothing, and it costs clarity that shows up as quality.
- **Switching provider for a 5% price difference.** The migration costs more
  than a year of the saving, and prices move again.
- **Optimising before measuring.** Most teams' single largest line is one they
  had not thought about — usually retries, or an evaluation suite running
  against production models.

## After

Re-run the Monthly Budget Estimator with the new figures and compare against
the baseline you recorded. An optimisation you cannot show is an optimisation
that gets undone by the next feature.
