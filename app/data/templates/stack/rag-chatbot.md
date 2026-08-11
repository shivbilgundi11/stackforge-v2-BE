---
title: RAG Chatbot
category: stack
difficulty: intermediate
summary: >
  Retrieval over your own documents with a chat interface in front of it. The
  most common first AI project, and the one with the most ways to get the
  retrieval layer wrong.
use_cases: [rag, chat]
tags: [rag, chatbot, retrieval, embeddings]
related_tools: [llm-pricing, embedding-cost, chunking-strategy, vectordb-estimate]
stack_input:
  use_case: rag
  scale_target: medium
  monthly_budget: 2000
  team_skill: intermediate
  latency_ms: 2000
  sensitivity: internal
  deployment: any
  capabilities: [hybrid-search, streaming]
---

A chatbot that answers from your documents rather than from the model's
training data. Opening this template loads the constraints below into Stack
Architect, which scores the options against **today's** catalog — so what you
get is a current recommendation, not a snapshot of what was good when this page
was written.

## What the constraints say

A **$2,000/month** budget at **medium scale** with **internal** data. That
combination is what makes this the common case: the data is sensitive enough
that you think about it and not so sensitive that a managed vector store is off
the table, and the budget is large enough for a hosted model and too small for
a GPU fleet.

The **2,000 ms** latency target is deliberately relaxed. A chat interface that
streams tokens feels fast at 2 seconds to first token; the same budget on a
synchronous API call feels broken. If you are building an API rather than a
chat surface, tighten it and the recommendation will change.

## Where this goes wrong

**Chunking is the whole game and it is chosen last.** Most teams pick the
vector store first and the chunk size by copying a tutorial. It is the reverse
of the impact ordering: retrieval quality moves far more with chunk boundaries
that respect document structure than with which database stores the vectors.
Run the Chunking Strategy tool before you commit to an ingestion pipeline.

**The embedding model is a one-way door.** Changing it means re-embedding the
whole corpus, so the cost of the decision is the cost of your ingest run, not
the cost of one API call. Size that before you pick, not after.

**Nobody builds the evaluation set.** Without twenty real questions and their
expected answers, every later change is argued from whoever remembers the last
demo most confidently. It is half a day of work and it is the difference
between tuning and guessing.

## What to run next

Price the model calls, size the ingest, and check the chunking assumptions
before you commit to the stack this template recommends.
