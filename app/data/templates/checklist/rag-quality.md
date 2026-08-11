---
title: RAG Quality Checklist
category: checklist
difficulty: intermediate
summary: >
  Diagnosing bad answers by finding out which stage is wrong first, because
  retrieval failures and generation failures look identical from the outside
  and need opposite fixes.
use_cases: [rag, search]
tags: [rag, quality, evaluation, retrieval]
related_tools: [chunking-strategy, chunk-estimate, pipeline-cost]
---

"The answers are bad" is not a diagnosis. Work down this list in order — each
section only makes sense once the one above it is clean.

## First: which stage is failing?

- [ ] For twenty real questions, record **whether the right chunk was
      retrieved at all**. This one number splits every remaining problem in two.
- [ ] For the questions where retrieval succeeded, record **whether the answer
      used it**. Ignoring good context is a generation problem.
- [ ] **Return chunk ids with every answer.** Without them, "the model made
      this up" and "retrieval returned the wrong document" are the same
      observation.

If retrieval is the problem, keep reading. If generation is, skip to the last
section — tuning chunking will not help you.

## Parsing

- [ ] **Tables survive parsing.** A table flattened into a character stream is
      a table nobody can answer questions about.
- [ ] **Reading order is correct** for multi-column documents. PDFs are
      routinely parsed left-to-right across columns.
- [ ] **Headings are preserved** and attached to the content under them.
- [ ] Spot-check the parsed text of your **five most important documents** by
      reading them. This finds more than any metric.

## Chunking

- [ ] Chunks **respect structure** — paragraph or section boundaries, not a
      character count.
- [ ] **Chunk size is measured in tokens**, not characters.
- [ ] **Overlap exists**, so a fact spanning a boundary is retrievable from
      either side.
- [ ] **Each chunk carries its document title and section** in the text. A
      chunk that reads as an orphan paragraph retrieves as one.
- [ ] You have **tried two chunk sizes and compared them on the evaluation
      set**. Chunking is the highest-leverage retrieval decision and it is
      usually copied from a tutorial.

## Retrieval

- [ ] **The index exists.** pgvector and friends do a sequential scan without
      one and stay fast enough in development to hide it.
- [ ] **One embedding model across the whole corpus.** Two models in one index
      produce subtly wrong retrieval that is very hard to see.
- [ ] **`top_k` has been tuned.** More context is not better — it dilutes.
- [ ] **Hybrid search considered** if queries contain names, codes, or exact
      strings. Dense vectors are poor at exact match.
- [ ] **Reranking evaluated** — after a baseline exists, not before.

## Generation

- [ ] The prompt says **what to do when the context does not contain the
      answer**, and the model does it.
- [ ] **Citations are required** in the prompt and checked in the evaluation.
- [ ] **Context is ordered** with the most relevant material closest to the
      question.
- [ ] The prompt is **versioned**, and a regression can be traced to a change.

## The one to do first

Build the evaluation set. Twenty real questions with expected answers is half a
day, and without it every item above is argued from whoever remembers the last
demo most confidently.
