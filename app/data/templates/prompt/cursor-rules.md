---
title: Cursor Rules Template
category: prompt
difficulty: beginner
summary: >
  A .cursorrules that changes completions rather than decorating the repository:
  the actual stack, the actual gotchas, and explicit instructions not to invent
  numbers.
use_cases: [coding]
tags: [cursor, rules, assistant, conventions]
related_tools: [architecture]
---

A rules file that says "use best practices" is noise an assistant will ignore.
One that pins the libraries, names the versions of the decisions, and states
the gotchas of *these* components is a file that changes what gets typed.

StackForge generates one of these from a saved stack, with the component
gotchas filled in. This is the shape if you are writing it by hand.

```text path=.cursorrules
# [PROJECT NAME]

This project is built on a fixed stack. Do not introduce an alternative to any
component below without being asked — a suggestion that swaps the vector store
is a suggestion to rewrite the retrieval layer.

## The stack

- **[LLM provider]** — generation
- **[Framework]** — retrieval and orchestration
- **[Vector store]** — embedding storage
- **[Database]** — application state
- **[Cache]** — hot paths and rate limiting

## Constraints that decided this stack

- **Latency budget**: [N] ms end to end. Anything synchronous on the request
  path has to fit inside it.
- **Data sensitivity**: [LEVEL].
- **Scale target**: [SMALL | MEDIUM | LARGE].

## Component rules

- **[Vector store]**: [THE GOTCHA THAT COSTS A DAY. e.g. "Create the HNSW
  index explicitly — it does a sequential scan without one and stays fast
  enough in development to hide it."]
- **[Framework]**: [e.g. "Set the embedding model on the Settings object once.
  Passing it per-call is where two embedding models end up in one index."]
- **[Provider]**: [e.g. "Structured output belongs in response_format, not in
  a prompt asking for JSON. The API guarantees the schema; the prompt does not."]

## Do not

- Do not invent pricing, rate limits, or context-window figures. If a number is
  needed and not in the code, say so instead of guessing.
- Do not add a dependency without saying why an existing one will not do.
- Do not write placeholder implementations and describe them as finished.
- Do not put a blocking call inside an async handler. It stalls the whole event
  loop and is invisible until load.

## General

- Match the conventions already in the file you are editing, including comment
  density and naming.
- Prefer the boring documented path over the clever one, and explain the trade
  when you take it.
```

## The rule that earns its place

**"Do not invent pricing, rate limits, or context-window figures."** Assistants
are confidently wrong about all three, the numbers change monthly, and a
hardcoded stale limit fails in production rather than at review. This one line
prevents a category of bug rather than a style issue.
