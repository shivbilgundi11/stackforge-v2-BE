---
title: Voice AI
category: stack
difficulty: advanced
summary: >
  Speech in, speech out, under a latency budget that eliminates most of the
  catalog. The constraint that shapes everything is that people notice a pause.
use_cases: [chat, agents]
tags: [voice, realtime, streaming, latency]
related_tools: [llm-pricing, token-calculator, gpu-cost]
stack_input:
  use_case: chat
  scale_target: medium
  monthly_budget: 4000
  team_skill: advanced
  latency_ms: 400
  sensitivity: confidential
  deployment: managed
  capabilities: [streaming, realtime]
---

Conversational voice, where the whole design is downstream of one number.

## 400 milliseconds

A pause longer than roughly half a second reads as the system being broken
rather than thinking. That budget has to cover speech recognition, retrieval if
any, generation to first token, and speech synthesis — so each stage gets
around a hundred milliseconds, and anything batch-oriented is eliminated before
scoring even starts.

This is why the recommendation is short. Most of the catalog cannot fit in the
budget, and the Architect removes those rather than ranking them down.

## What this rules out

**Retrieval in the critical path**, usually. A vector search plus a rerank is
comfortably 200 ms on its own, which is half the budget for one stage. Voice
systems that need retrieval generally do it speculatively — starting the search
on partial transcription rather than waiting for the final one.

**Large models.** Time-to-first-token scales with model size, and the largest
model that fits your accelerator is rarely the one that fits your latency
budget. Size for the budget, then check quality, not the other way round.

**Anything with a cold start.** A component that takes thirty seconds to load
weights is a component that fails the first call after every scale-up.

## What to measure

Time to first *audio*, not time to first token. They are different numbers and
only one of them is what the caller experiences.
