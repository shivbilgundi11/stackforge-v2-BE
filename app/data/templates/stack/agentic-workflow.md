---
title: Agentic Workflow
category: stack
difficulty: advanced
summary: >
  Tool-calling agents that do multi-step work. An agent framework rather than a
  RAG one, durable orchestration rather than a request handler, and observability
  that is not optional.
use_cases: [agents, automation]
tags: [agents, tool-calling, orchestration, mcp]
related_tools: [agent-cost, workflow-plan, mcp-config, function-schema]
stack_input:
  use_case: agents
  scale_target: medium
  monthly_budget: 3000
  team_skill: advanced
  latency_ms: 30000
  sensitivity: internal
  deployment: managed
  capabilities: [tool-calling, streaming]
---

Agents that take actions rather than answer questions. Three things about this
template differ from every RAG stack, and each follows from the workload.

## Why the recommendation looks different

**The framework slot resolves to an agent framework, not a RAG one.** Agent-
shaped work wants a library with a tool-calling loop, state, and a retry story
— which is a different category in the catalog, and the Architect switches to
it automatically when the use case is agents, automation, or coding.

**The latency budget is 30 seconds, not two.** An agent doing five tool calls
is not a request-response workload, and pricing it as one produces a stack
optimised for the wrong thing. What matters here is durability under a long
run, not first-token time.

**Orchestration is not optional.** A multi-step agent inside an HTTP request is
an agent that loses its work when the process restarts. That is what the
orchestration role is for, and it is the component teams skip first and add back
after the first production incident.

## The cost surprise

Agent cost is dominated by the **schema overhead you did not count**. Every tool
definition is re-sent on every turn, so a ten-tool agent running a five-step
task pays for those definitions five times. Run the Agent Cost tool with your
real tool count before you budget — the naive estimate is usually wrong by more
than a factor of two.

## Before you build

Sketch the topology with the Workflow Plan tool. Agents fail in ways that are
hard to see from inside a single trace, and a diagram of which step can call
which is the cheapest debugging artifact you will make.
