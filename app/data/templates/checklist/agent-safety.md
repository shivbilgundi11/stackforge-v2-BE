---
title: Agent Safety Checklist
category: checklist
difficulty: advanced
summary: >
  Bounding what an agent can do wrong: loop ceilings, side-effect isolation,
  and the injection surface that every tool-using agent has whether or not it
  has been thought about.
use_cases: [agents, automation]
tags: [agents, safety, security, tool-calling]
related_tools: [workflow-plan, function-schema, rate-limits]
premium: true
---

An agent is a program that decides what to run next. This is the list for
bounding what it can decide.

## Termination

- [ ] **A hard step ceiling**, enforced outside the model's judgement.
- [ ] **A wall-clock budget** per run, independent of the step count.
- [ ] **A token budget** per run. Steps and time both stay low while a context
      window fills.
- [ ] **Repeated identical tool calls are detected** and break the loop. The
      commonest runaway is the same failing call retried forever.
- [ ] Someone can **kill a running agent** without deploying.

## Side effects

- [ ] Every tool is classified **read or write**, and the classification is in
      the code rather than in someone's head.
- [ ] **Write tools are idempotent or carry an idempotency key.** If step four
      sends an email and step five fails, a retry from four sends it twice.
- [ ] **Destructive operations need a second call to confirm**, not a flag on
      the first.
- [ ] **Spend-incurring tools have their own budget**, separate from the token
      budget.
- [ ] **No tool can grant the agent more permission** than it started with.

## Tool contracts

- [ ] Every tool **validates its own input** rather than trusting the model.
- [ ] Every tool returns a **structured success or failure**, not prose. A
      model cannot reliably distinguish "no results" from "the query was
      malformed" by reading English.
- [ ] **Results are bounded.** A tool that can return the whole table will, and
      it will fill the context window with it.
- [ ] Descriptions say **when not to use the tool**, not only when to.
- [ ] **Errors are actionable.** "Invalid date format, expected YYYY-MM-DD"
      gets corrected; "400 Bad Request" gets retried identically.

## Injection

- [ ] You know **which tools return attacker-influenced text** — search
      results, fetched pages, user-submitted documents, email bodies.
- [ ] Untrusted content is **delimited and labelled as data** in the prompt.
- [ ] **The blast radius is written down**: if the model followed instructions
      inside a retrieved document, what could it reach? That set is the real
      security boundary, not the prompt.
- [ ] **High-impact tools are unavailable** in the same run as untrusted
      content, or gated behind human approval.
- [ ] **Output is escaped** wherever it is rendered.

## Observability

- [ ] **Every step is traced** with the tool, the arguments, and the result.
- [ ] **The full input to each model call is recoverable**, including the tool
      definitions.
- [ ] **Cost per run is recorded**, and the distribution is watched rather than
      the mean. Runaways live in the tail.
- [ ] **A run can be replayed** from its trace.

## The uncomfortable one

The blast-radius item. It is the only entry here that cannot be satisfied by
adding code, and it is the one that decides whether the others matter.
