---
title: Claude Code Build Prompt
category: prompt
difficulty: beginner
summary: >
  A prompt that gets a coding assistant to build the right thing: constraints
  before requirements, the stack pinned, and an explicit instruction to ask
  rather than guess.
use_cases: [coding, automation]
tags: [claude, prompt, coding, scaffolding]
related_tools: [architecture, mcp-config]
---

The failure mode with a coding assistant is not bad code. It is confidently
built code that solves a slightly different problem, and the cause is almost
always a prompt that stated the feature and left the constraints implicit.

Fill the bracketed parts in. The ordering matters: constraints first, because
they eliminate approaches, and an assistant given the requirement first will
have chosen one before it reads them.

## The prompt

    I am building [WHAT] for [WHO].

    ## Fixed constraints

    These are decided. Do not propose alternatives to them unless something
    here is genuinely impossible, in which case stop and say so.

    - Stack: [LANGUAGE / FRAMEWORK / DATABASE]
    - Runs on: [WHERE]
    - Data sensitivity: [public | internal | confidential | regulated]
    - Latency budget: [N] ms end to end
    - Team experience: [beginner | intermediate | advanced]

    ## What I want built

    [ONE PARAGRAPH. The behaviour, not the implementation.]

    ## Definition of done

    - [OBSERVABLE OUTCOME 1]
    - [OBSERVABLE OUTCOME 2]
    - Tests cover [WHAT SPECIFICALLY]

    ## How to work

    - Read the existing code before writing any. Match its conventions,
      including comment density and naming — do not impose a different style.
    - If a requirement is ambiguous in a way that changes the design, ask.
      Do not pick one and proceed.
    - Do not add dependencies without saying why the standard library or an
      existing dependency will not do.
    - Do not write placeholder implementations. If something cannot be
      finished, say which part and why.
    - When you are done, tell me what you did not do.

## Why each rule is there

**"Do not propose alternatives"** stops the assistant relitigating a stack
decision you already made, which is where a surprising amount of a session goes.

**"Read the existing code before writing any"** is the single highest-value
line in the prompt. Code that matches the surrounding conventions gets reviewed;
code in a different dialect gets rewritten.

**"Ask rather than pick"** is scoped deliberately to ambiguity *that changes the
design*. Without the qualifier you get an assistant that asks about everything,
which is its own kind of useless.

**"Tell me what you did not do"** catches the quiet scope reduction — the
assistant that finished four of five things and reported success.
