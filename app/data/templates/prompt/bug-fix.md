---
title: Bug Fix Prompt
category: prompt
difficulty: beginner
summary: >
  Gets to a root cause instead of a plausible patch, by demanding an explanation
  of the observed behaviour before any code changes.
use_cases: [coding]
tags: [debugging, prompt, root-cause]
related_tools: []
---

The characteristic failure when asking a model to fix a bug is a change that
looks right, makes the symptom go away in the reproduction, and does not
address the cause. It happens because the prompt asked for a fix, and a fix is
what it produced.

The fix is to require an explanation first.

## The prompt

    There is a bug. Do not change any code yet.

    ## What happens

    [OBSERVED BEHAVIOUR, precisely. Include the exact error and the full
    stack trace if there is one.]

    ## What should happen

    [EXPECTED BEHAVIOUR]

    ## How to reproduce

    [STEPS, or the failing test]

    ## What I have already ruled out

    [ANYTHING YOU CHECKED. Saves the assistant re-checking it.]

    ## What I want, in this order

    1. **An explanation of the observed behaviour.** Not a hypothesis — walk
       the actual code path and show me where it diverges from what I expected.
       If you cannot explain it from the code I have given you, say what else
       you need to see.
    2. **The root cause**, stated in one sentence.
    3. **Whether this bug can exist anywhere else** in the codebase for the
       same reason.
    4. **Then** the smallest change that fixes the cause.
    5. **A test that fails before the change and passes after it.**

    Do not change unrelated code. Do not reformat. If the correct fix is large
    or risky, say so instead of making it.

## The two lines that do the work

**"Walk the actual code path"** blocks the plausible-sounding guess. A model
that has to trace execution finds the real divergence or admits it cannot; one
asked for a cause will produce a confident guess in the same tone either way.

**"Whether this bug can exist anywhere else"** is where the value is. Most bugs
worth fixing carefully are instances of a pattern, and fixing the one you found
while leaving four others is how the same incident happens again next quarter.

## If it cannot explain the behaviour

Take that seriously rather than pushing for a fix anyway. It usually means the
cause is in code you have not shared — a dependency version, a config value, or
a caching layer. Those are exactly the bugs that survive three plausible
patches.
