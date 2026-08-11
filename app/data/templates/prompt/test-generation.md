---
title: Test Generation Prompt
category: prompt
difficulty: intermediate
summary: >
  Produces tests that assert computed values rather than status codes, by
  banning the specific patterns that make a suite pass while the code is broken.
use_cases: [coding]
tags: [testing, prompt, quality]
related_tools: []
---

Ask for tests and you get coverage. Coverage is not the goal — a suite that
exercises every line and asserts nothing about the results passes just as
happily when the arithmetic is wrong.

This prompt bans the patterns that produce that suite.

## The prompt

    Write tests for the code below.

    ## The code

    [PASTE, or name the file]

    ## What this code is for

    [ONE PARAGRAPH. What would be *wrong* if it broke — not what it does.]

    ## Rules

    - **Assert computed values, not status codes.** A test that checks for 200
      passes when the response body is empty. If the function returns a number,
      the expected number goes in the test, worked out by hand.
    - **No mocks for code I own.** Mock the network and the clock. Mocking my
      own function tests the mock.
    - **One behaviour per test**, and the test name is the behaviour in a
      sentence — not `test_function_name_2`.
    - **Cover the edges I will actually hit**: empty input, one item, the
      boundary value, and the value just past it. Not every theoretical input.
    - **Include the failure cases.** What should raise, and what it should
      raise. A suite with no failing paths is testing half the contract.
    - No test that would pass if I deleted the function body and returned a
      constant. If you write one, delete it.

    ## What I do not want

    - Setup that is longer than the assertion.
    - Tests for framework behaviour. I am not testing that the router routes.
    - A test per getter.

    Tell me which behaviours you chose not to cover, and why.

## The rule that matters most

**"Assert computed values, worked out by hand."** A test that recomputes the
implementation's own formula proves the code is stable, not that it is correct.
If the function says a stack costs $126.00 a month, the number `126.00` belongs
in the test, derived independently — that is the only version that catches a
sign error.

## The last line

**"Tell me which behaviours you chose not to cover."** Every generated suite has
gaps. The ones stated are gaps you can decide about; the ones unstated are gaps
you find in production.
