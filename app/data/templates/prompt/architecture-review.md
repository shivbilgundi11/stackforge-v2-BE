---
title: Architecture Review Prompt
category: prompt
difficulty: intermediate
summary: >
  Gets a model to review a design like a sceptical colleague rather than
  congratulate it: failure modes first, the one-way doors named, and an explicit
  instruction to disagree.
use_cases: [coding, automation]
tags: [review, architecture, prompt, critique]
related_tools: [architecture, compatibility, readiness-checklist]
---

Ask a model to review a design and you get a list of things the design does
well. That is not a review. The prompt below asks for the opposite, and the
constraints are what make the difference.

## The prompt

    Review this architecture as a sceptical staff engineer who will be on call
    for it. I want the problems, not the summary.

    ## The design

    [PASTE THE ARCHITECTURE DOCUMENT OR DIAGRAM]

    ## Context

    - Scale: [REQUESTS/DAY, DATA VOLUME]
    - Team: [SIZE, EXPERIENCE]
    - Latency budget: [N] ms
    - Data sensitivity: [LEVEL]
    - Budget: [$N/month]

    ## What I want back

    1. **The three most likely ways this fails in production.** Concrete
       failure, concrete trigger. Not "it might not scale".
    2. **Every one-way door.** Which decisions here are expensive to reverse,
       and what each would cost to change in six months.
    3. **What is missing.** Components or concerns the design does not mention
       at all — and say if the omission is fine.
    4. **Where the design is more complex than the requirements justify.**
    5. **The one thing you would change first**, and what it buys.

    ## How to answer

    - If something is fine, say nothing about it. I am not looking for balance.
    - Where you are uncertain, say so and say what would resolve it.
    - Disagree with the design's stated reasoning where you think it is wrong.
      Do not restate it back to me as agreement.
    - Do not suggest a rewrite. Assume the stack is fixed.

## Why it works

**"Who will be on call for it"** is the highest-leverage phrase in the prompt.
It shifts the frame from evaluating a document to imagining an incident, and
incident-shaped thinking produces specific failures rather than generic risks.

**"If something is fine, say nothing"** removes the padding. Most review output
is padding, and padding is where the two real findings get lost.

**"Do not suggest a rewrite"** keeps the answer actionable. An unconstrained
model will propose a different architecture, which is easy to write and
useless to someone who has already built half of this one.

## After the review

Run the compatibility checker over the components it questioned. A pairing
nobody has reviewed is reported as unknown rather than assumed to work, and
"unknown" is exactly what a reviewer should be told.
