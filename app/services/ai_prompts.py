"""The prompt registry.

Every prompt the product sends, with a version. Prompts are **never accepted
from the client** — a user-supplied prompt reaching a model with our key is
the whole prompt-injection surface, and there is no feature here that needs
one.

`PROMPT_VERSION` is written to `ai_calls` on every call. Without it, a quality
change after a prompt edit is unattributable: output got worse last Tuesday
and nothing records what changed.

**The system/user split is load-bearing.** The system text carries the role,
the grounding rule, and the output contract, and is byte-identical across
every request for a purpose. The rule-engine output that varies per request
sits in the user turn. Interpolating anything variable into the system text
would make the instructions differ per request, which is exactly the change
nobody would notice and nobody could attribute afterwards.

Every schema here obeys the constraints `strict` structured output imposes:
`additionalProperties` is `false` on every object, every property is listed in
`required`, nothing recurses, and there are no numeric or length keywords.
Unlike a prose "reply with JSON" prompt, breaking one of these is a 400 at the
first call rather than a slow drift in output quality — which is the point.
"""

from __future__ import annotations

import json
from typing import Any, Final, Literal, NamedTuple

#: The values the provider accepts for `reasoning_effort`. Spelled as a type
#: rather than checked at the call site: a bad effort is a 400, and the
#: registry below is the only place one is ever chosen.
Effort = Literal["low", "medium", "high"]

#: Bumped whenever any prompt text or schema below changes. One version for
#: the registry rather than one per prompt: the interesting question is "which
#: build produced this output", and a per-prompt version answers a question
#: nobody asks while making the comparison harder.
PROMPT_VERSION: Final = "v4"

# Models, by what the call is for. Named here rather than at the call site so
# a re-tier is one edit — and so nothing in a route can pick a model.
#
# These are Gemini ids, and the tiering is not cosmetic. The free tier's
# allowance is **20 requests per day per model**, so two tiers is two
# allowances: pointing every prompt at one id would take the whole product
# down after twenty requests, while the split lets the short rationales keep
# working after the flagship has spent its own.

#: The heavy tier — judgement calls the user will check line by line, and the
#: architecture prose they will hand to someone else.
LARGE: Final = "gemini-3.6-flash"
#: The default. Same model today; kept as its own name so re-tiering the
#: middle of the registry stays one edit rather than a search and replace.
MEDIUM: Final = "gemini-3.6-flash"
#: Short single-paragraph rationales, where the lite model is
#: indistinguishable and materially faster — and, more to the point, draws on
#: a separate daily allowance.
SMALL: Final = "gemini-3.5-flash-lite"


#: The largest grounding payload any prompt here sends, measured from a real
#: `stack-architect` run. Kept next to the reservations below because it is
#: the other half of what has to fit.
GROUNDING_ALLOWANCE: Final = 4_500

#: What a prompt may reserve for its answer.
#:
#: A spending ceiling, not a provider limit — the models here will emit far
#: more than this. At the heavy tier's output rate a full reservation is about
#: three cents, which is the most any single synthesis call is worth.
#:
#: The binding constraint is the other direction, and it is the one that bites:
#: **thinking tokens come out of the same reservation as the answer**. Exhaust
#: it and the call returns a 200 with `MAX_TOKENS` and no content at all — not
#: a truncated answer, an empty one, which degrades to `rule_based` and reads
#: like a schema fault. Measured on the real prompts, thinking runs from 200
#: tokens on a one-paragraph rationale to 2,700 on the Architect's assessment,
#: so every number below is sized against *that* and not against the length of
#: the prose it is meant to produce.
MAX_OUTPUT_RESERVATION: Final = 8_000

#: And the floor, which is the number that actually gets violated. A prompt
#: reserving less than this has no room for the model to think before it
#: answers, and the failure is silent: a 200, no content, `rule_based` on the
#: page. Every prompt here was measured against a real request before its
#: number was chosen.
MIN_OUTPUT_RESERVATION: Final = 2_000


class Prompt(NamedTuple):
    purpose: str
    model: str
    #: Passed through as `thinkingConfig.thinkingLevel`. Every call here is a
    #: short, bounded piece of writing over grounding that is already
    #: computed, so none of them need the top of the ladder — and it is the
    #: main lever on both latency and spend, because thinking tokens are
    #: billed at the output rate.
    effort: Effort
    #: The **reservation**, not a prediction — and on this provider it is
    #: only charged for what is used, so the risk runs the other way. Too low
    #: is the failure that matters: thinking is drawn from this budget before
    #: the answer is, and a reservation thinking exhausts comes back empty.
    #:
    #: Sized at several times the observed prose, which is what leaves room
    #: for the thinking that precedes it. The short-rationale prompts sat at
    #: 1,500 and intermittently produced a complete first field and nothing
    #: after it, because a two-field schema whose first field runs long has no
    #: budget left for the second. That failure reads as a schema fault and is
    #: a budget one, which is the whole reason this field carries a paragraph.
    max_tokens: int
    system: str
    schema: dict[str, Any]


def _obj(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    """An object schema in the shape structured outputs require."""
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


_STR = {"type": "string"}
_STR_LIST = {"type": "array", "items": {"type": "string"}}


# ── the grounding rule ───────────────────────────────────────────────────────

#: Prepended to every system prompt. This is the sentence that makes the whole
#: layer safe: the engine has already produced a complete, correct answer, and
#: the model's job is to explain it. A model that invents a tool or a price
#: here would be inventing one *on top of* a correct result, which is worse
#: than a wrong result — it looks verified.
GROUNDING: Final = """\
You are the explanation layer of an engineering planning tool. A deterministic \
rule engine has already produced the answer below. Your job is to explain, \
rank, and caveat what it produced — never to change it.

Rules you must follow:
- Use only the tools, models, numbers, and options present in the grounding data.
- Never introduce a tool, vendor, model, or price that is not in the grounding data.
- Never restate a number differently from the grounding data. If a figure looks \
wrong, say so in a caveat rather than correcting it.
- If the grounding data does not support a claim, leave the claim out.
- Write for a senior engineer who will check your reasoning. Be specific and brief; \
no marketing language, no hedging filler."""


def _system(role: str) -> str:
    return f"{GROUNDING}\n\n{role}"


# ── prompts ──────────────────────────────────────────────────────────────────

STACK_SYNTHESIS: Final = Prompt(
    purpose="stack_synthesis",
    model=LARGE,
    effort="medium",
    max_tokens=6000,
    system=_system(
        "For this call: the engine has ranked candidate stacks and scored every "
        "dimension. Choose which of the ranked candidates to recommend — you may "
        "only choose one the engine produced — and explain why it beats the others "
        "for these specific requirements. Name the real trade-off being made, and "
        "the condition under which the second choice would be the better answer."
    ),
    schema=_obj(
        {
            "recommended_id": _STR,
            "score_breakdown": {
                "type": "array",
                "minItems": 10,
                "maxItems": 10,
                "items": _obj(
                    {
                        "key": {
                            "type": "string",
                            "enum": [
                                "cost_efficiency",
                                "scalability",
                                "developer_experience",
                                "production_readiness",
                                "security_readiness",
                                "vendor_lock_in",
                                "integration_compatibility",
                                "deployment_complexity",
                                "community_maturity",
                                "documentation_quality",
                            ],
                        },
                        "score": {"type": "number"},
                    },
                    ["key", "score"],
                ),
            },
            "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
            "summary": _STR,
            "why": _STR,
            "trade_offs": _STR_LIST,
            "switch_when": _STR_LIST,
            "risks": {
                "type": "array",
                "items": _obj(
                    {
                        "risk": _STR,
                        "severity": {"type": "string", "enum": ["high", "medium", "low"]},
                        "mitigation": _STR,
                    },
                    ["risk", "severity", "mitigation"],
                ),
            },
        },
        [
            "recommended_id",
            "score_breakdown",
            "confidence",
            "summary",
            "why",
            "trade_offs",
            "switch_when",
            "risks",
        ],
    ),
)

ROADMAP: Final = Prompt(
    purpose="roadmap",
    model=MEDIUM,
    effort="low",
    max_tokens=3000,
    system=_system(
        "For this call: write a five-step implementation roadmap for the stack the "
        "engine selected. Each step names what is built, roughly how long it takes "
        "for one engineer, what it depends on, and how you know it is done. Order "
        "by dependency, not by importance. Steps must only involve components in "
        "the grounding data."
    ),
    schema=_obj(
        {
            "steps": {
                "type": "array",
                "items": _obj(
                    {
                        "title": _STR,
                        "detail": _STR,
                        "effort": _STR,
                        "depends_on": _STR,
                        "done_when": _STR,
                    },
                    ["title", "detail", "effort", "depends_on", "done_when"],
                ),
            }
        },
        ["steps"],
    ),
)

RAG_ARCHITECTURE: Final = Prompt(
    purpose="rag_architecture",
    model=MEDIUM,
    effort="low",
    max_tokens=3000,
    system=_system(
        "For this call: the engine has selected a component for every stage of a "
        "RAG pipeline under hard constraints. Explain the design as a whole — what "
        "the shape of it is optimising for, which single stage most determines "
        "retrieval quality here, and what to measure first."
    ),
    schema=_obj(
        {"summary": _STR, "why": _STR, "watch_out_for": _STR_LIST, "measure_first": _STR_LIST},
        ["summary", "why", "watch_out_for", "measure_first"],
    ),
)

AGENT_PLAN: Final = Prompt(
    purpose="agent_plan",
    model=MEDIUM,
    effort="low",
    max_tokens=3000,
    system=_system(
        "For this call: the engine has chosen an agent topology, assigned roles, "
        "and written handoff contracts. Explain why this shape suits the goal, and "
        "which edge is most likely to be where it breaks. Do not add or remove "
        "agents."
    ),
    schema=_obj(
        {"summary": _STR, "why": _STR, "weakest_link": _STR, "watch_out_for": _STR_LIST},
        ["summary", "why", "weakest_link", "watch_out_for"],
    ),
)

ARCHITECTURE_DOCUMENT: Final = Prompt(
    purpose="architecture_document",
    model=LARGE,
    effort="medium",
    max_tokens=4000,
    system=_system(
        "For this call: write the prose sections of an architecture document for "
        "the selected stack — an overview, the reasoning behind each major choice, "
        "and the operational concerns someone inherits with it. Markdown body text "
        "only; no headings, the document assembles those."
    ),
    schema=_obj(
        {"overview": _STR, "decisions": _STR, "operations": _STR},
        ["overview", "decisions", "operations"],
    ),
)

COMPATIBILITY_RATIONALE: Final = Prompt(
    purpose="compatibility_rationale",
    model=SMALL,
    effort="low",
    max_tokens=2500,
    system=_system(
        "For this call: the engine has scored a set of tools pairwise. Explain "
        "what the weakest pairing means for the team that has to run it — the "
        "integration work it implies, what breaks first, and what has to be "
        "operated by hand. Engineering effort, not money: the grounding "
        "carries no prices and a paragraph about cost becomes a paragraph "
        "about not being able to calculate one."
    ),
    schema=_obj({"summary": _STR, "weakest_pair_impact": _STR}, ["summary", "weakest_pair_impact"]),
)

COMPARISON_RATIONALE: Final = Prompt(
    purpose="comparison_rationale",
    model=SMALL,
    effort="low",
    max_tokens=2500,
    system=_system(
        "For this call: the engine has scored options against weighted criteria and "
        "picked a winner. Say why the winner won for this profile and when the "
        "runner-up would be the better call."
    ),
    schema=_obj({"why": _STR, "switch_when": _STR}, ["why", "switch_when"]),
)

COST_OPTIMIZATION: Final = Prompt(
    purpose="cost_optimization",
    model=SMALL,
    effort="low",
    max_tokens=2500,
    system=_system(
        "For this call: the engine has broken a cost down by line. Name the two "
        "changes that would reduce it most, in order, and what each one costs in "
        "quality or effort. Only suggest changes the breakdown supports."
    ),
    schema=_obj(
        {
            "suggestions": {
                "type": "array",
                "items": _obj(
                    {"change": _STR, "saves": _STR, "costs_you": _STR},
                    ["change", "saves", "costs_you"],
                ),
            }
        },
        ["suggestions"],
    ),
)

REGISTRY: Final[dict[str, Prompt]] = {
    prompt.purpose: prompt
    for prompt in (
        STACK_SYNTHESIS,
        ROADMAP,
        RAG_ARCHITECTURE,
        AGENT_PLAN,
        ARCHITECTURE_DOCUMENT,
        COMPATIBILITY_RATIONALE,
        COMPARISON_RATIONALE,
        COST_OPTIMIZATION,
    )
}


def user_turn(grounding: dict[str, Any], variables: dict[str, Any]) -> str:
    """The varying half of the request, after the cache breakpoint.

    JSON with sorted keys: two runs with the same data must produce the same
    bytes, or the prefix differs and nothing caches. `default=str` because
    grounding routinely carries `Decimal` — a serialisation failure here would
    turn a working tool into a degraded one for no visible reason.
    """
    return (
        "## Requirements\n\n"
        + json.dumps(variables, indent=2, sort_keys=True, default=str)
        + "\n\n## Rule engine output (the grounding data)\n\n"
        + json.dumps(grounding, indent=2, sort_keys=True, default=str)
    )
