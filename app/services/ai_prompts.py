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
every request for a purpose — so it caches. The rule-engine output that varies
per request sits in the user turn, after the cache breakpoint. Interpolating
anything variable into the system text would put it ahead of the breakpoint
and quietly turn every request into a cache miss.

Every schema here obeys the structured-output constraints: `additionalProperties`
is `false` on every object, nothing recurses, and there are no numeric or
length constraints (they are not supported and are silently dropped).
"""

from __future__ import annotations

import json
from typing import Any, Final, NamedTuple

#: Bumped whenever any prompt text or schema below changes. One version for
#: the registry rather than one per prompt: the interesting question is "which
#: build produced this output", and a per-prompt version answers a question
#: nobody asks while making the comparison harder.
PROMPT_VERSION: Final = "v1"

# Models, by what the call is for. Named here rather than at the call site so
# a re-tier is one edit — and so nothing in a route can pick a model.
OPUS: Final = "claude-opus-5"
SONNET: Final = "claude-sonnet-5"
HAIKU: Final = "claude-haiku-4-5"


class Prompt(NamedTuple):
    purpose: str
    model: str
    #: `low` | `medium` | `high` | `xhigh` | `max`. Every call here is a short,
    #: bounded piece of writing over grounding that is already computed, so
    #: none of them need the top of the ladder — and effort is the main lever
    #: on both latency and spend.
    effort: str
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
    model=OPUS,
    effort="medium",
    max_tokens=8000,
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
        ["recommended_id", "confidence", "summary", "why", "trade_offs", "switch_when", "risks"],
    ),
)

ROADMAP: Final = Prompt(
    purpose="roadmap",
    model=SONNET,
    effort="low",
    max_tokens=4000,
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
    model=SONNET,
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
    model=SONNET,
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
    model=OPUS,
    effort="medium",
    max_tokens=8000,
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
    model=HAIKU,
    effort="low",
    max_tokens=1500,
    system=_system(
        "For this call: the engine has scored a set of tools pairwise. Explain what "
        "the weakest pairing actually costs in practice, in one short paragraph."
    ),
    schema=_obj({"summary": _STR, "weakest_pair_impact": _STR}, ["summary", "weakest_pair_impact"]),
)

COMPARISON_RATIONALE: Final = Prompt(
    purpose="comparison_rationale",
    model=HAIKU,
    effort="low",
    max_tokens=1500,
    system=_system(
        "For this call: the engine has scored options against weighted criteria and "
        "picked a winner. Say why the winner won for this profile and when the "
        "runner-up would be the better call."
    ),
    schema=_obj({"why": _STR, "switch_when": _STR}, ["why", "switch_when"]),
)

COST_OPTIMIZATION: Final = Prompt(
    purpose="cost_optimization",
    model=HAIKU,
    effort="low",
    max_tokens=1500,
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
