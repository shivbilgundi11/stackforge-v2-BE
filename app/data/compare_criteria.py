"""Comparison criteria, as data.

Adding a criterion — or a whole comparison type — is an edit to this file plus
a spec file on the frontend. No renderer change, no migration. That is the
`PRD.md` §22 requirement, and it is why criteria are declarative records rather
than branches in a scoring function.

Two kinds of criterion:

  * **fact** — reads a number straight off `tool_catalog.facts`, normalised
    from its natural range onto 0-100.
  * **computed** — the service calculates it from the user's stated scale
    (cost at 10M vectors, cost for this token mix). Never stored, because a
    hardcoded "cost: 7/10" is wrong the day a provider changes price.

`priority` reweights. A comparison with one fixed weighting is an opinion; one
that reweights is a tool — the honest answer to "which vector DB" genuinely
does change depending on whether you are optimising for cost or for not being
paged at 3am.
"""

from __future__ import annotations

from typing import Literal, NamedTuple

Priority = Literal["balanced", "cost", "scale", "speed", "simplicity", "control"]

PRIORITIES: tuple[Priority, ...] = (
    "balanced",
    "cost",
    "scale",
    "speed",
    "simplicity",
    "control",
)


class Criterion(NamedTuple):
    key: str
    label: str
    description: str
    kind: Literal["fact", "computed"]
    # Base weight, then per-priority multipliers. A multiplier of 0 drops the
    # criterion entirely for that priority.
    weight: float
    weights: dict[str, float]
    # For `fact` criteria: which `facts` key, its natural maximum, and whether
    # a high raw value is bad (ops burden, lock-in) and must be inverted.
    fact_key: str | None = None
    fact_max: float = 5.0
    invert: bool = False
    unit: str | None = None


def _w(**overrides: float) -> dict[str, float]:
    """Per-priority multipliers, defaulting to 1.0."""
    # `list(PRIORITIES)` widens the key type from the Literal to plain str,
    # which is what the Criterion field is annotated as.
    base: dict[str, float] = dict.fromkeys(list(PRIORITIES), 1.0)
    base.update(overrides)
    return base


VECTOR_DB_CRITERIA: tuple[Criterion, ...] = (
    Criterion(
        "monthly_cost",
        "Monthly cost at your scale",
        "Computed from your vector count and dimensions, not a stored score.",
        "computed",
        1.0,
        _w(cost=3.0, scale=1.0, simplicity=0.5, speed=0.5),
        unit="USD/month",
    ),
    Criterion(
        "ops_burden",
        "Operational burden",
        "How much of your week this takes. Managed services score highest.",
        "fact",
        1.0,
        _w(simplicity=3.0, control=0.4, cost=0.7),
        fact_key="ops_burden",
        invert=True,
    ),
    Criterion(
        "filtering",
        "Metadata filtering",
        "Expressiveness of filters alongside vector search.",
        "fact",
        1.0,
        _w(scale=1.2),
        fact_key="filtering",
    ),
    Criterion(
        "hybrid_search",
        "Hybrid search",
        "Built-in BM25 or sparse retrieval alongside dense vectors.",
        "fact",
        0.8,
        _w(),
        fact_key="hybrid_search",
        fact_max=1.0,
    ),
    Criterion(
        "scale_ceiling",
        "Scaling ceiling",
        "The largest deployment this is credible at.",
        "fact",
        1.0,
        _w(scale=3.0, cost=0.6),
        fact_key="scale_ceiling",
    ),
    Criterion(
        "ecosystem",
        "Ecosystem maturity",
        "Integrations, documentation, and how easy it is to hire for.",
        "fact",
        1.0,
        _w(simplicity=1.5),
        fact_key="ecosystem",
    ),
    Criterion(
        "vendor_lock_in",
        "Portability",
        "How hard it is to leave. Open-source and self-hostable score highest.",
        "fact",
        1.0,
        _w(control=3.0, cost=1.2, simplicity=0.5),
        fact_key="lock_in",
        invert=True,
    ),
    Criterion(
        "lifecycle",
        "Lifecycle status",
        "Catalog status. A deprecated tool cannot win regardless of its numbers.",
        "computed",
        1.2,
        _w(),
    ),
)

MODEL_CRITERIA: tuple[Criterion, ...] = (
    Criterion(
        "blended_cost",
        "Cost for your usage profile",
        "Computed from the token mix you supplied against live catalog pricing.",
        "computed",
        1.0,
        _w(cost=3.0, speed=0.7, scale=1.2),
        unit="USD/month",
    ),
    Criterion(
        "context_window",
        "Context window",
        "Largest prompt the model accepts.",
        "computed",
        1.0,
        _w(scale=2.0, cost=0.6),
        unit="tokens",
    ),
    Criterion(
        "cache_support",
        "Prompt caching",
        "A published cached-input rate, which changes RAG and agent economics.",
        "computed",
        0.9,
        _w(cost=2.0, speed=1.5),
    ),
    Criterion(
        "reasoning",
        "Reasoning depth",
        "Extended or adaptive thinking support.",
        "computed",
        1.0,
        _w(speed=0.5, simplicity=0.8),
    ),
    Criterion(
        "tool_use",
        "Tool use",
        "Function calling and structured output support.",
        "computed",
        1.0,
        _w(),
    ),
    Criterion(
        "multimodal",
        "Multimodal input",
        "Image and document understanding.",
        "computed",
        0.7,
        _w(),
    ),
    Criterion(
        "output_ceiling",
        "Max output",
        "Longest single response.",
        "computed",
        0.7,
        _w(scale=1.3),
    ),
    Criterion(
        "freshness",
        "Lifecycle status",
        "Active, deprecated, or retired.",
        "computed",
        1.1,
        _w(),
    ),
)

STACK_CRITERIA: tuple[Criterion, ...] = (
    Criterion(
        "tco_12_month",
        "12-month total cost",
        "Infrastructure plus model spend plus estimated engineering time.",
        "computed",
        1.0,
        _w(cost=3.0),
        unit="USD",
    ),
    Criterion(
        "time_to_deploy",
        "Time to first deploy",
        "Working days from zero to something running.",
        "computed",
        1.0,
        _w(speed=3.0, simplicity=2.0),
        unit="days",
    ),
    Criterion(
        "scaling_ceiling",
        "Scaling ceiling",
        "Where this stack stops being the right answer.",
        "computed",
        1.0,
        _w(scale=3.0),
    ),
    Criterion(
        "vendor_lock_in",
        "Portability",
        "How much of this moves if you change provider.",
        "computed",
        1.0,
        _w(control=3.0, cost=1.2),
    ),
    Criterion(
        "team_skill",
        "Team skill required",
        "Lower is better: how senior the team has to be to run it.",
        "computed",
        1.0,
        _w(simplicity=2.5, control=0.6),
    ),
    Criterion(
        "operational_burden",
        "Operational burden",
        "Ongoing time spent keeping it alive.",
        "computed",
        1.0,
        _w(simplicity=2.5, control=0.5),
    ),
)

BUILD_VS_BUY_CRITERIA: tuple[Criterion, ...] = (
    Criterion(
        "total_cost_12m",
        "12-month cost",
        "All-in cost over the first year.",
        "computed",
        1.0,
        _w(cost=3.0),
        unit="USD",
    ),
    Criterion(
        "total_cost_36m",
        "36-month cost",
        "All-in cost over three years.",
        "computed",
        1.0,
        _w(cost=2.0, scale=1.5),
        unit="USD",
    ),
    Criterion(
        "time_to_value",
        "Time to value",
        "How soon this is in production.",
        "computed",
        1.0,
        _w(speed=3.0, simplicity=1.5),
        unit="months",
    ),
    Criterion(
        "control",
        "Control and customisation",
        "How much of the behaviour you can change.",
        "computed",
        1.0,
        _w(control=3.0),
    ),
    Criterion(
        "risk",
        "Delivery risk",
        "Odds of it not landing as planned.",
        "computed",
        1.0,
        _w(simplicity=1.5, speed=1.2),
    ),
    Criterion(
        "maintenance",
        "Ongoing maintenance",
        "Engineering time consumed after launch.",
        "computed",
        1.0,
        _w(cost=1.5, simplicity=2.0),
    ),
)

CRITERIA_BY_TOOL: dict[str, tuple[Criterion, ...]] = {
    "compare-models": MODEL_CRITERIA,
    "compare-vector-db": VECTOR_DB_CRITERIA,
    "compare-stacks": STACK_CRITERIA,
    "compare-build-vs-buy": BUILD_VS_BUY_CRITERIA,
}


class StackArchetype(NamedTuple):
    key: str
    name: str
    description: str
    components: tuple[str, ...]
    infra_monthly: float
    setup_days: int
    scaling_ceiling: int  # 1-5
    lock_in: int  # 1-5, higher is worse
    team_skill: int  # 1-5, higher means more senior
    ops_burden: int  # 1-5, higher is worse


STACK_ARCHETYPES: tuple[StackArchetype, ...] = (
    StackArchetype(
        "mvp",
        "MVP",
        "Ship this week. Managed everything, minimal moving parts.",
        ("vercel", "supabase", "pgvector", "openai-api", "langchain"),
        45.0,
        3,
        2,
        3,
        2,
        1,
    ),
    StackArchetype(
        "serverless",
        "Serverless",
        "Scale to zero between requests, pay per invocation.",
        ("cloudflare-workers", "neon", "turbopuffer", "anthropic-api", "inngest"),
        90.0,
        7,
        4,
        4,
        3,
        2,
    ),
    StackArchetype(
        "open-source",
        "Open source",
        "No proprietary dependency in the critical path.",
        ("kubernetes", "postgresql", "qdrant", "vllm", "langgraph", "langfuse"),
        620.0,
        21,
        5,
        1,
        5,
        5,
    ),
    StackArchetype(
        "enterprise",
        "Enterprise",
        "Compliance, audit, and a support contract behind every layer.",
        ("aws-ecs", "postgresql", "elasticsearch", "aws-bedrock", "temporal", "sentry"),
        1850.0,
        35,
        5,
        4,
        4,
        4,
    ),
    StackArchetype(
        "self-hosted",
        "Self-hosted",
        "Runs entirely inside your own network. No data leaves.",
        ("kubernetes", "postgresql", "milvus", "vllm", "ollama", "grafana"),
        2400.0,
        42,
        4,
        1,
        5,
        5,
    ),
)

STACK_ARCHETYPES_BY_KEY = {archetype.key: archetype for archetype in STACK_ARCHETYPES}
