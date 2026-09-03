"""The RAG architecture rule engine.

Rules first, AI second (D-06). The engine picks components out of the tool
catalog by hard constraint, and only the selection it produces is handed to a
model to write prose about. With no AI available the tool returns the same
components, the same diagram, and a rule-written summary marked
`rule_based` - a degraded explanation of a correct answer, never a degraded
answer.

The constraints are hard filters, not preferences, because that is what they
are in reality: a team that cannot send data to a third party cannot send data
to a third party, and a recommendation that ranks a managed store slightly
lower is no use to them.
"""

from __future__ import annotations

from typing import Any, Final

from app.schemas.catalog import ModelOut, ToolOut
from app.schemas.tools import Artifact, ToolOutput, ToolWarning
from app.services import diagram_theme

# Cross-encoder reranking is a second model pass over every candidate. It buys
# real precision and costs real milliseconds, so it comes out below this.
CROSS_ENCODER_BUDGET_MS: Final = 500

STAGES: Final = ("loader", "splitter", "embedder", "store", "retriever", "reranker", "generator")


# Statuses fit to recommend in a fresh design. `caution` and below are
# excluded outright rather than ranked down: the graveyard exists so people
# stop reaching for these, and putting one in a new architecture undoes that
# in a single screen.
RECOMMENDABLE: Final = frozenset({"recommended", "stable"})


def _pick(
    candidates: list[ToolOut],
    *,
    self_host_only: bool,
    prefer: list[str] | None = None,
) -> ToolOut | None:
    """Highest-maturity recommendable tool, preferring named slugs.

    The self-hosting filter reads `self_hostable`, not `managed`. They are
    different claims: `managed` means a hosted offering exists, which is true
    of pgvector and Qdrant and says nothing about whether you can run them
    yourself. Filtering on `managed` excluded every store that had a cloud
    option and left restricted deployments with nothing at all.
    """
    viable = [tool for tool in candidates if tool.status in RECOMMENDABLE]
    if self_host_only:
        viable = [tool for tool in viable if tool.self_hostable]

    for slug in prefer or []:
        for tool in viable:
            if tool.slug == slug:
                return tool

    viable.sort(key=lambda tool: tool.maturity_score, reverse=True)
    return viable[0] if viable else None


def _pick_reranker(models: list[ModelOut]) -> ModelOut | None:
    """Cheapest active reranker, comparing like with like.

    Only per-token models are ranked on price here. A per-search price and a
    per-token price are not comparable without knowing the query shape (D-18),
    and picking the numerically smaller of the two would be the exact error
    the price unit exists to prevent.
    """
    active = [model for model in models if model.status == "active"]
    if not active:
        return None

    per_token = [model for model in active if model.price_unit == "tokens"]
    pool = per_token or active
    return min(pool, key=lambda model: model.input_cost_per_1k)


def design(
    *,
    use_case: str,
    corpus_documents: int,
    sensitivity: str,
    latency_target_ms: int,
    scale: str,
    team_skill: str,
    catalog: list[ToolOut],
    rerank_models: list[ModelOut] | None = None,
) -> ToolOutput:
    """Select a pipeline, diagram it, and say why each choice was made."""
    restricted = sensitivity == "restricted"
    self_host_only = restricted or sensitivity == "internal-only"

    by_category: dict[str, list[ToolOut]] = {}
    for tool in catalog:
        by_category.setdefault(tool.category, []).append(tool)

    store = _pick(
        by_category.get("vector-db", []),
        self_host_only=self_host_only,
        prefer=["pgvector", "qdrant"] if self_host_only else ["pinecone", "qdrant"],
    )
    framework = _pick(
        by_category.get("rag-framework", []),
        self_host_only=False,
        prefer=["llamaindex", "langchain"],
    )

    # Latency is the reason a reranker gets dropped, not cost. A cross-encoder
    # over 20 candidates is a second forward pass, and under half a second
    # end-to-end there is no room for it.
    # Rerankers are models, not tools: they live in `model_pricing` under
    # family `rerank`, not in `tool_catalog`. Looking them up by tool category
    # returns nothing and silently drops the stage from every pipeline.
    use_reranker = latency_target_ms >= CROSS_ENCODER_BUDGET_MS
    reranker = _pick_reranker(rerank_models or []) if use_reranker else None

    warnings: list[ToolWarning] = []
    if restricted:
        warnings.append(
            ToolWarning(
                level="info",
                message=(
                    "Sensitivity is restricted, so managed vector stores are excluded "
                    "outright rather than ranked lower. Data that cannot leave your "
                    "network cannot leave your network."
                ),
            )
        )
    if not use_reranker:
        warnings.append(
            ToolWarning(
                level="warning",
                message=(
                    f"A {latency_target_ms}ms budget has no room for cross-encoder "
                    "reranking, which is a second model pass over every candidate. "
                    "Expect lower precision, and consider retrieving fewer, larger chunks."
                ),
            )
        )
    if store is None:
        warnings.append(
            ToolWarning(
                level="critical",
                message=(
                    "No vector store in the catalog satisfies these constraints. "
                    "Self-hosting pgvector alongside an existing Postgres is usually "
                    "the fallback."
                ),
            )
        )
    if corpus_documents > 5_000_000 and store and (store.facts or {}).get("scale_ceiling"):
        warnings.append(
            ToolWarning(
                level="warning",
                message=(
                    f"{corpus_documents:,} documents is near or past the comfortable "
                    f"ceiling for {store.name}. Check the sizing tool before committing."
                ),
            )
        )

    splitter = "Markdown-aware" if use_case in {"docs", "code"} else "Recursive character"
    components = [
        {
            "stage": "loader",
            "choice": framework.name if framework else "Custom loader",
            "why": "Handles format detection and extraction across a mixed corpus.",
        },
        {
            "stage": "splitter",
            "choice": splitter,
            "why": (
                "Keeps headings with their content."
                if splitter.startswith("Markdown")
                else "Splits on the largest natural boundary that fits."
            ),
        },
        {
            "stage": "embedder",
            "choice": "Self-hosted BGE / E5" if restricted else "Managed embedding API",
            "why": (
                "Restricted data cannot be sent to a third-party embedding endpoint."
                if restricted
                else "Managed embeddings avoid running and scaling a model for a cheap operation."
            ),
        },
        {
            "stage": "store",
            "choice": store.name if store else "None available",
            "why": _store_reason(store, self_host_only=self_host_only, scale=scale),
        },
        {
            "stage": "retriever",
            "choice": "Hybrid (dense + BM25)" if scale != "small" else "Dense similarity",
            "why": (
                "Hybrid recovers exact-match queries that pure vectors miss - names, "
                "error codes, SKUs."
                if scale != "small"
                else "At this corpus size the added complexity of hybrid search is not repaid."
            ),
        },
        {
            "stage": "reranker",
            "choice": reranker.display_name if reranker else "None (latency budget)",
            "why": (
                "Cross-encoder reranking is the single largest precision gain available."
                if reranker
                else f"Excluded: a {latency_target_ms}ms budget cannot absorb a second pass."
            ),
        },
        {
            "stage": "generator",
            "choice": "Self-hosted open model" if restricted else "Managed LLM API",
            "why": (
                "Same constraint as the embedder."
                if restricted
                else "Quality per unit cost is far better on a managed frontier model."
            ),
        },
    ]

    diagram = _mermaid(components, framework=framework, store=store)
    summary = _summary(
        use_case=use_case,
        sensitivity=sensitivity,
        components=components,
        reranker=reranker is not None,
        team_skill=team_skill,
    )

    return ToolOutput(
        metrics={
            "store": store.slug if store else "none",
            "reranking": "yes" if reranker else "no",
            "self_hosted": "yes" if restricted else "no",
            "stages": len(components),
            "summary": summary,
        },
        tables={"components": components},
        artifacts=[
            Artifact(
                type="diagram",
                format="mermaid",
                filename="rag-pipeline.mmd",
                content=diagram,
            ),
            Artifact(
                type="architecture",
                format="markdown",
                filename="rag-architecture.md",
                content=_document(components, diagram, summary, warnings),
            ),
        ],
        warnings=warnings,
        sourced_from=[tool.slug for tool in (store, framework) if tool is not None]
        + ([reranker.id] if reranker is not None else []),
    )


def _store_reason(store: ToolOut | None, *, self_host_only: bool, scale: str) -> str:
    if store is None:
        return "No catalog entry satisfies the sensitivity constraint."
    if self_host_only:
        return f"{store.name} runs inside your own network, which the sensitivity setting requires."
    if scale in {"large", "xlarge"}:
        return f"{store.name} is the highest-maturity managed option at this scale."
    return f"{store.name} is the lowest-operations option that fits this corpus."


def _mermaid(
    components: list[dict[str, Any]],
    *,
    framework: ToolOut | None = None,
    store: ToolOut | None = None,
) -> str:
    """A left-to-right pipeline.

    Node ids are the stage names, which are a fixed vocabulary, so nothing
    user-supplied reaches an identifier position. Labels are quoted and have
    quotes stripped - a tool name with a bracket in it otherwise produces a
    diagram that silently fails to parse in the renderer.

    `framework` and `store` are passed rather than read back out of
    `components`, because only two of the seven stages are a catalog entry at
    all - a splitter is a strategy and an embedder is a class of model. The
    rest are coloured by stage and get a monogram, which is the honest picture:
    there is no logo for "recursive character".
    """
    lines = ["graph LR"]
    roles: dict[str, str] = {}
    slugs = {
        "loader": framework.slug if framework else None,
        "store": store.slug if store else None,
    }

    for index, component in enumerate(components):
        stage = component["stage"]
        label = str(component["choice"]).replace('"', "").replace("\n", " ")
        lines.append(f'    {stage}["{stage.title()}<br/>{label}"]')
        if index:
            lines.append(f"    {components[index - 1]['stage']} --> {stage}")
        roles[stage] = stage

    tools: dict[str, str | None] = {stage: slugs.get(stage) for stage in roles}
    return "\n".join(diagram_theme.decorate(lines, roles=roles, tools=tools))


def _summary(
    *,
    use_case: str,
    sensitivity: str,
    components: list[dict[str, Any]],
    reranker: bool,
    team_skill: str,
) -> str:
    """The rule-written rationale.

    This is what ships when AI is unavailable, so it has to read as a real
    explanation rather than a placeholder. M16 replaces it with a synthesised
    version; it does not replace the components, which are already correct.
    """
    store = next(c["choice"] for c in components if c["stage"] == "store")
    retriever = next(c["choice"] for c in components if c["stage"] == "retriever")

    parts = [
        f"A {use_case} pipeline over {sensitivity} data, built on {store} "
        f"with {retriever.lower()} retrieval.",
    ]
    if reranker:
        parts.append(
            "A reranking pass sits between retrieval and generation, which is where "
            "most of the precision in this design comes from."
        )
    else:
        parts.append(
            "There is no reranking pass: the latency budget does not allow one, so "
            "retrieval quality rests entirely on chunking and the embedding model."
        )
    if team_skill == "beginner":
        parts.append(
            "Given the stated team experience, prefer the managed option at every "
            "stage where one exists. Operating a vector store is a distraction from "
            "the retrieval quality work that actually determines whether this succeeds."
        )
    return " ".join(parts)


def _document(
    components: list[dict[str, Any]],
    diagram: str,
    summary: str,
    warnings: list[ToolWarning],
) -> str:
    rows = "\n".join(f"| {c['stage'].title()} | {c['choice']} | {c['why']} |" for c in components)
    risks = "\n".join(f"- {w.message}" for w in warnings) or "- None flagged."

    return f"""# RAG pipeline architecture

{summary}

## Pipeline

```mermaid
{diagram}
```

## Components

| Stage | Choice | Why |
| --- | --- | --- |
{rows}

## Risks and constraints

{risks}

---

Generated by StackForge. Component selection is rule-based against the tool
catalog; every choice above is traceable to a stated constraint.
"""
