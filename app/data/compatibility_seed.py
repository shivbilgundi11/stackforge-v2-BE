"""The compatibility matrix.

Nine sub-scores per pair, from `PRD.md` §8.1: integration maturity,
documentation quality, deployment complexity, community adoption, cost risk,
vendor lock-in, security readiness, scalability, developer experience.

**Pairs are derived, then overridden.** Two hundred hand-written rows would be
two hundred rows that drift out of agreement with the tool catalog the first
time a maturity score changes. Instead the baseline is computed from each
tool's own catalog facts — so a tool marked `deprecated` immediately drags down
every pair it appears in — and editorial judgement is applied on top as
explicit overrides for the pairs where the general rule is wrong.

The override table is where the actual expertise lives: "LangChain and
LlamaIndex both work, but running both means two chunking implementations and
two retriever abstractions" is not derivable from any score.

Ordering: pairs are emitted with `tool_a < tool_b` lexicographically, matching
the check constraint on the table. Overrides may be written in either order.
"""

from __future__ import annotations

from typing import NamedTuple

from app.data.tools_seed import TOOLS, ToolSeed

DIMENSIONS: tuple[str, ...] = (
    "integration_maturity",
    "documentation_quality",
    "deployment_complexity",
    "community_adoption",
    "cost_risk",
    "vendor_lock_in",
    "security_readiness",
    "scalability",
    "developer_experience",
)

# Which category pairings are worth scoring. A vector DB and an LLM provider
# appear in the same stack constantly; two caches do not. Scoring every
# possible pair would produce thousands of rows nobody queries and bury the
# ~200 that matter.
COMBINABLE: frozenset[frozenset[str]] = frozenset(
    frozenset(pair)
    for pair in (
        ("vector-db", "llm-provider"),
        ("vector-db", "rag-framework"),
        ("vector-db", "agent-framework"),
        ("vector-db", "deployment"),
        ("llm-provider", "agent-framework"),
        ("llm-provider", "rag-framework"),
        ("llm-provider", "observability"),
        ("llm-provider", "orchestration"),
        ("llm-provider", "deployment"),
        ("agent-framework", "orchestration"),
        ("agent-framework", "observability"),
        ("agent-framework", "deployment"),
        ("rag-framework", "observability"),
        ("rag-framework", "orchestration"),
        ("rag-framework", "database"),
        ("orchestration", "deployment"),
        ("orchestration", "database"),
        ("observability", "deployment"),
        ("database", "deployment"),
        ("database", "cache"),
        ("deployment", "cache"),
        ("vector-db", "database"),
        ("agent-framework", "cache"),
        ("llm-provider", "cache"),
    )
)


class CompatPair(NamedTuple):
    tool_a: str
    tool_b: str
    score: int
    dimensions: dict[str, int]
    notes: str | None
    warnings: tuple[str, ...]


class Override(NamedTuple):
    """Editorial judgement applied on top of the derived baseline.

    `score` replaces the computed score outright when set; `notes` and
    `warnings` are additive. An override with no score is a pair that scores
    correctly but needs something said about it.
    """

    score: int | None = None
    notes: str | None = None
    warnings: tuple[str, ...] = ()


def _clamp(value: float) -> int:
    return max(0, min(100, round(value)))


def _derive(a: ToolSeed, b: ToolSeed) -> dict[str, int]:
    """The baseline nine, from catalog facts.

    Each dimension is a defensible function of properties the catalog already
    tracks. Where a dimension is a *risk* (cost, lock-in, deployment burden)
    the score is inverted so that on every axis, higher is better — a matrix
    where some columns mean the opposite of others is unreadable.
    """
    fa, fb = a.facts, b.facts
    maturity = (a.maturity + b.maturity) / 2

    # A pair is only as integrated as its weaker half, with a bonus when both
    # sides are broadly adopted and integrations therefore already exist.
    weakest = min(a.maturity, b.maturity)
    ecosystem = min(int(fa["ecosystem"]), int(fb["ecosystem"]))

    ops = max(int(fa["ops_burden"]), int(fb["ops_burden"]))
    lock_in = max(int(fa["lock_in"]), int(fb["lock_in"]))
    scale = min(int(fa["scale_ceiling"]), int(fb["scale_ceiling"]))

    # Two self-hosted components mean two things to operate, and the burden
    # compounds rather than averaging.
    both_self_hosted = not fa["managed"] and not fb["managed"]

    return {
        "integration_maturity": _clamp(weakest * 0.7 + ecosystem * 6),
        "documentation_quality": _clamp(maturity * 0.9 + ecosystem * 2),
        "deployment_complexity": _clamp(100 - ops * 15 - (10 if both_self_hosted else 0)),
        "community_adoption": _clamp(ecosystem * 18 + maturity * 0.1),
        # Usage-based on both sides is the combination that produces surprise
        # invoices; a free or flat component on either side caps the exposure.
        "cost_risk": _clamp(
            100
            - (25 if a.pricing_model == "usage-based" else 0)
            - (25 if b.pricing_model == "usage-based" else 0)
        ),
        "vendor_lock_in": _clamp(100 - lock_in * 18),
        "security_readiness": _clamp(maturity * 0.85 + (10 if a.self_hostable else 0)),
        "scalability": _clamp(scale * 19),
        "developer_experience": _clamp(maturity * 0.6 + ecosystem * 7 - ops * 3),
    }


def _status_penalty(tool: ToolSeed) -> int:
    """A pair is not usable just because both halves integrate cleanly."""
    return {
        "recommended": 0,
        "stable": 0,
        "caution": 12,
        "deprecated": 35,
        "not_for_production": 45,
    }[tool.status]


# Editorial overrides. Written in either slug order.
OVERRIDES: dict[tuple[str, str], Override] = {
    ("langchain", "llamaindex"): Override(
        score=52,
        notes=(
            "Both work, and plenty of codebases contain both — usually by accident. "
            "Running the two side by side means two chunking implementations, two "
            "retriever abstractions, and two upgrade cadences for one job. Pick the "
            "one whose retrieval model you prefer and use its integrations."
        ),
        warnings=("Overlapping responsibilities — prefer one as the primary framework",),
    ),
    ("pgvector", "postgresql"): Override(
        score=98,
        notes=(
            "pgvector is a Postgres extension, so this is not really an integration: "
            "one datastore, one backup story, one set of credentials, and filters are "
            "ordinary SQL WHERE clauses. Below roughly 5M vectors this beats every "
            "dedicated vector database on total operational cost."
        ),
    ),
    ("pgvector", "supabase"): Override(
        score=95,
        notes="pgvector ships enabled on Supabase — no extension install, no separate service.",
    ),
    ("neon", "pgvector"): Override(
        score=93,
        notes=(
            "pgvector on Neon gets database branching, so an index rebuild can be "
            "tested on a branch before it touches production."
        ),
    ),
    ("anthropic-api", "langgraph"): Override(
        score=92,
        notes=(
            "First-class support with prompt caching and interleaved thinking across "
            "tool calls. LangGraph's checkpointing pairs well with Claude's longer "
            "autonomous runs."
        ),
    ),
    ("claude-agent-sdk", "openai-api"): Override(
        score=25,
        notes=(
            "The Claude Agent SDK is Claude Code as a library — it targets the "
            "Anthropic API specifically. Pointing it at OpenAI means replacing the "
            "model layer, which is most of what the SDK provides."
        ),
        warnings=("Not a supported combination — the SDK is Anthropic-specific",),
    ),
    ("temporal", "langgraph"): Override(
        score=88,
        notes=(
            "Two durability layers that compose rather than conflict: Temporal owns "
            "the retry and recovery boundary, LangGraph owns the agent's own state "
            "graph. Run each LangGraph invocation as a Temporal activity."
        ),
        warnings=("Keep checkpoint ownership in one place — do not persist the same state twice",),
    ),
    ("chroma", "kubernetes"): Override(
        score=38,
        notes=(
            "Chroma's operational model assumes a single process. Running it on "
            "Kubernetes means solving persistence, replication, and backup yourself, "
            "which is the work a production vector database is supposed to save you."
        ),
        warnings=("Chroma is not designed for multi-replica production deployment",),
    ),
    ("faiss", "kubernetes"): Override(
        score=45,
        notes=(
            "FAISS is a library, not a service. Deploying it means writing the "
            "service around it — persistence, sharding, updates, and filtering are "
            "all yours."
        ),
        warnings=("No built-in persistence or metadata filtering — expect real engineering",),
    ),
    ("modal", "vllm"): Override(
        score=90,
        notes=(
            "A common self-hosted inference shape: vLLM for throughput, Modal for "
            "GPU scheduling and scale-to-zero, so idle time is not billed."
        ),
    ),
    ("kubernetes", "vllm"): Override(
        score=85,
        notes=(
            "The standard production self-hosting path. Budget for GPU node pools, "
            "autoscaling on queue depth rather than CPU, and model weights in a "
            "shared volume so pods do not each pull 140GB."
        ),
    ),
    ("cloudflare-workers", "vllm"): Override(
        score=15,
        notes="Workers have no GPU and a constrained runtime. vLLM cannot run there.",
        warnings=("Incompatible — Workers cannot host GPU inference",),
    ),
    ("cloudflare-workers", "postgresql"): Override(
        score=62,
        notes=(
            "Workable via Hyperdrive or a HTTP-based driver, but the classic TCP "
            "Postgres driver does not run in a V8 isolate. Plan the connection layer "
            "up front rather than discovering it at deploy time."
        ),
        warnings=("Requires Hyperdrive or an HTTP driver — no raw TCP connections",),
    ),
    ("langfuse", "langgraph"): Override(
        score=90,
        notes="OpenTelemetry-based tracing that captures each graph node as a span.",
    ),
    ("langfuse", "postgresql"): Override(
        score=92,
        notes="Self-hosted Langfuse runs on Postgres and ClickHouse; you likely already run one.",
    ),
    ("autogpt", "kubernetes"): Override(
        score=20,
        warnings=(
            "AutoGPT has no cost ceiling and no durable state — do not give it a "
            "cluster and a budget",
        ),
    ),
    ("gptcache", "redis"): Override(
        score=30,
        notes=(
            "GPTCache can use Redis as a backend, but provider-native prompt caching "
            "now does this correctly at the API layer. A semantic cache returns an "
            "answer to a question the user did not ask."
        ),
        warnings=("Prefer provider prompt caching — semantic caching risks wrong answers",),
    ),
    ("redis", "upstash"): Override(
        score=70,
        notes=(
            "Upstash speaks the Redis protocol, so client code ports directly. "
            "Per-request pricing wins for spiky serverless traffic and loses badly "
            "for a chatty always-on service."
        ),
    ),
    ("elasticsearch", "kubernetes"): Override(
        score=68,
        notes=(
            "ECK makes this tractable, but an Elasticsearch cluster is a system with "
            "its own capacity planning. Do not adopt it purely for vector search."
        ),
        warnings=("Significant operational commitment — needs a dedicated owner",),
    ),
    ("milvus", "kubernetes"): Override(
        score=80,
        notes=(
            "Milvus is designed for Kubernetes and its operator is the supported "
            "path. Justified above roughly 50M vectors; over-engineered below that."
        ),
    ),
    ("pinecone", "vercel"): Override(
        score=94,
        notes=(
            "Two managed services with HTTP APIs and no connection pooling to think "
            "about — the lowest-friction RAG deployment available, and priced "
            "accordingly."
        ),
    ),
    ("ollama", "vercel"): Override(
        score=10,
        notes="Ollama runs models locally. Vercel functions have no GPU and no persistent host.",
        warnings=("Incompatible — Ollama is a local runtime",),
    ),
    ("crewai", "temporal"): Override(
        score=48,
        notes=(
            "Temporal wants deterministic, replayable workflow code. CrewAI's role "
            "abstraction hides where the non-determinism is, so activity boundaries "
            "are hard to place correctly."
        ),
        warnings=("Determinism boundaries are unclear — wrap whole crews as single activities",),
    ),
    ("autogen", "langgraph"): Override(
        score=40,
        notes="Two agent runtimes. Pick one; running both means two state models.",
        warnings=("Overlapping responsibilities",),
    ),
    ("heroku", "postgresql"): Override(
        score=72,
        notes=(
            "Heroku Postgres is mature and well-run. The reservation is Heroku's "
            "pricing rather than the database."
        ),
    ),
    ("qdrant", "railway"): Override(
        score=76,
        notes=(
            "Qdrant deploys cleanly as a Railway container. Attach a volume — the "
            "default ephemeral filesystem loses the index on redeploy."
        ),
        warnings=("Attach a persistent volume or the index is lost on every deploy",),
    ),
    ("duckdb", "kubernetes"): Override(
        score=40,
        notes=(
            "DuckDB is in-process and single-writer. On Kubernetes each replica gets "
            "its own database, which is almost never what was intended."
        ),
        warnings=("In-process and single-writer — does not share state across replicas",),
    ),
    ("sqlite", "kubernetes"): Override(
        score=35,
        notes="Same shape as DuckDB: a file per pod, and no shared state between replicas.",
        warnings=("A file-per-pod database — use Postgres for multi-replica services",),
    ),
    ("openai-api", "vllm"): Override(
        score=82,
        notes=(
            "vLLM serves an OpenAI-compatible endpoint, so the same client library "
            "points at either. This is the cheapest hedge against provider lock-in "
            "available: one base-URL change."
        ),
    ),
    ("groq", "openrouter"): Override(
        score=78,
        notes="OpenRouter can route to Groq, which buys failover without a second integration.",
    ),
    ("clickhouse", "langfuse"): Override(
        score=88,
        notes="Langfuse uses ClickHouse for trace storage — the intended pairing at volume.",
    ),
    ("opentelemetry", "langfuse"): Override(
        score=90,
        notes=(
            "Langfuse ingests OTLP, so instrument once with OpenTelemetry and keep "
            "the option to move to another backend later."
        ),
    ),
    ("prometheus", "grafana"): Override(
        score=97,
        notes="The default pairing. Grafana's Prometheus support is its most complete.",
    ),
}


def _normalise(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a < b else (b, a)


_OVERRIDES_NORMALISED: dict[tuple[str, str], Override] = {
    _normalise(*key): value for key, value in OVERRIDES.items()
}


def build_pairs() -> list[CompatPair]:
    """Every scored pair, deterministically.

    Deterministic matters: the seed runs on every deploy, and a matrix that
    shuffled between runs would make "did this change?" unanswerable.
    """
    by_slug = {tool.slug: tool for tool in TOOLS}
    slugs = sorted(by_slug)
    pairs: list[CompatPair] = []

    for index, slug_a in enumerate(slugs):
        for slug_b in slugs[index + 1 :]:
            a, b = by_slug[slug_a], by_slug[slug_b]
            override = _OVERRIDES_NORMALISED.get((slug_a, slug_b))

            # Non-combinable categories are skipped unless an editor has
            # explicitly said something about the pair — which is itself the
            # signal that it is worth storing.
            if override is None and frozenset((a.category, b.category)) not in COMBINABLE:
                continue

            dimensions = _derive(a, b)
            baseline = sum(dimensions.values()) / len(dimensions)
            baseline -= _status_penalty(a) + _status_penalty(b)
            score = _clamp(baseline)

            warnings = list(override.warnings) if override else []
            for tool in (a, b):
                if tool.status in ("deprecated", "not_for_production") and tool.status_reason:
                    warnings.append(f"{tool.name}: {tool.status_reason}")

            pairs.append(
                CompatPair(
                    tool_a=slug_a,
                    tool_b=slug_b,
                    score=override.score if override and override.score is not None else score,
                    dimensions=dimensions,
                    notes=override.notes if override else None,
                    warnings=tuple(warnings),
                )
            )

    return pairs
