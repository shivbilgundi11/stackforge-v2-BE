""".cursorrules for the chosen stack.

A small artifact with outsized pull: it is the one thing this product makes
that lands directly in someone's editor and changes what they type next.

The content is worth being careful about. A rules file that says "use best
practices" is noise an assistant will ignore; one that pins the actual
libraries, names the actual versions of the decisions, and states the gotchas
of *these* components is a file that changes the completions. So every line
here is either a fact from the catalog or a rule derived from a stated
requirement — nothing is generic advice.
"""

from __future__ import annotations

from typing import Final

from app.core.database import utcnow
from app.data import disclaimers
from app.schemas.catalog import ToolOut
from app.schemas.tools import Artifact
from app.services.artifacts.sources import Source, StackSource

TYPE = "cursor-rules"

#: Component-specific gotchas. Each one is a mistake that is easy to make, hard
#: to see, and cheap to prevent with a sentence in front of the assistant.
GOTCHAS: Final[dict[str, str]] = {
    "pgvector": (
        "Create the HNSW or IVFFlat index explicitly — pgvector does a sequential "
        "scan without one and stays fast enough in development to hide it."
    ),
    "qdrant": (
        "Use named vectors from the start. Adding a second embedding model to a "
        "collection created with an unnamed vector means recreating the collection."
    ),
    "weaviate": (
        "Set the vectorizer to `none` and supply vectors yourself unless you "
        "genuinely want Weaviate calling an embedding provider on your behalf."
    ),
    "pinecone": (
        "Namespaces, not separate indexes, for multi-tenancy. An index per tenant "
        "hits the account index limit and costs a minimum each."
    ),
    "chroma": (
        "Chroma runs embedded by default. Point it at a persistent directory or "
        "the collection disappears with the process."
    ),
    "elasticsearch": (
        "Dense-vector fields need `index: true` and an explicit similarity. The "
        "default mapping stores the vector and will not search it."
    ),
    "redis": (
        "Set an eviction policy explicitly. The default `noeviction` turns a full "
        "cache into write errors on the application path."
    ),
    "postgresql": (
        "Connection pooling belongs in front of Postgres, not inside every worker. "
        "Serverless functions without a pooler exhaust connections quickly."
    ),
    "langchain": (
        "Pin the version and the integration packages together. LangChain's "
        "integrations move independently and a minor bump changes import paths."
    ),
    "llamaindex": (
        "Set the embedding model and the LLM on the Settings object once. Passing "
        "them per-call is where two different embedding models end up in one index."
    ),
    "langgraph": (
        "Checkpoint the graph state to durable storage before running anything "
        "long. An in-memory checkpointer loses the run on restart."
    ),
    "anthropic-api": (
        "Prompt caching is a cache_control block on the content, not a parameter. "
        "Set `max_tokens` deliberately — it is required and it bounds the bill."
    ),
    "openai-api": (
        "Structured output belongs in `response_format`, not in a prompt asking "
        "for JSON. The API guarantees the schema; the prompt does not."
    ),
    "ollama": (
        "Ollama's OpenAI-compatible endpoint is at /v1 and does not implement "
        "every parameter. Check before assuming a client works unchanged."
    ),
    "vllm": (
        "vLLM preallocates GPU memory from `gpu_memory_utilization`. The default "
        "of 0.9 leaves no room for anything else on the card."
    ),
    "temporal": (
        "Workflow code must be deterministic — no clock reads, no random, no "
        "direct IO. Everything non-deterministic goes in an activity."
    ),
    "celery": (
        "Set `acks_late` and a visibility timeout longer than the slowest task, or "
        "long tasks are redelivered while still running."
    ),
    "langfuse": (
        "Flush traces on shutdown. The SDK batches, and a short-lived process "
        "exits with the last spans still in the buffer."
    ),
}


def supports(source: Source) -> bool:
    return isinstance(source, StackSource)


def generate(source: StackSource) -> Artifact:
    components = "\n".join(
        f"- **{tool.name}** ({tool.category.replace('-', ' ')}) — {tool.description}"
        for tool in source.components
    )
    gotchas = _gotchas(source.components)
    constraints = _constraints(source)
    avoid = _avoid(source)

    return Artifact(
        type=TYPE,
        format="text",
        filename=".cursorrules",
        content=f"""{disclaimers.file_header(utcnow().date())}

# {source.title}

This project is built on a fixed stack. Do not introduce an alternative to any
component listed below without being asked — a suggestion that swaps the vector
store is a suggestion to rewrite the retrieval layer.

## The stack

{components}

## Constraints that decided this stack

{constraints}

## Component rules

{gotchas}

## Do not

{avoid}

## General

- Prefer the library already in the stack over adding one. If something is
  missing, say what is missing rather than reaching for a new dependency.
- Every price, limit, and model name changes. Do not hardcode one in more than
  one place, and do not assert one from memory — read it from configuration.
- Async all the way down or sync all the way down. A blocking call inside an
  async handler stalls the whole event loop and is invisible until load.

---
Generated by StackForge from the saved stack "{source.title}" (version
{source.version}). Regenerate this file when the stack changes; a rules file
describing a stack you no longer run is worse than none.
""",
        language="markdown",
    )


def _gotchas(components: list[ToolOut]) -> str:
    lines = [
        f"- **{tool.name}**: {GOTCHAS[tool.slug]}" for tool in components if tool.slug in GOTCHAS
    ]
    if not lines:
        return (
            "_No component-specific rules are recorded for this stack. That means "
            "nobody has written them down yet, not that there are none._"
        )
    return "\n".join(lines)


def _constraints(source: StackSource) -> str:
    requirements = source.requirements
    lines = [
        f"- **Use case**: {requirements.use_case}. Optimise for this shape of work.",
        f"- **Scale target**: {requirements.scale_target}.",
        f"- **Latency budget**: {requirements.latency_ms} ms end to end. Anything "
        f"synchronous on the request path has to fit inside it.",
        f"- **Data sensitivity**: {requirements.sensitivity}.",
        f"- **Deployment**: {requirements.deployment}.",
        f"- **Budget**: ${requirements.monthly_budget:,}/month.",
    ]
    if requirements.sensitivity in {"restricted", "regulated"}:
        lines.append(
            "- **Data must not leave the network.** Do not suggest a managed API "
            "for anything that touches user data, including for a quick test."
        )
    if requirements.team_skill == "beginner":
        lines.append(
            "- The team is new to this. Prefer the boring, documented path over "
            "the clever one, and explain the trade when you take it."
        )
    return "\n".join(lines)


def _avoid(source: StackSource) -> str:
    lines: list[str] = []
    for tool in source.deprecated:
        lines.append(
            f"- Do not build further on **{tool.name}** — the catalog marks it "
            f"{tool.status}: {tool.status_reason or 'no reason recorded'}."
            + (f" Alternatives: {', '.join(tool.alternatives)}." if tool.alternatives else "")
        )
    if source.requirements.latency_ms < 500:
        lines.append(
            "- Do not add a batch-oriented component to the request path. The "
            "latency budget has no room for one."
        )
    lines.append(
        "- Do not invent pricing, rate limits, or context-window figures. If a "
        "number is needed and not in the code, say so instead of guessing."
    )
    return "\n".join(lines)
