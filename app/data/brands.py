"""Catalog slug to brand mark.

A diagram of eight grey rectangles asks the reader to do the sorting the
picture was meant to do. A logo is the fastest possible answer to "what is
that box" — faster than the label, which is why it goes on the box at all.

## Why this is a map and not a catalog column

The obvious home is `tool_catalog`, and the repo's rule is that adding a tool
should be a data change rather than a code change. The rule is right and this
is the exception: the mapping is not a fact about the tool, it is a fact about
which third-party icon set happens to carry it. Seeding eighty-eight rows with
a value most of them cannot have — and re-seeding them every time an icon set
adds or removes a brand — puts editorial churn in the catalog for something
nobody is describing when they add a tool.

## Why it is partial

Forty-seven of eighty-eight. The set this reads from delists a brand on
trademark request, and it has done so for most of the large vendors: AWS,
Azure, OpenAI and Heroku all had marks and no longer do. Everything else is
simply not in it — `langfuse`, `weaviate`, `pinecone`, `dagster` and the rest
are real products with real logos that this particular set does not carry.

An unmatched tool is not a hole. The renderer draws a monogram in the role's
colour instead, which is a deliberate-looking badge rather than a gap, and the
diagram reads the same whether a given box got a logo or a letter.

`hex` is stored without the leading `#` because it travels through a
colon-separated comment in the diagram source, where a `#` would read as the
start of a fragment to anything trying to parse a URL out of it.
"""

from __future__ import annotations

from typing import Final, NamedTuple


class Mark(NamedTuple):
    #: The icon's name in the set the renderers draw from.
    icon: str
    #: The brand colour, `rrggbb`, no leading `#`.
    hex: str


#: Keyed on the catalog slug. Several slugs share a mark on purpose: pgvector
#: *is* PostgreSQL, `redis-vector` is Redis, and the Claude Agent SDK is
#: Anthropic's — showing the parent's logo is more informative than showing
#: nothing, and is what the reader is looking for anyway.
BRANDS: Final[dict[str, Mark]] = {
    # ── Vector stores ───────────────────────────────────────────────────────
    "elasticsearch": Mark("elasticsearch", "005571"),
    "milvus": Mark("milvus", "00A1EA"),
    "qdrant": Mark("qdrant", "DC244C"),
    "pgvector": Mark("postgresql", "4169E1"),
    "redis-vector": Mark("redis", "FF4438"),
    # ── Model providers ─────────────────────────────────────────────────────
    "anthropic-api": Mark("anthropic", "191919"),
    "google-gemini": Mark("googlegemini", "8E75B2"),
    "ollama": Mark("ollama", "000000"),
    "openrouter": Mark("openrouter", "94A3B8"),
    "vllm": Mark("vllm", "30A2FF"),
    # Text Generation Inference is Hugging Face's server, and the logo people
    # recognise for it is theirs.
    "tgi": Mark("huggingface", "FFD21E"),
    # ── Frameworks ──────────────────────────────────────────────────────────
    "crewai": Mark("crewai", "FF5A50"),
    "langgraph": Mark("langgraph", "7FC8FF"),
    "pydantic-ai": Mark("pydantic", "E92063"),
    "smolagents": Mark("huggingface", "FFD21E"),
    "claude-agent-sdk": Mark("anthropic", "191919"),
    "haystack": Mark("haystack", "0EAF9C"),
    "langchain": Mark("langchain", "7FC8FF"),
    # ── Orchestration ───────────────────────────────────────────────────────
    "airflow": Mark("apacheairflow", "017CEE"),
    "celery": Mark("celery", "37814A"),
    "prefect": Mark("prefect", "070E10"),
    "temporal": Mark("temporal", "000000"),
    # ── Observability ───────────────────────────────────────────────────────
    "braintrust": Mark("braintrust", "000000"),
    "grafana": Mark("grafana", "F46800"),
    "opentelemetry": Mark("opentelemetry", "000000"),
    "prometheus": Mark("prometheus", "E6522C"),
    "sentry": Mark("sentry", "362D59"),
    # ── Deployment ──────────────────────────────────────────────────────────
    "cloudflare-workers": Mark("cloudflareworkers", "F38020"),
    "fly-io": Mark("flydotio", "24175B"),
    "kubernetes": Mark("kubernetes", "326CE5"),
    "modal": Mark("modal", "7FEE64"),
    "railway": Mark("railway", "0B0D0E"),
    "render": Mark("render", "000000"),
    "vercel": Mark("vercel", "000000"),
    # ── Databases and caches ────────────────────────────────────────────────
    "clickhouse": Mark("clickhouse", "FFCC01"),
    "duckdb": Mark("duckdb", "FFF000"),
    "mongodb": Mark("mongodb", "47A248"),
    "neon": Mark("neon", "34D59A"),
    "planetscale": Mark("planetscale", "000000"),
    "postgresql": Mark("postgresql", "4169E1"),
    "sqlite": Mark("sqlite", "003B57"),
    "supabase": Mark("supabase", "3FCF8E"),
    "redis": Mark("redis", "FF4438"),
    "upstash": Mark("upstash", "00E9A3"),
    # ── Compute and guardrails ──────────────────────────────────────────────
    "gcp-gpu": Mark("googlecloud", "4285F4"),
    "llama-guard": Mark("meta", "0467DF"),
    "nemo-guardrails": Mark("nvidia", "76B900"),
}


def brand_for(slug: str) -> Mark | None:
    """The mark for a catalog slug, or None to fall back to a monogram."""
    return BRANDS.get(slug)
