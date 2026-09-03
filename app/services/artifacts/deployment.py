"""`docker-compose.yml` and `.env.example`, generated from the stack itself.

Not from an archetype. M13's four archetypes are fixed templates that answer
"what does a vLLM + Qdrant stack look like"; this answers "what does *your*
stack look like", and those are different questions the moment someone picks
Weaviate. A plan bundle whose compose file describes a different stack from the
architecture document beside it is worse than no compose file.

The mapping is deliberately explicit and hand-written. A generated image tag
is a thing someone will run, so every entry here is a real published image with
a real port, and a component with no entry produces a comment saying so rather
than an invented service. Guessing `image: {slug}:latest` would produce a file
that fails at `docker compose up` for reasons the user cannot see.

Every emitted file is dumped from a Python structure through `yaml.safe_dump`,
never interpolated — a model name with a colon in it turns a hand-built YAML
line into a different key that parses successfully as the wrong thing.
"""

from __future__ import annotations

from typing import Any, Final, NamedTuple

from app.schemas.tools import Artifact
from app.services.artifacts.sources import Source, StackSource

# Reused rather than reimplemented: `_service` fixes the restart policy, the
# health-check timings, and the `service_healthy` dependency condition that
# M13 got right. A second copy of those would drift.
from app.services.infra_artifacts import _dump, _service, starter_header

TYPE_COMPOSE = "compose"
TYPE_ENV = "env"


class Runnable(NamedTuple):
    """A catalog component that has a container worth running."""

    service: str
    image: str
    ports: tuple[str, ...] = ()
    environment: dict[str, str] | None = None
    volumes: tuple[str, ...] = ()
    healthcheck: tuple[str, ...] = ()
    command: str | None = None
    gpu: bool = False


#: Self-hostable components, by catalog slug. Every image and port here is a
#: published default; a slug absent from this table is either managed (see
#: `MANAGED` below), an embedded library with nothing to run, or something
#: nobody should be handed a one-line deployment for.
RUNNABLE: Final[dict[str, Runnable]] = {
    # vector stores
    "qdrant": Runnable(
        "qdrant",
        "qdrant/qdrant:latest",
        ("6333:6333",),
        volumes=("qdrant-data:/qdrant/storage",),
        healthcheck=("CMD-SHELL", "bash -c ':> /dev/tcp/127.0.0.1/6333'"),
    ),
    "weaviate": Runnable(
        "weaviate",
        "cr.weaviate.io/semitechnologies/weaviate:1.27.0",
        ("8080:8080",),
        environment={
            "PERSISTENCE_DATA_PATH": "/var/lib/weaviate",
            "AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED": "true",
            "DEFAULT_VECTORIZER_MODULE": "none",
        },
        volumes=("weaviate-data:/var/lib/weaviate",),
        healthcheck=("CMD", "curl", "-f", "http://localhost:8080/v1/.well-known/ready"),
    ),
    "milvus": Runnable(
        "milvus",
        "milvusdb/milvus:v2.4.13",
        ("19530:19530",),
        environment={"ETCD_USE_EMBED": "true", "COMMON_STORAGETYPE": "local"},
        volumes=("milvus-data:/var/lib/milvus",),
        command="milvus run standalone",
        healthcheck=("CMD", "curl", "-f", "http://localhost:9091/healthz"),
    ),
    "chroma": Runnable(
        "chroma",
        "chromadb/chroma:0.5.20",
        ("8000:8000",),
        volumes=("chroma-data:/chroma/chroma",),
        healthcheck=("CMD", "curl", "-f", "http://localhost:8000/api/v1/heartbeat"),
    ),
    "elasticsearch": Runnable(
        "elasticsearch",
        "docker.elastic.co/elasticsearch/elasticsearch:8.15.3",
        ("9200:9200",),
        environment={
            "discovery.type": "single-node",
            "xpack.security.enabled": "false",
            "ES_JAVA_OPTS": "-Xms1g -Xmx1g",
        },
        volumes=("es-data:/usr/share/elasticsearch/data",),
        healthcheck=("CMD-SHELL", "curl -sf http://localhost:9200/_cluster/health"),
    ),
    "redis-vector": Runnable(
        "redis",
        "redis/redis-stack-server:7.4.0-v1",
        ("6379:6379",),
        volumes=("redis-data:/data",),
        healthcheck=("CMD", "redis-cli", "ping"),
    ),
    "pgvector": Runnable(
        "postgres",
        "pgvector/pgvector:pg17",
        ("5432:5432",),
        environment={"POSTGRES_PASSWORD": "${POSTGRES_PASSWORD}", "POSTGRES_DB": "app"},
        volumes=("pgdata:/var/lib/postgresql/data",),
        healthcheck=("CMD-SHELL", "pg_isready -U postgres"),
    ),
    # databases
    "postgresql": Runnable(
        "postgres",
        "postgres:17-alpine",
        ("5432:5432",),
        environment={"POSTGRES_PASSWORD": "${POSTGRES_PASSWORD}", "POSTGRES_DB": "app"},
        volumes=("pgdata:/var/lib/postgresql/data",),
        healthcheck=("CMD-SHELL", "pg_isready -U postgres"),
    ),
    "mongodb": Runnable(
        "mongodb",
        "mongo:7",
        ("27017:27017",),
        environment={
            "MONGO_INITDB_ROOT_USERNAME": "root",
            "MONGO_INITDB_ROOT_PASSWORD": "${MONGO_PASSWORD}",
        },
        volumes=("mongo-data:/data/db",),
        healthcheck=("CMD", "mongosh", "--eval", "db.adminCommand('ping')"),
    ),
    "clickhouse": Runnable(
        "clickhouse",
        "clickhouse/clickhouse-server:24.8-alpine",
        ("8123:8123",),
        volumes=("clickhouse-data:/var/lib/clickhouse",),
        healthcheck=("CMD", "wget", "-qO-", "http://localhost:8123/ping"),
    ),
    # caches
    "redis": Runnable(
        "redis",
        "redis:7-alpine",
        ("6379:6379",),
        volumes=("redis-data:/data",),
        healthcheck=("CMD", "redis-cli", "ping"),
    ),
    "valkey": Runnable(
        "valkey",
        "valkey/valkey:8-alpine",
        ("6379:6379",),
        volumes=("valkey-data:/data",),
        healthcheck=("CMD", "valkey-cli", "ping"),
    ),
    "memcached": Runnable("memcached", "memcached:1.6-alpine", ("11211:11211",)),
    # model servers
    "ollama": Runnable(
        "ollama",
        "ollama/ollama:latest",
        ("11434:11434",),
        volumes=("ollama-models:/root/.ollama",),
        healthcheck=("CMD", "ollama", "list"),
        gpu=True,
    ),
    "vllm": Runnable(
        "vllm",
        "vllm/vllm-openai:latest",
        ("8000:8000",),
        environment={"HUGGING_FACE_HUB_TOKEN": "${HUGGING_FACE_HUB_TOKEN}"},
        volumes=("hf-cache:/root/.cache/huggingface",),
        command="--model ${VLLM_MODEL} --host 0.0.0.0",
        healthcheck=("CMD", "curl", "-f", "http://localhost:8000/health"),
        gpu=True,
    ),
    "tgi": Runnable(
        "tgi",
        "ghcr.io/huggingface/text-generation-inference:3.0.1",
        ("8080:80",),
        environment={"HUGGING_FACE_HUB_TOKEN": "${HUGGING_FACE_HUB_TOKEN}"},
        volumes=("hf-cache:/data",),
        command="--model-id ${TGI_MODEL}",
        healthcheck=("CMD", "curl", "-f", "http://localhost:80/health"),
        gpu=True,
    ),
    # observability
    "langfuse": Runnable(
        "langfuse",
        "langfuse/langfuse:2",
        ("3001:3000",),
        environment={
            "DATABASE_URL": "postgresql://postgres:${POSTGRES_PASSWORD}@postgres:5432/langfuse",
            "NEXTAUTH_SECRET": "${LANGFUSE_NEXTAUTH_SECRET}",
            "SALT": "${LANGFUSE_SALT}",
            "NEXTAUTH_URL": "http://localhost:3001",
        },
        healthcheck=("CMD", "wget", "-qO-", "http://localhost:3000/api/public/health"),
    ),
    "phoenix": Runnable(
        "phoenix",
        "arizephoenix/phoenix:latest",
        ("6006:6006",),
        volumes=("phoenix-data:/mnt/data",),
    ),
    "grafana": Runnable(
        "grafana",
        "grafana/grafana:11.3.0",
        ("3002:3000",),
        environment={"GF_SECURITY_ADMIN_PASSWORD": "${GRAFANA_PASSWORD}"},
        volumes=("grafana-data:/var/lib/grafana",),
    ),
    "prometheus": Runnable(
        "prometheus",
        "prom/prometheus:v2.55.1",
        ("9090:9090",),
        volumes=("prometheus-data:/prometheus",),
    ),
    "opentelemetry": Runnable(
        "otel-collector",
        "otel/opentelemetry-collector-contrib:0.114.0",
        ("4317:4317", "4318:4318"),
    ),
    # orchestration
    "temporal": Runnable(
        "temporal",
        "temporalio/auto-setup:1.25.2",
        ("7233:7233",),
        environment={
            "DB": "postgres12",
            "DB_PORT": "5432",
            "POSTGRES_USER": "postgres",
            "POSTGRES_PWD": "${POSTGRES_PASSWORD}",
            "POSTGRES_SEEDS": "postgres",
        },
    ),
    "prefect": Runnable(
        "prefect",
        "prefecthq/prefect:3-latest",
        ("4200:4200",),
        command="prefect server start --host 0.0.0.0",
        volumes=("prefect-data:/root/.prefect",),
    ),
    "airflow": Runnable(
        "airflow",
        "apache/airflow:2.10.3",
        ("8081:8080",),
        environment={"AIRFLOW__CORE__EXECUTOR": "LocalExecutor"},
        command="standalone",
    ),
}

#: Managed components, and the environment variables their SDKs read. The
#: names are the providers' own — an invented variable name is a file that
#: looks configured and is not.
MANAGED: Final[dict[str, tuple[tuple[str, str], ...]]] = {
    "anthropic-api": (("ANTHROPIC_API_KEY", "sk-ant-change-me"),),
    "openai-api": (("OPENAI_API_KEY", "sk-change-me"),),
    "google-gemini": (("GOOGLE_API_KEY", "change-me"),),
    "aws-bedrock": (
        ("AWS_ACCESS_KEY_ID", "change-me"),
        ("AWS_SECRET_ACCESS_KEY", "change-me"),
        ("AWS_REGION", "us-east-1"),
    ),
    "azure-openai": (
        ("AZURE_OPENAI_API_KEY", "change-me"),
        ("AZURE_OPENAI_ENDPOINT", "https://your-resource.openai.azure.com"),
    ),
    "together-ai": (("TOGETHER_API_KEY", "change-me"),),
    "groq": (("GROQ_API_KEY", "gsk-change-me"),),
    "fireworks-ai": (("FIREWORKS_API_KEY", "change-me"),),
    "openrouter": (("OPENROUTER_API_KEY", "sk-or-change-me"),),
    "pinecone": (("PINECONE_API_KEY", "change-me"), ("PINECONE_INDEX", "app")),
    "turbopuffer": (("TURBOPUFFER_API_KEY", "change-me"),),
    "langsmith": (("LANGCHAIN_API_KEY", "change-me"), ("LANGCHAIN_TRACING_V2", "true")),
    "braintrust": (("BRAINTRUST_API_KEY", "change-me"),),
    "helicone": (("HELICONE_API_KEY", "change-me"),),
    "langfuse": (
        ("LANGFUSE_PUBLIC_KEY", "pk-lf-change-me"),
        ("LANGFUSE_SECRET_KEY", "sk-lf-change-me"),
    ),
    "supabase": (
        ("SUPABASE_URL", "https://your-project.supabase.co"),
        ("SUPABASE_SERVICE_ROLE_KEY", "change-me"),
    ),
    "neon": (("DATABASE_URL", "postgresql://user:password@your-project.neon.tech/app"),),
    "planetscale": (("DATABASE_URL", "mysql://user:password@your-host/app"),),
    "upstash": (
        ("UPSTASH_REDIS_REST_URL", "https://your-db.upstash.io"),
        ("UPSTASH_REDIS_REST_TOKEN", "change-me"),
    ),
    "sentry": (("SENTRY_DSN", "https://change-me@sentry.io/0"),),
}

#: Components with genuinely nothing to run — embedded libraries and things
#: that live inside the application process. Listed so the generated file can
#: say why they are absent rather than leaving a reader to wonder.
EMBEDDED: Final[dict[str, str]] = {
    "faiss": "a library linked into your process, not a service",
    "lancedb": "an embedded store that writes to a local directory",
    "sqlite": "a file on disk",
    "duckdb": "an embedded analytical database",
    "chroma": "runs embedded by default; the service below is the client/server mode",
    "gptcache": "a library that wraps your model client",
    "celery": "workers run your own image — add them alongside the app service",
}


def supports(source: Source) -> bool:
    return isinstance(source, StackSource)


def _runnables(source: StackSource) -> dict[str, Runnable]:
    """Deduplicated by service name, in catalog order.

    Two components can map onto one container — `pgvector` and `postgresql`
    are the same Postgres — and emitting the service twice produces a file
    where the second definition silently wins.
    """
    chosen: dict[str, Runnable] = {}
    for tool in source.components:
        runnable = RUNNABLE.get(tool.slug)
        if runnable is not None and runnable.service not in chosen:
            chosen[runnable.service] = runnable
    return chosen


def compose(source: StackSource) -> Artifact:
    services: dict[str, Any] = {}
    chosen = _runnables(source)

    for name, runnable in chosen.items():
        services[name] = _service(
            runnable.image,
            ports=list(runnable.ports) or None,
            environment=dict(runnable.environment) if runnable.environment else None,
            volumes=list(runnable.volumes) or None,
            gpu=runnable.gpu,
            healthcheck=list(runnable.healthcheck) or None,
            command=runnable.command,
        )

    # The application itself, last in the file and first in the mind. A
    # placeholder image rather than a real one: we do not know what the user
    # builds, and an image that pretends to be theirs is worse than one that
    # says "replace me".
    services["app"] = _service(
        "python:3.13-slim",
        ports=["8080:8080"],
        environment=_app_environment(source, chosen),
        depends_on=[name for name in chosen if _has_healthcheck(chosen[name])],
        command="sh -c 'echo replace-with-your-application-image && sleep infinity'",
    )

    volumes: dict[str, None] = {}
    for service in services.values():
        for mount in service.get("volumes", []):
            name = str(mount).split(":", 1)[0]
            if not name.startswith((".", "/")):
                volumes[name] = None

    document: dict[str, Any] = {"services": services}
    if volumes:
        document["volumes"] = volumes

    return Artifact(
        type=TYPE_COMPOSE,
        format="yaml",
        filename="docker-compose.yml",
        content=starter_header() + _notes(source, chosen) + "\n" + _dump(document),
        language="yaml",
    )


def _has_healthcheck(runnable: Runnable) -> bool:
    """`depends_on: service_healthy` on a service with no health check makes
    Compose refuse to start the whole file, which is a worse failure than not
    waiting."""
    return bool(runnable.healthcheck)


def _app_environment(source: StackSource, chosen: dict[str, Runnable]) -> dict[str, str]:
    environment: dict[str, str] = {}

    if "postgres" in chosen:
        environment["DATABASE_URL"] = "postgresql://postgres:${POSTGRES_PASSWORD}@postgres:5432/app"
    if "redis" in chosen or "valkey" in chosen:
        host = "redis" if "redis" in chosen else "valkey"
        environment["REDIS_URL"] = f"redis://{host}:6379/0"
    if "qdrant" in chosen:
        environment["QDRANT_URL"] = "http://qdrant:6333"
    if "weaviate" in chosen:
        environment["WEAVIATE_URL"] = "http://weaviate:8080"
    if "milvus" in chosen:
        environment["MILVUS_URI"] = "http://milvus:19530"
    if "chroma" in chosen:
        environment["CHROMA_URL"] = "http://chroma:8000"
    if "elasticsearch" in chosen:
        environment["ELASTICSEARCH_URL"] = "http://elasticsearch:9200"
    if "ollama" in chosen:
        environment["OLLAMA_BASE_URL"] = "http://ollama:11434"
    if "vllm" in chosen:
        environment["OPENAI_BASE_URL"] = "http://vllm:8000/v1"
    if "tgi" in chosen:
        environment["TGI_URL"] = "http://tgi:80"
    if "otel-collector" in chosen:
        environment["OTEL_EXPORTER_OTLP_ENDPOINT"] = "http://otel-collector:4317"

    # Managed providers are read from the environment, so the app service
    # forwards them rather than redefining them.
    for tool in source.components:
        for key, _ in MANAGED.get(tool.slug, ()):
            environment[key] = "${" + key + "}"

    return environment


def _notes(source: StackSource, chosen: dict[str, Runnable]) -> str:
    lines: list[str] = ["#", f"# Stack: {source.title}", "#"]

    for tool in source.components:
        if tool.slug in RUNNABLE and RUNNABLE[tool.slug].service in chosen:
            continue
        if tool.slug in MANAGED:
            keys = ", ".join(key for key, _ in MANAGED[tool.slug])
            lines.append(f"#   {tool.name}: managed service — configured via {keys}")
        elif tool.slug in EMBEDDED:
            lines.append(f"#   {tool.name}: {EMBEDDED[tool.slug]} — no container")
        else:
            lines.append(
                f"#   {tool.name}: no container image is published for this component, "
                f"so nothing is generated for it"
            )

    return "\n".join(lines) + "\n"


def env_example(source: StackSource) -> Artifact:
    """The variables the compose file and the SDKs actually read.

    Derived from the same two tables the compose file is, so a service added
    above cannot leave its password out of here.
    """
    values: list[tuple[str, str]] = []
    seen: set[str] = set()

    def add(key: str, value: str) -> None:
        if key not in seen:
            seen.add(key)
            values.append((key, value))

    chosen = _runnables(source)
    if "postgres" in chosen:
        add("POSTGRES_PASSWORD", "change-me")
    if "mongodb" in chosen:
        add("MONGO_PASSWORD", "change-me")
    if "grafana" in chosen:
        add("GRAFANA_PASSWORD", "change-me")
    if "langfuse" in chosen:
        add("LANGFUSE_NEXTAUTH_SECRET", "change-me")
        add("LANGFUSE_SALT", "change-me")
    if "vllm" in chosen:
        add("HUGGING_FACE_HUB_TOKEN", "hf_change_me")
        add("VLLM_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
    if "tgi" in chosen:
        add("HUGGING_FACE_HUB_TOKEN", "hf_change_me")
        add("TGI_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
    if "ollama" in chosen:
        add("OLLAMA_MODEL", "llama3.1:8b")

    for tool in source.components:
        for key, value in MANAGED.get(tool.slug, ()):
            add(key, value)

    body = "\n".join(f"{key}={value}" for key, value in values)
    header = (
        "# Generated by StackForge for the stack: "
        f"{source.title}\n"
        "# Every value here is a placeholder. Nothing in this file is a credential.\n"
        "# Copy to .env and replace before running anything.\n"
    )
    if not values:
        return Artifact(
            type=TYPE_ENV,
            format="text",
            filename=".env.example",
            content=header + "\n# No component in this stack reads configuration "
            "from the environment.\n",
        )

    return Artifact(
        type=TYPE_ENV,
        format="text",
        filename=".env.example",
        content=header + "\n" + body + "\n",
    )
