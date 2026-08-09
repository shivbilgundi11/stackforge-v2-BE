"""WF4 endpoints, and the generated files actually validated.

The two generator tools are checked against the real validators rather than
against "does it parse as YAML". YAML that parses but fails `docker compose
config` or `kubectl apply` is worse than no file, because the user finds out
at deploy time in their own terminal.

The Compose check shells out to Docker and skips when the daemon is not
running. The Kubernetes check uses the official client's generated models,
which come from the same OpenAPI spec `kubectl` validates against and, unlike
`kubectl --validate`, do not need a live cluster.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml
from httpx import AsyncClient
from kubernetes.client.api_client import ApiClient

pytestmark = pytest.mark.usefixtures("seeded_catalog")

VRAM = "/api/v1/tools/infra/vram-estimate"
GPU_COST = "/api/v1/tools/infra/gpu-cost"
CLOUD = "/api/v1/tools/infra/cloud-cost"
COMPOSE = "/api/v1/tools/infra/docker-compose"
K8S = "/api/v1/tools/infra/k8s-estimate"
READINESS = "/api/v1/tools/infra/readiness-checklist"


def _docker() -> str | None:
    """Path to a working Docker CLI with a reachable daemon, or None."""
    binary = shutil.which("docker") or shutil.which(
        "docker.exe",
        path=r"C:\Users\shivb\AppData\Local\Programs\DockerDesktop\resources\bin",
    )
    if not binary:
        return None
    probe = subprocess.run(  # noqa: S603
        [binary, "version", "--format", "{{.Server.Version}}"],
        capture_output=True,
        timeout=30,
    )
    return binary if probe.returncode == 0 else None


class _Resp:
    """Minimal shape `ApiClient.deserialize` reads."""

    def __init__(self, data: str) -> None:
        self.data = data


def _validate_k8s(document: dict[str, object], kind: str) -> object:
    """Deserialize through the official generated model.

    Raises on an unknown field type or a malformed structure, which is the
    same check `kubectl` performs against the cluster's OpenAPI document.
    """
    return ApiClient().deserialize(_Resp(json.dumps(document)), kind)


# ── vram-estimate ────────────────────────────────────────────────────────────


async def test_vram_estimate_end_to_end(client: AsyncClient) -> None:
    response = await client.post(
        VRAM,
        json={"architecture_key": "llama-3.1-8b", "quantisation": "fp16", "context": 8192},
    )
    assert response.status_code == 200

    data = response.json()["data"]
    assert data["metrics"]["weights_gb"] == "14.96"
    assert data["metrics"]["kv_cache_gb"] == "1.00"
    assert data["metrics"]["attention"] == "GQA"
    assert data["metrics"]["kv_bytes_per_token"] == 131072


async def test_the_fit_list_covers_the_seeded_gpu_catalog(client: AsyncClient) -> None:
    response = await client.post(
        VRAM, json={"architecture_key": "llama-3.3-70b", "quantisation": "fp16", "context": 8192}
    )
    rows = response.json()["data"]["tables"]["gpu_fit"]

    assert len(rows) >= 30
    verdicts = {row["verdict"] for row in rows}
    # A 70B at fp16 is ~131 GiB: the big nodes take it, the single cards do not.
    assert "fits" in verdicts
    assert "does not fit" in verdicts


async def test_an_unknown_architecture_is_404_not_500(client: AsyncClient) -> None:
    response = await client.post(VRAM, json={"architecture_key": "not-a-model"})
    assert response.status_code == 404


async def test_architectures_are_listed_without_a_provenance_chip(client: AsyncClient) -> None:
    """They are physical constants, not prices. A freshness badge on an
    immutable fact teaches people to ignore the badges that matter."""
    response = await client.get("/api/v1/catalog/architectures")
    assert response.status_code == 200

    rows = response.json()["data"]
    assert len(rows) >= 15
    assert all("provenance" not in row for row in rows)

    llama = next(row for row in rows if row["key"] == "llama-3.1-8b")
    assert llama["kv_heads"] == 8
    assert llama["heads"] == 32
    assert llama["uses_gqa"] is True


# ── gpu-cost ─────────────────────────────────────────────────────────────────


async def test_gpu_cost_returns_a_break_even_volume(client: AsyncClient) -> None:
    gpus = (await client.get("/api/v1/catalog/gpus", params={"provider": "lambda"})).json()["data"]
    gpu_id = next(row["id"] for row in gpus if row["instance_name"] == "gpu_1x_h100_pcie")

    response = await client.post(
        GPU_COST,
        json={
            "gpu_id": gpu_id,
            "hours_per_day": "24",
            "days_per_month": 30,
            "utilisation_pct": "100",
            "api_model_id": "gpt-4o-mini",
            "requests_per_day": 5000,
        },
    )
    assert response.status_code == 200

    metrics = response.json()["data"]["metrics"]
    # $3.29/h x 720 h = $2,368.80.
    assert metrics["self_host_monthly"] == "2368.800000"
    assert isinstance(metrics["break_even_requests_per_day"], int)
    assert len(response.json()["data"]["series"]["crossover"]) == 13


async def test_an_unknown_gpu_is_404(client: AsyncClient) -> None:
    response = await client.post(GPU_COST, json={"gpu_id": "gpu_nope"})
    assert response.status_code == 404


# ── cloud-cost ───────────────────────────────────────────────────────────────


async def test_cloud_cost_includes_egress_and_names_the_driver(client: AsyncClient) -> None:
    response = await client.post(
        CLOUD,
        json={
            "provider": "aws",
            "compute_monthly": "800",
            "database_monthly": "200",
            "cache_monthly": "50",
            "storage_gb": "1000",
            "egress_gb": "2000",
            "load_balancer_monthly": "25",
        },
    )
    metrics = response.json()["data"]["metrics"]

    assert metrics["monthly_total"] == "1278.000000"
    assert metrics["egress_cost"] == "180.000000"
    assert metrics["dominant_driver"] == "Compute"

    lines = {row["line"] for row in response.json()["data"]["tables"]["lines"]}
    assert "Egress" in lines


# ── docker-compose ───────────────────────────────────────────────────────────


async def test_compose_output_is_valid_yaml_with_declared_volumes(client: AsyncClient) -> None:
    response = await client.post(
        COMPOSE, json={"archetype": "rag-stack", "model": "meta-llama/Llama-3.1-8B", "gpu": True}
    )
    assert response.status_code == 200

    data = response.json()["data"]
    compose = next(a for a in data["artifacts"] if a["filename"] == "docker-compose.yml")
    document = yaml.safe_load(compose["content"])

    assert set(document["services"]) == {"vllm", "redis", "qdrant"}

    # Every mounted named volume is declared, or Compose refuses to start.
    mounted = {
        str(mount).split(":", 1)[0]
        for service in document["services"].values()
        for mount in service.get("volumes", [])
    }
    assert mounted <= set(document.get("volumes", {}))


async def test_a_model_tag_with_a_colon_does_not_corrupt_the_file(client: AsyncClient) -> None:
    """Every Ollama tag has a colon in it.

    Interpolated into a hand-built YAML line, `llama3.1:8b` becomes a nested
    key and the file parses successfully as the wrong thing.
    """
    response = await client.post(
        COMPOSE, json={"archetype": "ollama-webui", "model": "llama3.1:8b", "gpu": False}
    )
    data = response.json()["data"]
    env = next(a for a in data["artifacts"] if a["filename"] == ".env.example")

    assert "OLLAMA_MODEL=llama3.1:8b" in env["content"]

    compose = next(a for a in data["artifacts"] if a["filename"] == "docker-compose.yml")
    document = yaml.safe_load(compose["content"])
    assert isinstance(document["services"]["ollama"]["image"], str)


async def test_every_generated_file_says_it_is_a_starter(client: AsyncClient) -> None:
    response = await client.post(COMPOSE, json={"archetype": "vllm-redis", "model": "x"})
    data = response.json()["data"]

    compose = next(a for a in data["artifacts"] if a["filename"] == "docker-compose.yml")
    assert "STARTER TEMPLATE" in compose["content"]
    assert any("starter template" in w["message"].lower() for w in data["warnings"])


@pytest.mark.parametrize(
    "archetype", ["ollama-webui", "vllm-redis", "fastapi-pgvector", "rag-stack"]
)
async def test_compose_passes_docker_compose_config(
    client: AsyncClient, tmp_path: Path, archetype: str
) -> None:
    """The real validator, on every archetype.

    Skipped rather than faked when Docker is unavailable — a test that
    silently stops checking is worse than one that says it did not run.
    """
    binary = await asyncio.to_thread(_docker)
    if binary is None:
        pytest.skip("Docker daemon not available")

    response = await client.post(
        COMPOSE, json={"archetype": archetype, "model": "llama3.1:8b", "gpu": True}
    )
    compose = next(
        a for a in response.json()["data"]["artifacts"] if a["filename"] == "docker-compose.yml"
    )

    path = tmp_path / "docker-compose.yml"
    await asyncio.to_thread(path.write_text, compose["content"], "utf-8")

    result = await asyncio.to_thread(
        subprocess.run,
        [binary, "compose", "-f", str(path), "config", "--quiet"],
        capture_output=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr.decode(errors="replace")


# ── k8s-estimate ─────────────────────────────────────────────────────────────


async def test_k8s_manifests_validate_against_the_kubernetes_models(
    client: AsyncClient,
) -> None:
    response = await client.post(
        K8S,
        json={
            "name": "inference",
            "image": "vllm/vllm-openai:latest",
            "replicas": 2,
            "gpu_count": 1,
            "vram_required_gb": "24",
        },
    )
    assert response.status_code == 200

    artifacts = {a["filename"]: a["content"] for a in response.json()["data"]["artifacts"]}
    assert set(artifacts) == {"deployment.yaml", "service.yaml", "hpa.yaml", "pdb.yaml"}

    kinds = {
        "deployment.yaml": "V1Deployment",
        "service.yaml": "V1Service",
        "hpa.yaml": "V2HorizontalPodAutoscaler",
        "pdb.yaml": "V1PodDisruptionBudget",
    }
    for filename, kind in kinds.items():
        document = yaml.safe_load(artifacts[filename])
        _validate_k8s(document, kind)  # raises on a bad shape or type


async def test_the_gpu_resource_reaches_both_requests_and_limits(client: AsyncClient) -> None:
    """Kubernetes rejects a GPU request without a matching limit."""
    response = await client.post(K8S, json={"gpu_count": 2, "vram_required_gb": "80"})
    artifacts = {a["filename"]: a["content"] for a in response.json()["data"]["artifacts"]}

    deployment = yaml.safe_load(artifacts["deployment.yaml"])
    resources = deployment["spec"]["template"]["spec"]["containers"][0]["resources"]

    assert resources["requests"]["nvidia.com/gpu"] == 2
    assert resources["limits"]["nvidia.com/gpu"] == 2


async def test_a_gpu_deployment_warns_that_cpu_autoscaling_will_not_work(
    client: AsyncClient,
) -> None:
    response = await client.post(K8S, json={"gpu_count": 1, "vram_required_gb": "24"})
    warnings = response.json()["data"]["warnings"]
    assert any("CPU" in w["message"] and "saturates" in w["message"] for w in warnings)


async def test_the_readiness_probe_tolerates_a_slow_model_load(client: AsyncClient) -> None:
    """Default probe timings restart a model server before it finishes loading
    weights, which reads as a crash loop and is an impatient probe."""
    response = await client.post(K8S, json={"gpu_count": 1, "vram_required_gb": "24"})
    artifacts = {a["filename"]: a["content"] for a in response.json()["data"]["artifacts"]}

    container = yaml.safe_load(artifacts["deployment.yaml"])["spec"]["template"]["spec"][
        "containers"
    ][0]
    assert container["readinessProbe"]["failureThreshold"] >= 20
    assert container["livenessProbe"]["initialDelaySeconds"] >= 120


# ── readiness-checklist ──────────────────────────────────────────────────────


async def test_readiness_checklist_scores_and_conditions_its_items(client: AsyncClient) -> None:
    response = await client.post(
        READINESS, json={"self_hosted": True, "has_rag": True, "completed": []}
    )
    assert response.status_code == 200

    data = response.json()["data"]
    assert data["metrics"]["score"] == 0
    assert data["metrics"]["profile"] == "self-hosted"
    assert any("GPU node" in row["item"] for row in data["tables"]["checklist"])


async def test_completing_items_raises_the_score(client: AsyncClient) -> None:
    blank = await client.post(READINESS, json={"self_hosted": False, "has_rag": False})
    items = [row["item"] for row in blank.json()["data"]["tables"]["checklist"]][:3]

    partial = await client.post(
        READINESS, json={"self_hosted": False, "has_rag": False, "completed": items}
    )
    assert partial.json()["data"]["metrics"]["score"] > 0
    assert partial.json()["data"]["metrics"]["completed"] == 3
