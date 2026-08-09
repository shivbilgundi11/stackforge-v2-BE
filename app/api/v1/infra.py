"""Infra Planner endpoints (WF4)."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import yaml
from fastapi import APIRouter

from app.api.deps import Db, RunIdentity
from app.core.errors import NotFound
from app.core.responses import Envelope, ok
from app.data.architectures_seed import BY_KEY
from app.schemas.infra import (
    CloudCostIn,
    DockerComposeIn,
    GpuCostIn,
    K8sEstimateIn,
    ReadinessChecklistIn,
    VramEstimateIn,
)
from app.schemas.tools import ToolOutput, ToolRunOut
from app.services import catalog_service, infra_artifacts, infra_service, tool_service

router = APIRouter(tags=["infra"])

WORKFLOW = "infra"


@router.post("/vram-estimate", response_model=Envelope[ToolRunOut], name="run_vram_estimate")
async def run_vram_estimate(
    db: Db, identity: RunIdentity, payload: VramEstimateIn
) -> dict[str, Any]:
    arch = BY_KEY.get(payload.architecture_key)
    if arch is None:
        raise NotFound(f"No architecture named {payload.architecture_key!r}.")

    gpus = await catalog_service.list_gpus(db)

    result = await tool_service.run_tool(
        db,
        slug="vram-estimate",
        workflow=WORKFLOW,
        payload=payload,
        identity=identity,
        compute=lambda: infra_service.vram_estimate(
            arch,
            quantisation=payload.quantisation,
            context=payload.context,
            concurrency=payload.concurrency,
            kv_precision=payload.kv_precision,
            runtime=payload.runtime,
            lora_finetune=payload.lora_finetune,
            gpus=gpus,
        ),
    )
    return ok(result)


@router.post("/gpu-cost", response_model=Envelope[ToolRunOut], name="run_gpu_cost")
async def run_gpu_cost(db: Db, identity: RunIdentity, payload: GpuCostIn) -> dict[str, Any]:
    gpus = await catalog_service.list_gpus(db)
    gpu = next((row for row in gpus if row.id == payload.gpu_id), None)
    if gpu is None:
        raise NotFound(f"No GPU instance with id {payload.gpu_id!r}.")

    api_model = (
        await catalog_service.get_model(db, payload.api_model_id) if payload.api_model_id else None
    )

    result = await tool_service.run_tool(
        db,
        slug="gpu-cost",
        workflow=WORKFLOW,
        payload=payload,
        identity=identity,
        compute=lambda: infra_service.gpu_cost(
            gpu,
            hours_per_day=payload.hours_per_day,
            days_per_month=payload.days_per_month,
            utilisation_pct=payload.utilisation_pct,
            api_model=api_model,
            input_tokens=payload.input_tokens,
            output_tokens=payload.output_tokens,
            requests_per_day=payload.requests_per_day,
        ),
    )
    return ok(result)


@router.post("/cloud-cost", response_model=Envelope[ToolRunOut], name="run_cloud_cost")
async def run_cloud_cost(db: Db, identity: RunIdentity, payload: CloudCostIn) -> dict[str, Any]:
    result = await tool_service.run_tool(
        db,
        slug="cloud-cost",
        workflow=WORKFLOW,
        payload=payload,
        identity=identity,
        compute=lambda: infra_service.cloud_cost(
            provider=payload.provider,
            compute_monthly=payload.compute_monthly,
            database_monthly=payload.database_monthly,
            cache_monthly=payload.cache_monthly,
            storage_gb=payload.storage_gb,
            egress_gb=payload.egress_gb,
            load_balancer_monthly=payload.load_balancer_monthly,
        ),
    )
    return ok(result)


@router.post("/docker-compose", response_model=Envelope[ToolRunOut], name="run_docker_compose")
async def run_docker_compose(
    db: Db, identity: RunIdentity, payload: DockerComposeIn
) -> dict[str, Any]:
    def compute() -> ToolOutput:
        artifacts, warnings = infra_artifacts.docker_compose(
            archetype=payload.archetype, model=payload.model, gpu=payload.gpu
        )
        return ToolOutput(
            metrics={
                "archetype": payload.archetype,
                "services": _service_count(artifacts[0].content),
                "gpu_passthrough": "yes" if payload.gpu else "no",
                "files": len(artifacts),
            },
            artifacts=artifacts,
            warnings=warnings,
        )

    result = await tool_service.run_tool(
        db,
        slug="docker-compose",
        workflow=WORKFLOW,
        payload=payload,
        identity=identity,
        compute=compute,
    )
    return ok(result)


def _service_count(compose: str) -> int:
    """Services in the emitted document, counted by parsing it.

    Parsed rather than counted from the builder's dict, so the number reported
    is the number a reader would find in the file. If the dump ever produced
    something that did not round-trip, this is where it would show.
    """
    document = yaml.safe_load(compose)
    return len(document.get("services", {})) if isinstance(document, dict) else 0


@router.post("/k8s-estimate", response_model=Envelope[ToolRunOut], name="run_k8s_estimate")
async def run_k8s_estimate(db: Db, identity: RunIdentity, payload: K8sEstimateIn) -> dict[str, Any]:
    def compute() -> ToolOutput:
        # Requests and limits derived from the VRAM figure the user already
        # has, rather than asked for a second time in different units.
        memory_request = max(4, int(payload.vram_required_gb / Decimal(2)) + 4)
        memory_limit = int(memory_request * 1.5)
        cpu_request = "2" if payload.gpu_count else "1"
        cpu_limit = "8" if payload.gpu_count else "4"

        artifacts, warnings = infra_artifacts.k8s_manifests(
            name=payload.name,
            image=payload.image,
            replicas=payload.replicas,
            cpu_request=cpu_request,
            cpu_limit=cpu_limit,
            memory_request_gi=memory_request,
            memory_limit_gi=memory_limit,
            gpu_count=payload.gpu_count,
            target_utilisation=payload.target_utilisation,
            max_replicas=payload.max_replicas,
        )

        rps_per_replica = payload.requests_per_second / Decimal(payload.replicas)
        return ToolOutput(
            metrics={
                "replicas": payload.replicas,
                "gpus_total": payload.gpu_count * payload.replicas,
                "memory_request_gi": memory_request,
                "memory_limit_gi": memory_limit,
                "cpu_request": cpu_request,
                "rps_per_replica": rps_per_replica.quantize(Decimal("0.01")),
                "manifests": len(artifacts),
            },
            tables={
                "resources": [
                    {"resource": "CPU", "request": cpu_request, "limit": cpu_limit},
                    {
                        "resource": "Memory",
                        "request": f"{memory_request}Gi",
                        "limit": f"{memory_limit}Gi",
                    },
                    {
                        "resource": "GPU",
                        "request": str(payload.gpu_count),
                        "limit": str(payload.gpu_count),
                    },
                ]
            },
            artifacts=artifacts,
            warnings=warnings,
        )

    result = await tool_service.run_tool(
        db,
        slug="k8s-estimate",
        workflow=WORKFLOW,
        payload=payload,
        identity=identity,
        compute=compute,
    )
    return ok(result)


@router.post(
    "/readiness-checklist", response_model=Envelope[ToolRunOut], name="run_readiness_checklist"
)
async def run_readiness_checklist(
    db: Db, identity: RunIdentity, payload: ReadinessChecklistIn
) -> dict[str, Any]:
    result = await tool_service.run_tool(
        db,
        slug="readiness-checklist",
        workflow=WORKFLOW,
        payload=payload,
        identity=identity,
        compute=lambda: infra_service.readiness_checklist(
            self_hosted=payload.self_hosted,
            has_rag=payload.has_rag,
            completed=payload.completed,
        ),
    )
    return ok(result)
