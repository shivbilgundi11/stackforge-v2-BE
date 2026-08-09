"""Infra Planner request shapes (WF4)."""

from __future__ import annotations

from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, Field

Quantisation = Literal[
    "fp32",
    "fp16",
    "bf16",
    "int8",
    "fp8",
    "int4",
    "gguf-q8_0",
    "gguf-q6_k",
    "gguf-q5_k_m",
    "gguf-q4_k_m",
    "gguf-q3_k_m",
    "awq-4bit",
    "gptq-4bit",
]
KvPrecision = Literal["fp32", "fp16", "bf16", "fp8", "int8"]
Runtime = Literal["vllm", "tgi", "llama.cpp", "transformers", "sglang"]


class VramEstimateIn(BaseModel):
    architecture_key: str = Field(description="Key from /catalog/architectures.")
    quantisation: Quantisation = "fp16"
    context: int = Field(default=8192, ge=128, le=2_000_000)
    concurrency: int = Field(default=1, ge=1, le=512)
    kv_precision: KvPrecision = "fp16"
    runtime: Runtime = "vllm"
    lora_finetune: bool = False


class GpuCostIn(BaseModel):
    gpu_id: str
    hours_per_day: Decimal = Field(default=Decimal(24), gt=0, le=24)
    days_per_month: int = Field(default=30, ge=1, le=31)
    utilisation_pct: Decimal = Field(default=Decimal(60), ge=1, le=100)
    api_model_id: str | None = None
    input_tokens: int = Field(default=2000, ge=0, le=10_000_000)
    output_tokens: int = Field(default=500, ge=0, le=1_000_000)
    requests_per_day: int = Field(default=1000, ge=0, le=100_000_000)


class CloudCostIn(BaseModel):
    provider: Literal["aws", "gcp", "azure"] = "aws"
    compute_monthly: Decimal = Field(default=Decimal(0), ge=0)
    database_monthly: Decimal = Field(default=Decimal(0), ge=0)
    cache_monthly: Decimal = Field(default=Decimal(0), ge=0)
    storage_gb: Decimal = Field(default=Decimal(0), ge=0, le=10_000_000)
    egress_gb: Decimal = Field(default=Decimal(0), ge=0, le=10_000_000)
    load_balancer_monthly: Decimal = Field(default=Decimal(0), ge=0)


class DockerComposeIn(BaseModel):
    archetype: Literal["ollama-webui", "vllm-redis", "fastapi-pgvector", "rag-stack"]
    model: str = Field(
        default="llama3.1:8b",
        min_length=1,
        max_length=200,
        description=(
            "Model tag or HuggingFace id. Emitted through a YAML dumper, "
            "never interpolated into a template."
        ),
    )
    gpu: bool = True


class K8sEstimateIn(BaseModel):
    name: str = Field(default="inference", min_length=1, max_length=50, pattern=r"^[a-z0-9-]+$")
    image: str = Field(default="vllm/vllm-openai:latest", min_length=1, max_length=200)
    replicas: int = Field(default=2, ge=1, le=100)
    requests_per_second: Decimal = Field(default=Decimal(10), gt=0, le=100_000)
    gpu_count: int = Field(default=1, ge=0, le=8)
    vram_required_gb: Decimal = Field(default=Decimal(20), ge=0, le=2000)
    target_utilisation: int = Field(default=70, ge=10, le=95)
    max_replicas: int = Field(default=8, ge=1, le=200)


class ReadinessChecklistIn(BaseModel):
    self_hosted: bool = False
    has_rag: bool = True
    completed: list[str] = Field(default_factory=list, max_length=60)
