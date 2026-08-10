"""Agent & MCP Builder request shapes (WF3).

Tool names are deliberately *not* pattern-constrained here. The generator
derives a safe identifier from whatever arrives and reports what it changed,
which is the behaviour a user typing "Search Docs" wants. A 422 telling them
their tool name is not a Python identifier is technically correct and useless.
Safety comes from the generator never interpolating the input, not from the
schema refusing it.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, Field

JsonType = Literal["string", "number", "integer", "boolean", "array", "object"]
SchemaTarget = Literal["openai", "anthropic", "json-schema", "mcp"]
Coordination = Literal["sequential", "parallel", "hierarchical", "handoff"]


class ToolParameterIn(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    type: JsonType = "string"
    description: str = Field(default="", max_length=500)
    required: bool = True
    enum: list[str] = Field(default_factory=list, max_length=40)
    item_type: JsonType | None = Field(
        default=None, description="Element type when `type` is array."
    )


class ToolDefinitionIn(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    description: str = Field(default="", max_length=1000)
    parameters: list[ToolParameterIn] = Field(default_factory=list, max_length=30)


class ResourceIn(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    uri: str = Field(default="", max_length=200)
    description: str = Field(default="", max_length=500)


class PromptIn(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    description: str = Field(default="", max_length=500)
    template: str = Field(default="", max_length=2000)


class McpConfigIn(BaseModel):
    server_name: str = Field(min_length=1, max_length=80)
    description: str = Field(default="", max_length=1000)
    transport: Literal["stdio", "sse", "streamable-http"] = "stdio"
    auth: Literal["none", "api-key", "bearer"] = "none"
    tools: list[ToolDefinitionIn] = Field(min_length=1, max_length=30)
    resources: list[ResourceIn] = Field(default_factory=list, max_length=10)
    prompts: list[PromptIn] = Field(default_factory=list, max_length=10)


class AgentRoleIn(BaseModel):
    role: str = Field(min_length=1, max_length=60)
    model_id: str = Field(min_length=1, max_length=64)
    count: int = Field(default=1, ge=1, le=50)
    steps_per_task: int = Field(default=4, ge=1, le=200)


class AgentCostIn(BaseModel):
    agents: list[AgentRoleIn] = Field(min_length=1, max_length=8)
    tasks_per_day: int = Field(default=100, ge=1, le=10_000_000)
    input_tokens_per_step: int = Field(default=1200, ge=0, le=2_000_000)
    output_tokens_per_step: int = Field(default=400, ge=0, le=200_000)
    # The two inputs the naive calculator omits, and the reason its answer is
    # wrong by a multiple rather than a percent.
    tool_count: int = Field(default=8, ge=0, le=200)
    tokens_per_tool_schema: int = Field(default=120, ge=0, le=10_000)
    tool_calls_per_step: int = Field(default=1, ge=0, le=50)
    tokens_per_tool_result: int = Field(default=400, ge=0, le=200_000)
    memory_read_tokens: int = Field(default=0, ge=0, le=2_000_000)
    memory_write_tokens: int = Field(default=0, ge=0, le=200_000)
    retry_rate_pct: Decimal = Field(default=Decimal(15), ge=0, le=500)
    cached_input_ratio: Decimal = Field(default=Decimal(0), ge=0, le=1)


class WorkflowPlanIn(BaseModel):
    goal: str = Field(min_length=10, max_length=2000)
    coordination: Coordination = "sequential"
    available_tools: list[str] = Field(default_factory=list, max_length=20)
    constraints: list[str] = Field(default_factory=list, max_length=10)
    tasks_per_day: int = Field(default=100, ge=1, le=10_000_000)
    frontier_model_id: str = Field(min_length=1, max_length=64)
    fast_model_id: str | None = Field(
        default=None,
        max_length=64,
        description="Routes the worker roles. Omit to price every node on the frontier model.",
    )


class FunctionSchemaIn(BaseModel):
    tools: list[ToolDefinitionIn] = Field(min_length=1, max_length=30)
    target: SchemaTarget = "anthropic"


class RateLimitsIn(BaseModel):
    provider: Literal["anthropic", "openai", "google"] = "anthropic"
    tier: str = Field(default="tier-1", min_length=1, max_length=20)
    requests_per_min: int = Field(default=60, ge=1, le=10_000_000)
    input_tokens_per_request: int = Field(default=4000, ge=0, le=2_000_000)
    output_tokens_per_request: int = Field(default=800, ge=0, le=200_000)
    concurrency: int = Field(default=8, ge=1, le=100_000)
    avg_request_seconds: Decimal = Field(
        default=Decimal(4),
        gt=0,
        le=600,
        description="Per-request wall time. With concurrency this is your own ceiling.",
    )
    burst_multiplier: Decimal = Field(default=Decimal(1), ge=1, le=100)
    burst_duration_seconds: int = Field(default=60, ge=1, le=3600)
