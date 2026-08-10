"""Agent & MCP Builder endpoints (WF3)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from app.api.deps import Db, RunIdentity
from app.core.errors import NotFound, ValidationFailed
from app.core.responses import Envelope, ok
from app.data import rate_limits as rate_limit_data
from app.schemas.agents import (
    AgentCostIn,
    FunctionSchemaIn,
    McpConfigIn,
    RateLimitsIn,
    WorkflowPlanIn,
)
from app.schemas.tools import ToolOutput, ToolRunOut, ToolWarning
from app.services import (
    agent_planner_service,
    agent_service,
    ai_service,
    catalog_service,
    mcp_generator,
    tool_service,
)

router = APIRouter(tags=["agents"])

WORKFLOW = "agents"


@router.post("/mcp-config", response_model=Envelope[ToolRunOut], name="run_mcp_config")
async def run_mcp_config(db: Db, identity: RunIdentity, payload: McpConfigIn) -> dict[str, Any]:
    def compute() -> ToolOutput:
        bundle = mcp_generator.generate(
            server_name=payload.server_name,
            description=payload.description,
            tools=[tool.model_dump() for tool in payload.tools],
            transport=payload.transport,
            auth=payload.auth,
            resources=[resource.model_dump() for resource in payload.resources],
            prompts=[prompt.model_dump() for prompt in payload.prompts],
        )
        server = next(artifact for artifact in bundle.artifacts if artifact.type == "mcp-server")
        return ToolOutput(
            metrics={
                "server": payload.server_name,
                "tools": len(bundle.tool_names),
                "files": len(bundle.artifacts),
                "transport": payload.transport,
                "auth": payload.auth,
                "server_lines": server.content.count("\n") + 1,
                "spec_version": mcp_generator.MCP_SPEC_VERSION,
            },
            tables={
                "tools": [
                    {
                        "declared": declared.name,
                        "generated": generated,
                        "parameters": len(declared.parameters),
                        "required": sum(1 for param in declared.parameters if param.required),
                    }
                    for declared, generated in zip(payload.tools, bundle.tool_names, strict=True)
                ],
                "files": [
                    {"file": artifact.filename, "bytes": len(artifact.content)}
                    for artifact in bundle.artifacts
                ],
            },
            artifacts=bundle.artifacts,
            warnings=bundle.warnings,
        )

    result = await tool_service.run_tool(
        db,
        slug="mcp-config",
        workflow=WORKFLOW,
        payload=payload,
        identity=identity,
        compute=compute,
    )
    return ok(result)


@router.post("/agent-cost", response_model=Envelope[ToolRunOut], name="run_agent_cost")
async def run_agent_cost(db: Db, identity: RunIdentity, payload: AgentCostIn) -> dict[str, Any]:
    models = await catalog_service.get_models_by_ids(
        db, [agent.model_id for agent in payload.agents]
    )
    missing = [agent.model_id for agent in payload.agents if agent.model_id not in models]
    if missing:
        raise NotFound(f"No pricing for model(s): {', '.join(sorted(set(missing)))}.")

    roles = [
        agent_service.AgentRole(
            role=agent.role,
            model=models[agent.model_id],
            count=agent.count,
            steps_per_task=agent.steps_per_task,
        )
        for agent in payload.agents
    ]

    result = await tool_service.run_tool(
        db,
        slug="agent-cost",
        workflow=WORKFLOW,
        payload=payload,
        identity=identity,
        compute=lambda: agent_service.agent_cost(
            agents=roles,
            tasks_per_day=payload.tasks_per_day,
            input_tokens_per_step=payload.input_tokens_per_step,
            output_tokens_per_step=payload.output_tokens_per_step,
            tool_count=payload.tool_count,
            tokens_per_tool_schema=payload.tokens_per_tool_schema,
            tool_calls_per_step=payload.tool_calls_per_step,
            tokens_per_tool_result=payload.tokens_per_tool_result,
            memory_read_tokens=payload.memory_read_tokens,
            memory_write_tokens=payload.memory_write_tokens,
            retry_rate_pct=payload.retry_rate_pct,
            cached_input_ratio=payload.cached_input_ratio,
        ),
    )
    return ok(result)


@router.post("/workflow-plan", response_model=Envelope[ToolRunOut], name="run_workflow_plan")
async def run_workflow_plan(
    db: Db, identity: RunIdentity, payload: WorkflowPlanIn
) -> dict[str, Any]:
    frontier = await catalog_service.get_model(db, payload.frontier_model_id)
    fast = (
        await catalog_service.get_model(db, payload.fast_model_id)
        if payload.fast_model_id
        else None
    )
    catalog = await catalog_service.list_tools(db)

    result = await tool_service.run_tool(
        db,
        slug="workflow-plan",
        workflow=WORKFLOW,
        payload=payload,
        identity=identity,
        compute=lambda: agent_planner_service.plan(
            goal=payload.goal,
            coordination=payload.coordination,
            available_tools=payload.available_tools,
            constraints=payload.constraints,
            tasks_per_day=payload.tasks_per_day,
            frontier_model=frontier,
            fast_model=fast,
            catalog=catalog,
        ),
        # The topology, contracts, costs, and failure modes are complete before
        # this runs. Synthesis rewrites the summary and names the weakest edge;
        # without a key it returns None and the run stays `rule_based`.
        enrich=ai_service.enrichment(
            db,
            purpose="agent_plan",
            identity=identity,
            tool_slug="workflow-plan",
            variables=payload.model_dump(mode="json"),
            apply=_apply_plan_commentary,
        ),
    )
    return ok(result)


def _apply_plan_commentary(output: ToolOutput, data: dict[str, Any]) -> None:
    """Commentary only. The model never adds or removes an agent."""
    if summary := str(data.get("summary") or "").strip():
        output.metrics["summary"] = summary
    if why := str(data.get("why") or "").strip():
        output.metrics["rationale"] = why
    if weakest := str(data.get("weakest_link") or "").strip():
        output.metrics["weakest_link"] = weakest
    for item in data.get("watch_out_for") or []:
        output.warnings.append(ToolWarning(level="info", message=str(item)))


@router.post("/function-schema", response_model=Envelope[ToolRunOut], name="run_function_schema")
async def run_function_schema(
    db: Db, identity: RunIdentity, payload: FunctionSchemaIn
) -> dict[str, Any]:
    result = await tool_service.run_tool(
        db,
        slug="function-schema",
        workflow=WORKFLOW,
        payload=payload,
        identity=identity,
        compute=lambda: agent_service.function_schema(
            tools=[tool.model_dump() for tool in payload.tools],
            target=payload.target,
        ),
    )
    return ok(result)


@router.post("/rate-limits", response_model=Envelope[ToolRunOut], name="run_rate_limits")
async def run_rate_limits(db: Db, identity: RunIdentity, payload: RateLimitsIn) -> dict[str, Any]:
    provider = rate_limit_data.BY_KEY.get(payload.provider)
    if provider is None:
        raise NotFound(f"No published limits for provider {payload.provider!r}.")

    tier = provider.tier(payload.tier)
    if tier is None:
        available = ", ".join(item.key for item in provider.tiers)
        raise ValidationFailed.on_field(
            "tier",
            f"{provider.label} publishes: {available}.",
            summary=f"{provider.label} has no tier {payload.tier!r}.",
        )

    result = await tool_service.run_tool(
        db,
        slug="rate-limits",
        workflow=WORKFLOW,
        payload=payload,
        identity=identity,
        compute=lambda: agent_service.rate_limits(
            provider=provider,
            tier=tier,
            requests_per_min=payload.requests_per_min,
            input_tokens_per_request=payload.input_tokens_per_request,
            output_tokens_per_request=payload.output_tokens_per_request,
            concurrency=payload.concurrency,
            avg_request_seconds=payload.avg_request_seconds,
            burst_multiplier=payload.burst_multiplier,
            burst_duration_seconds=payload.burst_duration_seconds,
        ),
    )
    return ok(result)
