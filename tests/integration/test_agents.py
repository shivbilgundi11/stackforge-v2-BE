"""WF3 endpoints, and the generated MCP server actually started.

The handshake test is the one that matters. `mcp-config` claims to emit a
runnable server, and the only way to know whether it does is to write the
bundle to disk, start it in a subprocess, complete an MCP initialize over
stdio, and ask it for its tools. Everything short of that — it parses, it
imports, it looks right — is what the previous build had.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import pytest
from httpx import AsyncClient

pytestmark = pytest.mark.usefixtures("seeded_catalog")

MCP = "/api/v1/tools/agents/mcp-config"
COST = "/api/v1/tools/agents/agent-cost"
PLAN = "/api/v1/tools/agents/workflow-plan"
SCHEMA = "/api/v1/tools/agents/function-schema"
LIMITS = "/api/v1/tools/agents/rate-limits"

TOOLS: list[dict[str, Any]] = [
    {
        "name": "search_docs",
        "description": "Search the internal handbook and return matching passages.",
        "parameters": [
            {
                "name": "query",
                "type": "string",
                "description": "What to look for.",
                "required": True,
            },
            {
                "name": "limit",
                "type": "integer",
                "description": "Max results.",
                "required": False,
            },
        ],
    },
    {
        "name": "page_oncall",
        "description": "Page the engineer currently on call with a short message.",
        "parameters": [
            {"name": "message", "type": "string", "description": "What happened.", "required": True}
        ],
    },
]


# ── mcp-config ───────────────────────────────────────────────────────────────


async def test_mcp_config_returns_a_six_file_bundle(client: AsyncClient) -> None:
    response = await client.post(
        MCP,
        json={
            "server_name": "Ops Toolkit",
            "description": "Tools for the on-call rotation.",
            "tools": TOOLS,
        },
    )
    assert response.status_code == 200

    data = response.json()["data"]
    assert data["metrics"]["tools"] == 2
    assert data["metrics"]["files"] == 6
    assert data["source"] == "rule_based"

    filenames = {artifact["filename"] for artifact in data["artifacts"]}
    assert filenames == {
        "mcp-server-ops-toolkit/server.py",
        "mcp-server-ops-toolkit/pyproject.toml",
        "mcp-server-ops-toolkit/README.md",
        "mcp-server-ops-toolkit/.env.example",
        "mcp-server-ops-toolkit/tests/test_server.py",
        "mcp-server-ops-toolkit/claude_desktop_config.json",
    }


async def test_a_tool_name_with_a_quote_or_newline_does_not_break_the_file(
    client: AsyncClient,
) -> None:
    response = await client.post(
        MCP,
        json={
            "server_name": "Ops",
            "tools": [
                {
                    "name": '"; import os\nos.system("echo pwned")  #',
                    "description": 'Injection probe with """ and a newline.\nSecond line.',
                    "parameters": [
                        {
                            "name": "class",
                            "type": "string",
                            "description": "A reserved word.",
                            "required": True,
                        }
                    ],
                }
            ],
        },
    )
    assert response.status_code == 200

    data = response.json()["data"]
    source = next(a for a in data["artifacts"] if a["type"] == "mcp-server")["content"]

    import ast

    module = ast.parse(source)
    functions = [n.name for n in module.body if isinstance(n, ast.FunctionDef)]
    assert all(name.isidentifier() for name in functions)
    assert any("must be identifiers" in w["message"] for w in data["warnings"])


def _write_bundle(target: Path, artifacts: list[dict[str, Any]]) -> Path:
    """Write the bundle out exactly as a user unzipping it would."""
    for artifact in artifacts:
        path = target / str(artifact["filename"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(artifact["content"]), encoding="utf-8")
    package = str(artifacts[0]["filename"]).split("/", 1)[0]
    return target / package


async def test_the_generated_server_completes_an_mcp_handshake(
    client: AsyncClient, tmp_path: Path
) -> None:
    """Start it and talk to it. A generator whose output does not run is a demo."""
    mcp_client = pytest.importorskip("mcp.client.stdio")
    from mcp import ClientSession, StdioServerParameters
    from mcp.types import TextContent

    response = await client.post(
        MCP, json={"server_name": "Ops Toolkit", "description": "Tools.", "tools": TOOLS}
    )
    assert response.status_code == 200
    root = _write_bundle(tmp_path, response.json()["data"]["artifacts"])

    parameters = StdioServerParameters(
        command=sys.executable,
        args=[str(root / "server.py")],
        cwd=str(root),
    )

    async with (
        mcp_client.stdio_client(parameters) as (read, write),
        ClientSession(read, write) as session,
    ):
        await session.initialize()

        listed = await session.list_tools()
        assert {tool.name for tool in listed.tools} == {"search_docs", "page_oncall"}

        described = {tool.name: tool for tool in listed.tools}
        assert described["search_docs"].description
        # The declared parameters survived into the wire schema. The attribute
        # is snake_case; the JSON key it serialises to is `inputSchema`, which
        # is what `function-schema` emits for the MCP target.
        schema = described["search_docs"].input_schema
        assert set(schema["properties"]) == {"query", "limit"}
        assert schema["required"] == ["query"]

        called = await session.call_tool("page_oncall", {"message": "disk full"})
        assert not called.is_error
        block = called.content[0]
        assert isinstance(block, TextContent)
        payload = json.loads(block.text)
        assert payload["tool"] == "page_oncall"
        assert payload["arguments"]["message"] == "disk full"


# ── agent-cost ───────────────────────────────────────────────────────────────


async def test_agent_cost_prices_a_roster_against_real_catalog_rows(
    client: AsyncClient,
) -> None:
    response = await client.post(
        COST,
        json={
            "agents": [
                {"role": "planner", "model_id": "gpt-4o-mini", "count": 1, "steps_per_task": 4},
                {"role": "worker", "model_id": "gpt-4o-mini", "count": 1, "steps_per_task": 4},
            ],
            "tasks_per_day": 100,
            "tool_count": 10,
            "retry_rate_pct": "15",
        },
    )
    assert response.status_code == 200

    data = response.json()["data"]
    assert _is_money(data["metrics"]["cost_per_month"])
    # The two lines a naive calculator drops are both present and non-zero.
    lines = {row["line"] for row in data["tables"]["breakdown"]}
    assert "Tool definitions" in lines
    assert float(data["metrics"]["schema_overhead_pct"]) > 0
    assert float(data["metrics"]["retry_premium_monthly"]) > 0
    # The models it read are named in the provenance block.
    assert data["provenance"]["sources"]


def _is_money(value: str) -> bool:
    """A decimal string, not a JSON number. D-08, carried to the wire."""
    return bool(re.fullmatch(r"\d+\.\d{2}", value))


async def test_an_unknown_model_is_a_404_not_a_zero(client: AsyncClient) -> None:
    response = await client.post(
        COST,
        json={"agents": [{"role": "planner", "model_id": "no-such-model", "steps_per_task": 1}]},
    )
    assert response.status_code == 404


# ── workflow-plan ────────────────────────────────────────────────────────────


def _mermaid_is_well_formed(diagram: str) -> bool:
    """Every edge references a node the diagram declares."""
    lines = [line.strip() for line in diagram.splitlines() if line.strip()]
    if lines[0] != "graph TD":
        return False
    declared = {match.group(1) for line in lines if (match := re.match(r"^(\w+)\[", line))}
    edges = [match.groups() for line in lines if (match := re.match(r"^(\w+) --> (\w+)$", line))]
    return bool(edges) and all(
        source in declared and target in declared for source, target in edges
    )


@pytest.mark.parametrize("coordination", ["sequential", "parallel", "hierarchical", "handoff"])
async def test_every_topology_returns_a_rendering_dag(
    client: AsyncClient, coordination: str
) -> None:
    response = await client.post(
        PLAN,
        json={
            "goal": "Triage inbound support tickets and draft a reply for each.",
            "coordination": coordination,
            "available_tools": ["search_docs", "lookup_order", "issue_refund"],
            "constraints": ["No customer data leaves our network."],
            "frontier_model_id": "gpt-4o",
            "fast_model_id": "gpt-4o-mini",
        },
    )
    assert response.status_code == 200

    data = response.json()["data"]
    # With no AI service configured the topology is still complete, and the run
    # says so rather than pretending otherwise.
    assert data["source"] == "rule_based"
    assert data["ai"] is None
    assert data["metrics"]["topology"] == coordination
    assert int(data["metrics"]["agents"]) >= 2

    diagram = next(a for a in data["artifacts"] if a["format"] == "mermaid")["content"]
    assert _mermaid_is_well_formed(diagram)

    # Every edge carries a written contract, and every node a responsibility.
    assert len(data["tables"]["contracts"]) == int(data["metrics"]["handoffs"])
    assert all(row["responsibility"] for row in data["tables"]["nodes"])
    assert data["tables"]["failure_modes"]


async def test_the_plan_recommends_a_framework_from_the_catalog(
    client: AsyncClient,
) -> None:
    response = await client.post(
        PLAN,
        json={
            "goal": "Research a company and produce a one-page brief.",
            "coordination": "parallel",
            "available_tools": ["web_search", "fetch_page", "summarise"],
            "frontier_model_id": "gpt-4o",
        },
    )
    data = response.json()["data"]

    # A real catalog row, not a hardcoded string — and never a `caution` one.
    assert data["metrics"]["framework"] in {"LangGraph", "Claude Agent SDK", "Pydantic AI"}


async def test_pricing_every_node_on_the_frontier_model_is_flagged(
    client: AsyncClient,
) -> None:
    response = await client.post(
        PLAN,
        json={
            "goal": "Summarise the overnight alerts into a single digest.",
            "coordination": "sequential",
            "available_tools": ["fetch_alerts"],
            "frontier_model_id": "gpt-4o",
        },
    )
    data = response.json()["data"]

    assert any(w["field"] == "fast_model_id" for w in data["warnings"])


# ── function-schema ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("target", ["openai", "anthropic", "json-schema", "mcp"])
async def test_function_schema_emits_valid_output_for_every_target(
    client: AsyncClient, target: str
) -> None:
    response = await client.post(SCHEMA, json={"tools": TOOLS, "target": target})
    assert response.status_code == 200

    data = response.json()["data"]
    assert data["metrics"]["valid"] == "yes"

    emitted = json.loads(data["artifacts"][0]["content"])
    assert len(emitted) == 2


# ── rate-limits ──────────────────────────────────────────────────────────────


async def test_rate_limits_names_the_constraint_that_actually_binds(
    client: AsyncClient,
) -> None:
    response = await client.post(
        LIMITS,
        json={
            "provider": "anthropic",
            "tier": "tier-1",
            "requests_per_min": 10,
            "input_tokens_per_request": 2000,
            "output_tokens_per_request": 1000,
            "concurrency": 16,
            "avg_request_seconds": "4",
        },
    )
    assert response.status_code == 200

    data = response.json()["data"]
    assert data["metrics"]["binding_constraint"] == "Output tokens per minute"
    assert data["metrics"]["recommended_tier"] == "Tier 2"
    assert data["tables"]["backoff"]


async def test_an_unpublished_tier_is_a_field_error(client: AsyncClient) -> None:
    response = await client.post(
        LIMITS, json={"provider": "anthropic", "tier": "tier-9", "requests_per_min": 10}
    )
    assert response.status_code == 422

    body = response.json()["error"]
    assert body["details"]["fields"][0]["path"] == "tier"


# ── the shared engine ────────────────────────────────────────────────────────


async def test_every_wf3_run_is_logged_and_reopenable(client: AsyncClient) -> None:
    """The same treatment as every other tool — run logging, provenance, and
    the seven-key envelope — without the endpoints having to remember any of it.
    """
    response = await client.post(SCHEMA, json={"tools": TOOLS, "target": "anthropic"})
    run_id = response.json()["data"]["run_id"]

    reopened = await client.get(f"/api/v1/runs/{run_id}")
    assert reopened.status_code == 200
    assert reopened.json()["data"]["tool_slug"] == "function-schema"


async def test_the_workflow_hub_lists_recent_wf3_runs(client: AsyncClient) -> None:
    await client.post(SCHEMA, json={"tools": TOOLS, "target": "mcp"})

    response = await client.get("/api/v1/runs", params={"workflow": "agents"})
    assert response.status_code == 200
    slugs = {run["tool_slug"] for run in response.json()["data"]}
    assert "function-schema" in slugs
