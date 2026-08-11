---
title: Python MCP Server
category: code-starter
difficulty: intermediate
summary: >
  A minimal MCP server that runs, with a test that actually starts it and
  completes a handshake — the check that separates a working server from one
  that only looks like it.
use_cases: [agents, automation]
tags: [mcp, python, tools, protocol]
related_tools: [mcp-config, function-schema]
---

Three tools over stdio. The interesting file is the test: it launches the
server as a subprocess and completes a real MCP handshake, because a generated
server that has never been started is a server nobody knows runs.

## Running it

```bash
uv sync
uv run python -m server            # speaks MCP over stdio
uv run pytest                      # starts it and handshakes
```

Point a client at it with the config in the MCP Config template.

```python path=server.py
"""A minimal MCP server.

Every tool here does three things the protocol does not enforce and clients
depend on: it validates its own input, it returns a structured result rather
than prose, and its description says when *not* to use it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from mcp.server import MCPServer

server = MCPServer("example-tools")

EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


@dataclass(frozen=True)
class Customer:
    id: str
    email: str
    plan: str


CUSTOMERS = {
    "ada@example.com": Customer("cus_1", "ada@example.com", "pro"),
    "grace@example.com": Customer("cus_2", "grace@example.com", "free"),
}


@server.tool(
    description=(
        "Look up one customer by their exact email address. "
        "Use for account questions. Not for searching by name or partial "
        "email - this matches exactly or returns nothing."
    )
)
def find_customer(email: str) -> dict[str, object]:
    if not EMAIL.match(email):
        # A structured failure, not an exception and not a sentence. The model
        # has to be able to tell "no result" from "bad input" without parsing
        # English, or it will retry the same malformed call.
        return {"ok": False, "error": "invalid_email", "message": "Not an email address."}

    customer = CUSTOMERS.get(email.lower())
    if customer is None:
        return {"ok": True, "found": False}
    return {
        "ok": True,
        "found": True,
        "customer": {"id": customer.id, "email": customer.email, "plan": customer.plan},
    }


@server.tool(
    description=(
        "List customers on a plan, newest first. Returns at most `limit` rows. "
        "Always paginated - there is no way to fetch every customer at once."
    )
)
def list_customers(plan: str, limit: int = 20) -> dict[str, object]:
    if plan not in {"free", "pro", "team"}:
        return {"ok": False, "error": "unknown_plan", "allowed": ["free", "pro", "team"]}

    # Capped server-side as well as defaulted. A tool that can return the whole
    # table will, and it will fill the context window with it.
    rows = [c for c in CUSTOMERS.values() if c.plan == plan][: min(limit, 50)]
    return {
        "ok": True,
        "count": len(rows),
        "customers": [{"id": c.id, "email": c.email} for c in rows],
    }


@server.tool(
    description=(
        "Estimate the monthly cost of a plan for a number of seats. "
        "Arithmetic only - this does not read or change any account."
    )
)
def estimate_cost(plan: str, seats: int) -> dict[str, object]:
    prices = {"free": 0, "pro": 20, "team": 35}
    if plan not in prices:
        return {"ok": False, "error": "unknown_plan", "allowed": sorted(prices)}
    if seats < 1:
        return {"ok": False, "error": "invalid_seats"}
    return {"ok": True, "plan": plan, "seats": seats, "monthly_usd": prices[plan] * seats}


if __name__ == "__main__":
    server.run()
```

```python path=tests/test_server.py
"""Start the server and talk to it.

Importing the module and calling the functions proves the functions work. It
does not prove the server starts, registers the tools, or speaks the protocol -
and those are the three ways a generated server is broken.
"""

from __future__ import annotations

import sys

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


@pytest.fixture
def params() -> StdioServerParameters:
    return StdioServerParameters(command=sys.executable, args=["server.py"])


@pytest.mark.asyncio
async def test_every_tool_is_registered(params: StdioServerParameters) -> None:
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        names = {tool.name for tool in (await session.list_tools()).tools}

    assert names == {"find_customer", "list_customers", "estimate_cost"}


@pytest.mark.asyncio
async def test_every_tool_has_a_description(params: StdioServerParameters) -> None:
    """The description is the interface - the model picks from it alone."""
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        tools = (await session.list_tools()).tools

    for tool in tools:
        assert tool.description and len(tool.description) > 40, tool.name


@pytest.mark.asyncio
async def test_a_bad_argument_returns_structured_failure(
    params: StdioServerParameters,
) -> None:
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        result = await session.call_tool("find_customer", {"email": "not-an-email"})

    assert "invalid_email" in str(result.content)
```

```toml path=pyproject.toml
[project]
name = "example-mcp-server"
version = "0.1.0"
requires-python = ">=3.11"
# Pinned to a major version. An unpinned bundle installs a different SDK next
# month and stops running for reasons the user cannot see - SDK 2.0 removed
# FastMCP outright, so bundles generated against 1.x fail on import today.
dependencies = ["mcp>=2,<3"]

[dependency-groups]
dev = ["pytest>=8", "pytest-asyncio>=0.24"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
```
