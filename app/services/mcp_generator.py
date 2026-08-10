"""MCP server generation (WF3).

The bar for this tool is that the bundle it emits **runs**. Not that it looks
like an MCP server — the previous build emitted a string template that looked
like one, and the first person to download it found out otherwise. Here the
generated Python is AST-parsed before it is returned, the emitted test file
exercises every generated tool, and the integration suite starts the server in
a subprocess and completes an MCP handshake against it.

Two rules keep the generator from being an injection surface, which a code
generator genuinely is rather than a box to tick (`PRD.md` §22):

**Nothing user-supplied reaches an executable position.** Identifiers are
derived — sanitised to a valid Python identifier, de-keyworded, de-duplicated —
never interpolated. A tool named `"; os.system("rm -rf /") #` becomes
`os_system_rm_rf`, and the original is preserved in the description where it
cannot execute.

**Every string is emitted through `json.dumps`.** JSON string escaping is a
subset of Python's, so the output is always a valid Python literal no matter
what quotes, backslashes, or newlines the input contained. Descriptions in
particular never become docstrings — a triple quote in a description would end
one, and the rest of the file would be parsed as code. They are passed as
`description=` arguments instead, which is both safe and the SDK's own idiom.
"""

from __future__ import annotations

import ast
import json
import keyword
import re
import textwrap
from typing import Any, Final, NamedTuple

from app.schemas.tools import Artifact, ToolWarning

#: The spec revision the generated server targets. Stated in the README
#: because MCP is versioned and moving, which `PRD.md` §23 lists as a known
#: risk — a config with no spec version is undebuggable a year from now.
MCP_SPEC_VERSION: Final = "2025-06-18"

#: Pinned to a major version. A generated bundle that installs a different SDK
#: next month is a bundle that stops running for reasons the user cannot see —
#: and this is not hypothetical here: SDK 2.0 removed `FastMCP` outright and
#: replaced it with `MCPServer`, so an unpinned bundle generated against 1.x
#: fails on import today. `_sdk_interface_check` is what catches the next one.
MCP_SDK_REQUIREMENT: Final = "mcp>=2,<3"

#: The server class and its import path, in one place. Both appear in the
#: generated module and in the interface check, and the two disagreeing is
#: exactly how a generator starts emitting code for an SDK that no longer
#: exists.
SDK_IMPORT: Final = "from mcp.server import MCPServer"
SDK_CLASS: Final = "MCPServer"

TRANSPORTS: Final = ("stdio", "sse", "streamable-http")
AUTH_STYLES: Final = ("none", "api-key", "bearer")

PY_TYPES: Final[dict[str, str]] = {
    "string": "str",
    "integer": "int",
    "number": "float",
    "boolean": "bool",
    "array": "list[str]",
    "object": "dict[str, Any]",
}

MAX_LINE: Final = 88


class Bundle(NamedTuple):
    artifacts: list[Artifact]
    warnings: list[ToolWarning]
    tool_names: list[str]


# ── safety primitives ────────────────────────────────────────────────────────


def safe_identifier(raw: str, *, fallback: str = "tool") -> str:
    """A valid, non-keyword Python identifier derived from arbitrary text.

    Derived, not validated-and-rejected: the user typed a tool name, and
    turning "Search Docs" into `search_docs` is what they meant. What matters
    is that nothing reaching an identifier position was ever chosen by the
    caller.
    """
    cleaned = re.sub(r"[^0-9a-zA-Z_]+", "_", raw).strip("_").lower()
    cleaned = re.sub(r"_{2,}", "_", cleaned)
    if not cleaned or cleaned[0].isdigit():
        cleaned = f"{fallback}_{cleaned}" if cleaned else fallback
    if keyword.iskeyword(cleaned) or keyword.issoftkeyword(cleaned):
        cleaned = f"{cleaned}_"
    return cleaned[:64]


def _lit(text: str) -> str:
    """A Python string literal. JSON escaping is a subset of Python's."""
    return json.dumps(str(text), ensure_ascii=False)


def _slice(text: str, budget: int) -> list[str]:
    """Split into pieces of at most `budget` characters, preferring a space.

    Slicing rather than `textwrap.wrap`, because wrapping *reflows*: it
    collapses runs of whitespace and drops newlines, so a description written
    as a short list comes back as one flattened paragraph. `"".join` of these
    pieces is the original string, character for character.
    """
    pieces: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + budget)
        if end < len(text):
            space = text.rfind(" ", start + 1, end + 1)
            if space > start:
                end = space + 1
        pieces.append(text[start:end])
        start = end
    return pieces or [""]


def _wrapped_lit(text: str, *, indent: int) -> str:
    """A literal that fits the line budget, split across lines if it must.

    Implicit concatenation inside parentheses, which is ordinary Python and
    keeps the generated file inside a line-length limit without altering a
    character of what the user wrote.
    """
    single = _lit(text)
    if indent + len(single) <= MAX_LINE:
        return single

    # Escaping can double a character's width, so the budget is halved rather
    # than being a guess a backslash-heavy description would blow straight
    # through.
    budget = max(16, (MAX_LINE - indent - 8) // 2)
    pad = " " * (indent + 4)
    lines = [f"{pad}{_lit(piece)}" for piece in _slice(str(text), budget)]
    return "(\n" + "\n".join(lines) + f"\n{' ' * indent})"


def _renamed(tools: list[_Tool]) -> list[_Tool]:
    """Tools whose wire name had to be derived from what the user typed."""
    return [tool for tool in tools if tool.original_name and tool.wire_name != tool.original_name]


def _unique(names: list[str]) -> list[str]:
    """De-duplicate while preserving order. Two tools cannot share a function."""
    seen: dict[str, int] = {}
    result: list[str] = []
    for name in names:
        if name in seen:
            seen[name] += 1
            result.append(f"{name}_{seen[name]}")
        else:
            seen[name] = 1
            result.append(name)
    return result


# ── generation ───────────────────────────────────────────────────────────────


class _Tool(NamedTuple):
    func: str
    wire_name: str
    original_name: str
    description: str
    params: list[dict[str, Any]]


def _prepare(tools: list[dict[str, Any]]) -> list[_Tool]:
    funcs = _unique(
        [safe_identifier(str(tool.get("name") or ""), fallback="tool") for tool in tools]
    )

    prepared: list[_Tool] = []
    for func, tool in zip(funcs, tools, strict=True):
        params: list[dict[str, Any]] = []
        param_names = _unique(
            [
                safe_identifier(str(param.get("name") or ""), fallback="arg")
                for param in tool.get("parameters", [])
            ]
        )
        for name, param in zip(param_names, tool.get("parameters", []), strict=True):
            params.append(
                {
                    "name": name,
                    "original": str(param.get("name") or ""),
                    "type": PY_TYPES.get(str(param.get("type") or "string"), "str"),
                    "json_type": str(param.get("type") or "string"),
                    "description": str(param.get("description") or "").strip(),
                    "required": bool(param.get("required", True)),
                }
            )
        # Required first: Python forbids a non-default parameter after a
        # defaulted one, and the user's ordering is not worth a SyntaxError.
        params.sort(key=lambda item: not item["required"])

        prepared.append(
            _Tool(
                func=func,
                wire_name=func,
                original_name=str(tool.get("name") or ""),
                description=str(tool.get("description") or "").strip(),
                params=params,
            )
        )
    return prepared


def generate(
    *,
    server_name: str,
    description: str,
    tools: list[dict[str, Any]],
    transport: str = "stdio",
    auth: str = "none",
    resources: list[dict[str, Any]] | None = None,
    prompts: list[dict[str, Any]] | None = None,
) -> Bundle:
    """The full bundle: server, packaging, tests, README, and a paste-ready config."""
    slug = safe_identifier(server_name, fallback="server").replace("_", "-")
    package = f"mcp-server-{slug}"
    prepared = _prepare(tools)
    resources = resources or []
    prompts = prompts or []

    server_py = _server_module(
        server_name=server_name,
        slug=slug,
        description=description,
        tools=prepared,
        transport=transport,
        auth=auth,
        resources=resources,
        prompts=prompts,
    )

    warnings = _warnings(
        transport=transport, auth=auth, tools=prepared, source=server_py, slug=slug
    )

    artifacts = [
        Artifact(
            type="mcp-server",
            format="code",
            filename=f"{package}/server.py",
            content=server_py,
            language="python",
        ),
        Artifact(
            type="mcp-pyproject",
            format="text",
            filename=f"{package}/pyproject.toml",
            content=_pyproject(slug=slug, description=description),
            language="toml",
        ),
        Artifact(
            type="mcp-readme",
            format="markdown",
            filename=f"{package}/README.md",
            content=_readme(
                server_name=server_name,
                package=package,
                slug=slug,
                description=description,
                tools=prepared,
                transport=transport,
                auth=auth,
            ),
        ),
        Artifact(
            type="mcp-env",
            format="text",
            filename=f"{package}/.env.example",
            content=_env_example(auth=auth, transport=transport),
        ),
        Artifact(
            type="mcp-tests",
            format="code",
            filename=f"{package}/tests/test_server.py",
            content=_tests(tools=prepared),
            language="python",
        ),
        Artifact(
            type="mcp-client-config",
            format="json",
            filename=f"{package}/claude_desktop_config.json",
            content=_client_config(slug=slug, package=package, auth=auth),
            language="json",
        ),
    ]

    return Bundle(artifacts=artifacts, warnings=warnings, tool_names=[t.func for t in prepared])


def _server_module(
    *,
    server_name: str,
    slug: str,
    description: str,
    tools: list[_Tool],
    transport: str,
    auth: str,
    resources: list[dict[str, Any]],
    prompts: list[dict[str, Any]],
) -> str:
    needs_any = any(param["type"] == "dict[str, Any]" for tool in tools for param in tool.params)

    header = [
        '"""',
        f"MCP server: {_safe_comment(server_name)}",
        "",
        *(textwrap.wrap(_safe_comment(description), width=MAX_LINE) if description else []),
        "",
        "Generated by StackForge. The tool bodies are stubs that echo their",
        "arguments so the server runs and connects before you implement it —",
        "replace each one with the real call.",
        "",
        f"Targets MCP spec {MCP_SPEC_VERSION}.",
        '"""',
        "",
        "from __future__ import annotations",
        "",
        "import json",
        # Only when something reads the environment. An unused import is a Ruff
        # failure in the user's repository, caused by us.
        *(["import os"] if auth != "none" else []),
    ]
    typing_imports = ["Annotated"] + (["Any"] if needs_any else [])
    header += [
        f"from typing import {', '.join(typing_imports)}",
        "",
        SDK_IMPORT,
        *(["from pydantic import Field"] if any(tool.params for tool in tools) else []),
        # One blank line after the import block, two before the first `def`.
        # Ruff's isort rule treats extra blank lines here as part of the block
        # and reports the file as unsorted, which would make every generated
        # bundle fail the user's own lint on day one.
        "",
        f"server = {SDK_CLASS}({_lit(server_name)})",
        "",
        "",
    ]

    body: list[str] = []

    if auth != "none":
        body += _auth_block(auth)

    for tool in tools:
        body += _tool_block(tool)

    for index, resource in enumerate(resources):
        body += _resource_block(resource, index)

    for index, prompt in enumerate(prompts):
        body += _prompt_block(prompt, index)

    footer = [
        "def main() -> None:",
        f"    server.run(transport={_lit(transport)})",
        "",
        "",
        'if __name__ == "__main__":',
        "    main()",
        "",
    ]

    return "\n".join([*header, *body, *footer])


def _auth_block(auth: str) -> list[str]:
    env_var = "MCP_API_KEY" if auth == "api-key" else "MCP_BEARER_TOKEN"
    return [
        "def _credential() -> str:",
        '    """The upstream credential, read from the environment.',
        "",
        "    Read at call time rather than import time so a missing value is a",
        "    tool error the client can see, not a server that refuses to start",
        "    with no explanation in the host application's log.",
        '    """',
        f"    value = os.environ.get({_lit(env_var)})",
        "    if not value:",
        f"        raise RuntimeError({_lit(f'{env_var} is not set. See .env.example.')})",
        "    return value",
        "",
        "",
    ]


def _tool_block(tool: _Tool) -> list[str]:
    lines = [
        "@server.tool(",
        f"    name={_lit(tool.wire_name)},",
        # The description is an argument, never a docstring. A triple quote in
        # user text would close a docstring and turn the remainder of the file
        # into executable code; as an argument it is inert.
        f"    description={_wrapped_lit(tool.description or tool.original_name, indent=16)},",
        ")",
    ]

    if tool.params:
        lines.append(f"def {tool.func}(")
        for param in tool.params:
            # The union goes *inside* `Annotated`, not around it: pydantic reads
            # both, but `Annotated[int, ...] | None` puts the optionality where
            # a reader has to hunt for it.
            inner = param["type"] if param["required"] else f"{param['type']} | None"
            described = _lit(param["description"] or param["original"] or param["name"])
            annotation = f"Annotated[{inner}, Field(description={described})]"
            suffix = "," if param["required"] else " = None,"
            lines.append(f"    {param['name']}: {annotation}{suffix}")
        lines.append(") -> str:")
    else:
        lines.append(f"def {tool.func}() -> str:")

    received = ", ".join(f"{_lit(param['name'])}: {param['name']}" for param in tool.params)
    lines += [
        "    # TODO: implement. Until then this echoes its arguments, which is",
        "    # enough to verify the connection end to end from the client.",
        "    return json.dumps(",
        "        {",
        f'            "tool": {_lit(tool.wire_name)},',
        f'            "arguments": {{{received}}},',
        '            "status": "not_implemented",',
        "        },",
        "        default=str,",
        "    )",
        "",
        "",
    ]
    return lines


def _resource_block(resource: dict[str, Any], index: int) -> list[str]:
    uri = _safe_uri(str(resource.get("uri") or ""), index)
    func = safe_identifier(str(resource.get("name") or f"resource_{index}"), fallback="resource")
    return [
        f"@server.resource({_lit(uri)})",
        f"def {func}() -> str:",
        f"    # TODO: return the real contents of {_safe_comment(uri)}.",
        f"    return {_wrapped_lit(str(resource.get('description') or uri), indent=11)}",
        "",
        "",
    ]


def _prompt_block(prompt: dict[str, Any], index: int) -> list[str]:
    func = safe_identifier(str(prompt.get("name") or f"prompt_{index}"), fallback="prompt")
    template = str(prompt.get("template") or "").strip() or "Consider the following: "
    return [
        "@server.prompt(",
        f"    name={_lit(func)},",
        f"    description={_wrapped_lit(str(prompt.get('description') or func), indent=16)},",
        ")",
        f"def {func}(subject: str) -> str:",
        # Concatenation rather than an f-string: a template containing a brace
        # would make the f-string interpolate whatever the user typed.
        f"    return {_wrapped_lit(template, indent=11)} + subject",
        "",
        "",
    ]


def _safe_uri(raw: str, index: int) -> str:
    """A resource URI with only scheme-safe characters, always non-empty."""
    cleaned = re.sub(r"[^0-9a-zA-Z:/._-]+", "-", raw).strip("-")
    if "://" not in cleaned:
        cleaned = f"resource://{safe_identifier(cleaned or f'item_{index}', fallback='item')}"
    return cleaned[:200]


def _safe_comment(raw: str) -> str:
    """Text destined for a comment or docstring, stripped of anything that ends one."""
    return re.sub(r'["\\]|\s+', lambda m: " " if m.group().isspace() else "", str(raw))[:400]


def _pyproject(*, slug: str, description: str) -> str:
    return f"""[project]
name = "mcp-server-{slug}"
version = "0.1.0"
description = {_lit(_safe_comment(description) or f"MCP server: {slug}")}
requires-python = ">=3.10"
dependencies = [
    "{MCP_SDK_REQUIREMENT}",
]

[dependency-groups]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.24",
]

[project.scripts]
mcp-server-{slug} = "server:main"

[tool.pytest.ini_options]
asyncio_mode = "auto"

[tool.ruff]
line-length = {MAX_LINE}
"""


def _env_example(*, auth: str, transport: str) -> str:
    lines = ["# Copy to .env and fill in. Nothing here is committed.", ""]
    if auth == "api-key":
        lines += ["# Credential for the upstream service this server calls.", "MCP_API_KEY=", ""]
    elif auth == "bearer":
        lines += ["MCP_BEARER_TOKEN=", ""]
    if transport != "stdio":
        lines += ["# Bind address for the HTTP transport.", "HOST=127.0.0.1", "PORT=8000", ""]
    return "\n".join(lines)


def _client_config(*, slug: str, package: str, auth: str) -> str:
    env: dict[str, str] = {}
    if auth == "api-key":
        env["MCP_API_KEY"] = "your-key-here"
    elif auth == "bearer":
        # A placeholder for the user to replace, not a credential. Emitting a
        # real one here is the failure this file exists to avoid.
        env["MCP_BEARER_TOKEN"] = "your-token-here"  # noqa: S105

    entry: dict[str, Any] = {
        "command": "uv",
        # An absolute path, because Claude Desktop launches the server with an
        # unspecified working directory. A relative path here is the single
        # most common reason a working server never appears in the client.
        "args": ["--directory", f"/absolute/path/to/{package}", "run", "server.py"],
    }
    if env:
        entry["env"] = env

    return json.dumps({"mcpServers": {slug: entry}}, indent=2) + "\n"


def _tests(*, tools: list[_Tool]) -> str:
    """A test per generated tool, run against the server object itself.

    In-process rather than over a spawned transport: the thing worth asserting
    is that each tool is registered, described, and callable. A transport test
    would mostly be testing the SDK.
    """
    head = f'''"""Generated tests. One per tool, plus a registration check.

Run with `uv run pytest` from the bundle directory.
"""

from __future__ import annotations

from server import server

EXPECTED = {sorted(tool.wire_name for tool in tools)!r}


async def test_every_tool_is_registered() -> None:
    registered = {{tool.name for tool in await server.list_tools()}}
    assert registered == set(EXPECTED)


async def test_every_tool_has_a_description() -> None:
    for tool in await server.list_tools():
        assert tool.description, f"{{tool.name}} has no description"
'''

    blocks = [head]
    for tool in tools:
        args = {
            param["name"]: _example_value(param["json_type"])
            for param in tool.params
            if param["required"]
        }
        blocks.append(
            f"""

async def test_{tool.func}_is_callable() -> None:
    result = await server.call_tool({_lit(tool.wire_name)}, {args!r})
    assert not result.is_error
"""
        )
    return "".join(blocks)


def _example_value(json_type: str) -> Any:
    match json_type:
        case "integer":
            return 1
        case "number":
            return 1.0
        case "boolean":
            return True
        case "array":
            return ["example"]
        case "object":
            return {"key": "value"}
        case _:
            return "example"


def _readme(
    *,
    server_name: str,
    package: str,
    slug: str,
    description: str,
    tools: list[_Tool],
    transport: str,
    auth: str,
) -> str:
    tool_rows = "\n".join(
        f"| `{tool.wire_name}` | {', '.join(p['name'] for p in tool.params) or '—'} | "
        f"{(tool.description or '—').splitlines()[0][:120]} |"
        for tool in tools
    )
    renamed = _renamed(tools)
    renamed_section = (
        "### Renamed for the wire\n\n"
        "Tool names must be identifiers, so these were derived:\n\n"
        + "\n".join(f"- `{tool.original_name}` → `{tool.wire_name}`" for tool in renamed)
        + "\n"
        if renamed
        else ""
    )

    auth_section = {
        "none": (
            "This server takes no credentials. If the service it calls needs one, add it "
            "to `.env.example` and read it with `os.environ`."
        ),
        "api-key": (
            "Set `MCP_API_KEY` before starting. Over stdio the key protects the *upstream* "
            "service — it is not client authentication, because the client is whichever "
            "process spawned this one and is already trusted by the operating system."
        ),
        "bearer": (
            "Set `MCP_BEARER_TOKEN` before starting. Over stdio this protects the upstream "
            "service, not this server. If you switch to an HTTP transport you need real "
            "request authentication in front of it — the transport alone gives you none."
        ),
    }[auth]

    transport_note = {
        "stdio": (
            "`stdio` — the client starts this process and talks to it over pipes. This is "
            "what Claude Desktop uses."
        ),
        "sse": (
            "`sse` — served over HTTP. The process must be running before the client "
            "connects, and it is reachable by anything that can reach the port."
        ),
        "streamable-http": (
            "`streamable-http` — served over HTTP. The process must be running before the "
            "client connects, and it is reachable by anything that can reach the port."
        ),
    }[transport]

    return f"""# {server_name}

{description or "An MCP server generated by StackForge."}

Targets MCP specification **{MCP_SPEC_VERSION}**, built on `{MCP_SDK_REQUIREMENT}`.

## Install and run

```bash
cd {package}
uv sync
uv run server.py
```

Transport: {transport_note}

## Connect it to Claude Desktop

1. Open `claude_desktop_config.json` in this bundle and replace
   `/absolute/path/to/{package}` with the real path. Claude Desktop launches the
   server from an unspecified working directory, so a relative path will not work.
2. Merge that block into your own config:
   - macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
   - Windows: `%APPDATA%\\Claude\\claude_desktop_config.json`
3. Restart Claude Desktop. The tools appear under the connectors icon.

If they do not appear, run `uv run server.py` in a terminal first — a server that
fails to start shows up as an absent connector, not as an error.

## Tools

| Tool | Parameters | Description |
| --- | --- | --- |
{tool_rows}

{renamed_section}
## Authentication

{auth_section}

## Tests

```bash
uv run pytest
```

The generated tests assert that every tool is registered, described, and
callable. They pass against the stubs — they are a wiring check, and they stay
useful once you replace the bodies.

## What is not done

Every tool body is a stub that echoes its arguments and reports
`"status": "not_implemented"`. That is deliberate: the bundle runs and connects
before you have written anything, so the plumbing is verified separately from
the logic. Replace each body with the real call.
"""


def _warnings(
    *, transport: str, auth: str, tools: list[_Tool], source: str, slug: str
) -> list[ToolWarning]:
    warnings: list[ToolWarning] = []

    try:
        ast.parse(source)
    except SyntaxError as exc:  # pragma: no cover — a generator bug, not an input
        warnings.append(
            ToolWarning(
                level="critical",
                message=(
                    f"The generated server did not parse ({exc.msg} at line {exc.lineno}). "
                    f"This is a bug in StackForge — please flag it."
                ),
            )
        )

    interface = _sdk_interface_check()
    if interface:
        warnings.append(ToolWarning(level="warning", message=interface))

    renamed = _renamed(tools)
    if renamed:
        warnings.append(
            ToolWarning(
                level="info",
                message=(
                    "Tool names must be identifiers, so "
                    + ", ".join(f"{t.original_name!r} → {t.wire_name!r}" for t in renamed[:4])
                    + (" and others" if len(renamed) > 4 else "")
                    + ". The original text is kept in each description."
                ),
            )
        )

    if transport != "stdio":
        warnings.append(
            ToolWarning(
                level="warning",
                field="transport",
                message=(
                    "An HTTP transport exposes this server to anything that can reach the "
                    "port. The SDK does not authenticate requests for you — put real auth "
                    "in front of it before it leaves your machine."
                ),
            )
        )
    if auth != "none" and transport == "stdio":
        warnings.append(
            ToolWarning(
                level="info",
                message=(
                    "Over stdio the credential protects the upstream service, not this "
                    "server: the client is the process that spawned it and is already "
                    "trusted. That is fine — it is worth knowing which threat it covers."
                ),
            )
        )

    if len(tools) > 20:
        warnings.append(
            ToolWarning(
                level="warning",
                message=(
                    f"{len(tools)} tools are re-sent to the model on every turn. Past about "
                    f"20, selection accuracy drops and the definitions dominate input cost. "
                    f"Two focused servers usually beat one broad one."
                ),
            )
        )

    undescribed = [tool.wire_name for tool in tools if len(tool.description) < 15]
    if undescribed:
        warnings.append(
            ToolWarning(
                level="warning",
                message=(
                    "No usable description on "
                    + ", ".join(f"`{name}`" for name in undescribed[:5])
                    + ". The description is what the model selects on, so a thin one "
                    "produces a tool called at the wrong moments."
                ),
            )
        )

    return warnings


def _sdk_interface_check() -> str | None:
    """Check the generated code against the installed SDK's actual interface.

    Generating against a remembered API is how a generator quietly starts
    emitting code for a version that no longer exists. If the SDK is not
    installed here the check is skipped rather than faked — but it runs in CI,
    which is where a breaking SDK release needs to be caught.
    """
    try:
        from mcp.server import MCPServer
    except ImportError:
        return None

    expected = ("tool", "resource", "prompt", "run", "list_tools", "call_tool")
    missing = [name for name in expected if not hasattr(MCPServer, name)]
    if missing:
        return (
            f"The installed MCP SDK no longer exposes {', '.join(missing)} on "
            f"{SDK_CLASS}. The generated server targets spec {MCP_SPEC_VERSION} and may "
            f"not run against the newest SDK — please flag this."
        )
    return None
