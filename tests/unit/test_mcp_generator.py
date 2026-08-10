"""The MCP generator, checked against the things that make it a generator
rather than a template.

A code generator is a real injection surface, so the hostile cases are not a
footnote here — a tool name carrying a quote, a newline, a triple quote, or a
Python keyword all have to produce a file that parses as exactly what was
intended and nothing more.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from app.services import mcp_generator
from app.services.mcp_generator import generate, safe_identifier


def _tool(name: str = "search_docs", **overrides: Any) -> dict[str, Any]:
    tool: dict[str, Any] = {
        "name": name,
        "description": "Search the internal handbook and return matching passages.",
        "parameters": [
            {
                "name": "query",
                "type": "string",
                "description": "What to look for.",
                "required": True,
            }
        ],
    }
    tool.update(overrides)
    return tool


def _server_source(bundle: mcp_generator.Bundle) -> str:
    return next(a.content for a in bundle.artifacts if a.type == "mcp-server")


def _files(bundle: mcp_generator.Bundle) -> dict[str, str]:
    return {artifact.filename.split("/", 1)[1]: artifact.content for artifact in bundle.artifacts}


# ── the bundle ───────────────────────────────────────────────────────────────


def test_the_bundle_is_six_files() -> None:
    bundle = generate(server_name="Ops", description="Tools.", tools=[_tool()])

    assert set(_files(bundle)) == {
        "server.py",
        "pyproject.toml",
        "README.md",
        ".env.example",
        "tests/test_server.py",
        "claude_desktop_config.json",
    }


def test_the_generated_server_parses() -> None:
    bundle = generate(
        server_name="Ops",
        description="Tools.",
        tools=[_tool("search_docs"), _tool("page_oncall")],
        resources=[{"name": "runbook", "uri": "docs://runbook", "description": "The runbook."}],
        prompts=[{"name": "postmortem", "description": "Draft one.", "template": "Write: "}],
    )

    module = ast.parse(_server_source(bundle))
    functions = {node.name for node in module.body if isinstance(node, ast.FunctionDef)}
    assert {"search_docs", "page_oncall", "runbook", "postmortem", "main"} <= functions


def test_the_client_config_is_valid_json_and_names_the_server() -> None:
    bundle = generate(server_name="Ops Toolkit", description="", tools=[_tool()])
    config = json.loads(_files(bundle)["claude_desktop_config.json"])

    assert list(config["mcpServers"]) == ["ops-toolkit"]
    # An absolute path, because Claude Desktop launches with an unspecified cwd.
    assert "/absolute/path/to/" in config["mcpServers"]["ops-toolkit"]["args"][1]


def test_the_readme_states_the_spec_version_it_targets() -> None:
    bundle = generate(server_name="Ops", description="", tools=[_tool()])

    assert mcp_generator.MCP_SPEC_VERSION in _files(bundle)["README.md"]


def test_the_sdk_dependency_is_pinned_to_a_major_version() -> None:
    """SDK 2.0 removed `FastMCP` outright. An unpinned bundle is one that stops
    running on a release the user never asked for."""
    bundle = generate(server_name="Ops", description="", tools=[_tool()])

    assert mcp_generator.MCP_SDK_REQUIREMENT in _files(bundle)["pyproject.toml"]
    assert "<3" in mcp_generator.MCP_SDK_REQUIREMENT


# ── injection ────────────────────────────────────────────────────────────────

HOSTILE_NAMES = [
    '"; import os; os.system("echo pwned") #',
    'x"""\nimport os\nos.system("echo pwned")\n"""',
    "name with spaces",
    "class",  # a Python keyword
    "123_leading_digit",
    "",
    "\n\n\n",
    "тест-имя",
]


def _executable_names(module: ast.Module) -> set[str]:
    """Everything the module actually imports or calls by attribute.

    Asserted against the parse tree rather than the text, because hostile input
    *does* appear verbatim in the generated file — inside the module docstring
    and inside description literals, where it is inert. Grepping the source
    cannot tell those apart from an executable position; the AST can.
    """
    names: set[str] = set()
    for node in ast.walk(module):
        if isinstance(node, ast.Import | ast.ImportFrom):
            names |= {alias.name for alias in node.names}
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            names.add(node.func.attr)
    return names


EMITTED_NAMES = {
    "annotations",
    "json",
    "os",
    "Annotated",
    "Any",
    "MCPServer",
    "Field",
    # Attribute calls the generator itself emits.
    "dumps",
    "get",
    "tool",
    "resource",
    "prompt",
    "run",
}


@pytest.mark.parametrize("hostile", HOSTILE_NAMES)
def test_a_hostile_tool_name_still_produces_a_file_that_parses(hostile: str) -> None:
    bundle = generate(
        server_name=hostile,
        description=hostile,
        tools=[_tool(hostile, description=hostile)],
    )

    module = ast.parse(_server_source(bundle))

    # Nothing the caller supplied reached an import or a call position.
    assert _executable_names(module) <= EMITTED_NAMES

    # Whatever arrived, the function it produced is a plain identifier.
    functions = [node.name for node in module.body if isinstance(node, ast.FunctionDef)]
    assert all(name.isidentifier() and not name.startswith("__") for name in functions)


def test_a_description_containing_a_triple_quote_cannot_end_a_string() -> None:
    """The specific reason descriptions are decorator arguments and never
    docstrings: a triple quote would close the docstring and the rest of the
    file would be parsed as code."""
    payload = 'Normal text """ import os; os.system("echo pwned") """ more text'
    bundle = generate(server_name="Ops", description="", tools=[_tool(description=payload)])
    module = ast.parse(_server_source(bundle))

    assert _executable_names(module) <= EMITTED_NAMES
    assert "system" not in _executable_names(module)
    # The payload survives intact as the decorator's `description`, which is
    # the whole point — nothing is dropped or truncated, it is just data.
    assert _tool_descriptions(module) == [payload]


def test_a_long_multi_line_description_round_trips_exactly() -> None:
    """A description too long for one line is split across several literals.
    Splitting must not reflow it — a description written as a short list comes
    back as a flattened paragraph if the generator wraps instead of slicing.
    """
    payload = (
        "Search the handbook.\n\n"
        "Modes:\n"
        "  - fast:    lexical only, single digit milliseconds\n"
        "  - careful: adds a semantic pass over the top candidates\n\n"
        + "Trailing prose that pushes this well past any single line budget. "
        * 3
    )
    bundle = generate(server_name="Ops", description="", tools=[_tool(description=payload)])
    module = ast.parse(_server_source(bundle))

    # Stripped at the edges, untouched in the middle.
    assert _tool_descriptions(module) == [payload.strip()]
    for line in _server_source(bundle).splitlines():
        assert len(line) <= 88, line


def _tool_descriptions(module: ast.Module) -> list[str]:
    """Read the `description=` argument back out of each `@server.tool(...)`."""
    found: list[str] = []
    for node in module.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Call):
                continue
            for word in decorator.keywords:
                if word.arg == "description":
                    found.append(ast.literal_eval(word.value))
    return found


def test_a_keyword_parameter_name_is_de_keyworded() -> None:
    bundle = generate(
        server_name="Ops",
        description="",
        tools=[
            _tool(
                parameters=[
                    {"name": "class", "type": "string", "description": "A class.", "required": True}
                ]
            )
        ],
    )
    source = _server_source(bundle)

    ast.parse(source)
    assert "class_:" in source


def test_duplicate_tool_names_become_distinct_functions() -> None:
    bundle = generate(server_name="Ops", description="", tools=[_tool("search"), _tool("search")])
    module = ast.parse(_server_source(bundle))

    names = [node.name for node in module.body if isinstance(node, ast.FunctionDef)]
    assert len(names) == len(set(names))
    assert bundle.tool_names == ["search", "search_2"]


def test_safe_identifier_never_returns_something_unusable() -> None:
    for raw in [*HOSTILE_NAMES, "def", "None", "match", "_", "__init__", "a" * 200]:
        derived = safe_identifier(raw)
        assert derived.isidentifier()
        assert not derived.startswith("__")
        assert len(derived) <= 64


# ── parameter ordering ───────────────────────────────────────────────────────


def test_optional_parameters_are_emitted_after_required_ones() -> None:
    """Python forbids a non-default parameter after a defaulted one, so the
    user's ordering cannot be preserved verbatim."""
    bundle = generate(
        server_name="Ops",
        description="",
        tools=[
            _tool(
                parameters=[
                    {"name": "limit", "type": "integer", "description": "Max.", "required": False},
                    {"name": "query", "type": "string", "description": "Text.", "required": True},
                ]
            )
        ],
    )

    module = ast.parse(_server_source(bundle))
    function = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "search_docs"
    )
    assert [arg.arg for arg in function.args.args] == ["query", "limit"]


# ── ruff ─────────────────────────────────────────────────────────────────────


def _ruff() -> list[str] | None:
    candidate = Path(sys.executable).parent / "ruff.exe"
    if candidate.exists():
        return [str(candidate)]
    return [sys.executable, "-m", "ruff"]


def test_the_generated_server_is_ruff_clean(tmp_path: Path) -> None:
    """Real bug rules, not formatting: `F` catches the unused import and the
    undefined name a generator produces when it emits an import conditionally
    and then uses it unconditionally. `E9` catches anything that does not parse.
    """
    bundle = generate(
        server_name="Ops Toolkit",
        description="Tools for the on-call rotation.",
        tools=[
            _tool("search docs"),
            _tool(
                "page_oncall",
                parameters=[
                    {"name": "message", "type": "string", "description": "What.", "required": True},
                    {"name": "tags", "type": "array", "description": "Labels.", "required": False},
                    {"name": "meta", "type": "object", "description": "Extra.", "required": False},
                ],
            ),
        ],
        auth="api-key",
    )
    target = tmp_path / "server.py"
    target.write_text(_server_source(bundle), encoding="utf-8")

    result = subprocess.run(  # noqa: S603
        [*(_ruff() or []), "check", "--isolated", "--select", "F,E9,B,I", str(target)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_a_server_with_no_auth_does_not_import_os(tmp_path: Path) -> None:
    bundle = generate(server_name="Ops", description="", tools=[_tool()], auth="none")

    assert "import os" not in _server_source(bundle)


# ── warnings ─────────────────────────────────────────────────────────────────


def test_an_http_transport_is_flagged_as_unauthenticated() -> None:
    bundle = generate(
        server_name="Ops", description="", tools=[_tool()], transport="streamable-http"
    )

    assert any(w.field == "transport" and w.level == "warning" for w in bundle.warnings)


def test_a_thin_description_is_flagged_because_it_is_what_the_model_selects_on() -> None:
    bundle = generate(server_name="Ops", description="", tools=[_tool(description="Search.")])

    assert any("what the model selects on" in w.message for w in bundle.warnings)


def test_the_generator_is_checked_against_the_installed_sdk() -> None:
    """This is the check that catches the next breaking SDK release, rather
    than a user finding out that the generated import no longer resolves."""
    pytest.importorskip("mcp")
    from mcp.server import MCPServer

    for name in ("tool", "resource", "prompt", "run", "list_tools", "call_tool"):
        assert hasattr(MCPServer, name), f"MCPServer.{name} is gone — the generator emits it"

    bundle = generate(server_name="Ops", description="", tools=[_tool()])
    assert not [w for w in bundle.warnings if "no longer exposes" in w.message]
