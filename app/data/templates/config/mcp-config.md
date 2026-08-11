---
title: MCP Client Configuration
category: config
difficulty: beginner
summary: >
  The config block that points Claude Desktop, Claude Code, or Cursor at an MCP
  server — with the two mistakes that account for most "it will not connect".
use_cases: [agents, automation]
tags: [mcp, config, claude, cursor]
related_tools: [mcp-config, function-schema]
---

Every MCP client reads roughly the same shape. The differences are where the
file lives and what the top-level key is called.

## Where the file lives

| Client | Path |
| --- | --- |
| Claude Desktop (macOS) | `~/Library/Application Support/Claude/claude_desktop_config.json` |
| Claude Desktop (Windows) | `%APPDATA%\Claude\claude_desktop_config.json` |
| Claude Code | `.mcp.json` in the project root |
| Cursor | `.cursor/mcp.json` in the project root |

```json path=claude_desktop_config.json
{
  "mcpServers": {
    "example-tools": {
      "command": "uv",
      "args": ["--directory", "/absolute/path/to/server", "run", "python", "-m", "server"],
      "env": {
        "API_KEY": "sk-replace-me"
      }
    }
  }
}
```

```json path=.mcp.json
{
  "mcpServers": {
    "example-tools": {
      "command": "uv",
      "args": ["--directory", "./mcp-server", "run", "python", "-m", "server"],
      "env": {}
    },
    "remote-tools": {
      "type": "http",
      "url": "https://tools.example.com/mcp",
      "headers": {
        "Authorization": "Bearer ${TOOLS_TOKEN}"
      }
    }
  }
}
```

## The two mistakes

**A relative path in a desktop client.** Claude Desktop does not launch from
your project directory, so `./server` resolves somewhere you did not expect and
the server never starts. Use an absolute path there. Project-scoped clients
like Claude Code and Cursor do run from the project root, so relative paths are
fine — and better, because the file is shared with the team.

**A command that is not on the launcher's PATH.** A desktop app started from
the Dock or the Start menu does not inherit the PATH from your shell, so `uv`
or `node` may simply not be found. If the server will not start and the logs
say nothing, use the absolute path to the binary — `which uv` gives it to you.

## Checking it

Restart the client fully; most read this file only at launch. If the tools do
not appear, run the server by hand first:

```bash
uv --directory /absolute/path/to/server run python -m server
```

A server that fails there will fail identically under the client, with less to
read.

## Secrets

Nothing in this file should be a real credential in a repository. Project-scoped
configs get committed, so use `${VAR}` interpolation where the client supports
it and keep the value in your environment.
