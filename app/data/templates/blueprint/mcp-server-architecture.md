---
title: MCP Server Blueprint
category: blueprint
difficulty: intermediate
summary: >
  How to expose your systems to an assistant as tools: transport, auth, schema
  design, and the boundary questions that decide what you should not expose.
use_cases: [agents, automation]
tags: [mcp, tools, integration, protocol]
related_tools: [mcp-config, function-schema, rate-limits]
---

MCP turns your internal systems into tools an assistant can call. The protocol
is the easy part; the design question is what belongs on the other side of it.

## Transport

| Transport | When |
| --- | --- |
| `stdio` | Local, single user, the assistant launches the process. The default. |
| `streamable-http` | Remote, multi-user, you control the endpoint. |
| `sse` | Legacy remote. Prefer streamable-http for anything new. |

Start with stdio. It has no network surface, no auth to get wrong, and it is
what every client supports.

## Tool design

**Name tools by what they do, not by the endpoint they wrap.** `find_customer`
is a tool an assistant can select correctly; `get_v2_customer_record` is one it
will use in the wrong situation.

**Descriptions are the interface.** The model chooses a tool from its
description alone. State when to use it *and when not to* — "search orders by
customer email; not for order status, use `get_order` for that" prevents a
whole class of wrong calls.

**Keep parameter counts low and types narrow.** An enum with four values is
selected correctly far more often than a free string with a documented format.

## What not to expose

The boundary question is not "can I" but "what happens on the worst call the
model could make".

- **No unbounded reads.** A tool that can return the whole table will, and it
  will fill the context window with it. Paginate, and cap the page.
- **No destructive verbs without a confirmation step.** Delete and refund
  belong behind a two-call flow, not behind one tool the model can select.
- **Nothing whose auth is the model's to hold.** The server authenticates as
  itself against your systems; the assistant never sees a credential.

## Rate limits

An assistant will retry, and it will retry faster than a human. Rate limit per
client at the server rather than trusting the caller, and return a structured
error the model can act on rather than a 500 it will simply try again.

## Generating one

The Agent and MCP Builder emits a runnable Python server with tests, pinned to
a major SDK version. Generated code that does not run is a demo — the generator
AST-parses its output and the test suite starts the server and handshakes
against it.
