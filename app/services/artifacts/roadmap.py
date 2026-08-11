"""The implementation roadmap.

Phases are derived from the roles the stack actually fills, in the order a
team would stand them up: storage before retrieval, retrieval before the
model, observability before anyone trusts what they see. A roadmap that lists
every role the *vocabulary* has — including ones this stack does not use — is
a checklist someone has to prune before it is useful.

Effort figures are ranges, and they are stated as a starting point rather than
an estimate. A generated roadmap that says "2 days" with no qualification gets
copied into a plan and then defended.
"""

from __future__ import annotations

from typing import Any, Final, NamedTuple

from app.schemas.tools import Artifact
from app.services.artifacts.sources import Source, StackSource

TYPE = "roadmap"


class Phase(NamedTuple):
    role_key: str
    title: str
    detail: str
    effort: str


#: Ordered by dependency, not by importance. Each phase names the role that
#: has to be present for it to apply; a stack without that role skips it.
PHASES: Final[tuple[Phase, ...]] = (
    Phase(
        "database",
        "Stand up application state",
        "Provision {name} and get migrations running in CI. Everything downstream "
        "assumes a schema that exists and a way to change it.",
        "1-2 days",
    ),
    Phase(
        "vector_db",
        "Provision the vector store",
        "Deploy {name}, decide the embedding dimension, and index a representative "
        "sample. Re-indexing later is cheap; discovering the dimension is wrong "
        "after ingesting everything is not.",
        "1-3 days",
    ),
    Phase(
        "cache",
        "Put the cache in front of the hot paths",
        "Add {name} for embedding reuse and rate limiting. Cache the embedding "
        "call before the completion call — it is the one that repeats.",
        "0.5-1 day",
    ),
    Phase(
        "llm",
        "Wire the model provider",
        "Connect {name} behind one interface, with a timeout and a fallback. The "
        "provider is the component most likely to be swapped, so nothing above it "
        "should name it.",
        "1-2 days",
    ),
    Phase(
        "framework",
        "Build the retrieval and orchestration layer",
        "Assemble the pipeline in {name}: chunking, retrieval, prompt assembly, "
        "and the response path. This is where the product actually lives.",
        "1-2 weeks",
    ),
    Phase(
        "orchestration",
        "Move the long work off the request path",
        "Run ingestion, re-indexing, and evaluation through {name}. Anything that "
        "can take longer than a request should not be inside one.",
        "2-4 days",
    ),
    Phase(
        "observability",
        "Instrument before you tune",
        "Trace every call through {name} with token counts and cost attached. "
        "Tuning a pipeline you cannot see is guessing with extra steps.",
        "1-2 days",
    ),
    Phase(
        "deployment",
        "Ship it",
        "Deploy on {name} with health checks, a rollback path, and secrets out of "
        "the image. Exercise the rollback once before you need it.",
        "2-5 days",
    ),
)

#: A phase every stack gets, regardless of shape.
CLOSING: Final = Phase(
    "",
    "Evaluate against real inputs",
    "Assemble a fixed evaluation set from real queries and score every change "
    "against it. Without one, every later decision is settled by whoever "
    "remembers the last demo most confidently.",
    "2-4 days",
)


def supports(source: Source) -> bool:
    return isinstance(source, StackSource)


def steps(source: StackSource) -> list[dict[str, Any]]:
    """The roadmap as rows, so the architecture document and the standalone
    artifact render the same list."""
    from app.services.stack_architect_service import ROLES, _role_category

    by_category = {tool.category: tool for tool in source.components}
    present = {
        role.key: by_category[_role_category(role, source.requirements)]
        for role in ROLES
        if _role_category(role, source.requirements) in by_category
    }

    rows: list[dict[str, Any]] = []
    for phase in PHASES:
        tool = present.get(phase.role_key)
        if tool is None:
            continue
        rows.append(
            {
                "title": phase.title,
                "detail": phase.detail.format(name=tool.name),
                "effort": phase.effort,
                "component": tool.name,
            }
        )

    rows.append(
        {
            "title": CLOSING.title,
            "detail": CLOSING.detail,
            "effort": CLOSING.effort,
            "component": "",
        }
    )
    return rows


def generate(source: StackSource) -> Artifact:
    rows = steps(source)
    body = "\n\n".join(
        f"### {index}. {row['title']}\n\n{row['detail']}\n\n_Starting point: {row['effort']}._"
        for index, row in enumerate(rows, 1)
    )
    total = len(rows)

    return Artifact(
        type=TYPE,
        format="markdown",
        filename="roadmap.md",
        content=f"""# Implementation roadmap — {source.title}

{total} phases, ordered by dependency rather than by importance: each one
assumes the ones above it are standing up.

The effort figures are starting points for a team that has built something like
this before, not estimates of your work. They exclude the discovery you will do
in phase one and the evaluation you will redo in every phase after it.

{body}

---

Generated by StackForge from the saved stack. The phases present are the roles
this stack actually fills — a role the stack does not use produces no phase,
rather than a checklist item to delete.
""",
        language="markdown",
    )
