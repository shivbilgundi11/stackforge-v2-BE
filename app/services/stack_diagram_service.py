"""Mermaid for a recommended stack.

Used as the diagram whether or not synthesis is available — the shape of a
stack is determined by its roles, which the engine already knows, so asking a
model to draw it would add a failure mode without adding information.

Node ids come from the role vocabulary, which is fixed in code, so nothing
user- or catalog-supplied ever reaches an identifier position. Labels are
quoted and stripped of the characters that break the parser: a tool name with
a bracket in it otherwise produces a diagram that silently fails to render.
"""

from __future__ import annotations

from itertools import pairwise
from typing import Final

from app.schemas.catalog import ToolOut
from app.services import diagram_theme
from app.services.stack_architect_service import ROLES, Requirements, _role_category

#: Left-to-right request path, then the supporting roles hanging off it. Drawn
#: as the request actually flows rather than as a category list, because the
#: question a reader brings to an architecture diagram is "what calls what".
#:
#: Each entry is a chain rather than a single edge, and a role nothing filled
#: drops out so its neighbours join directly. That is what lets guardrails sit
#: *between* the framework and the model — which is what guardrails does,
#: inspecting what goes in and what comes back — without costing the stacks
#: that have none their `framework --> llm` edge.
FLOW: Final[tuple[tuple[str, ...], ...]] = (
    ("client", "framework", "guardrails", "llm"),
    ("framework", "vector_db"),
    ("framework", "database"),
    ("framework", "cache"),
    ("framework", "orchestration"),
)

#: Supporting role, and the node it hangs off. Compute anchors to the model
#: rather than to the framework: it is where the weights run, and hanging it
#: off the orchestration glue would put it in the wrong place for the one
#: reader who asked for it.
SUPPORTING: Final[tuple[tuple[str, str], ...]] = (
    ("observability", "framework"),
    ("deployment", "framework"),
    ("compute", "llm"),
)


def _label(text: str) -> str:
    return str(text).replace('"', "").replace("[", "(").replace("]", ")").replace("\n", " ")


def mermaid(components: list[ToolOut], requirements: Requirements) -> str:
    by_category = {tool.category: tool for tool in components}
    present: dict[str, ToolOut] = {}
    for role in ROLES:
        tool = by_category.get(_role_category(role, requirements))
        if tool is not None:
            present[role.key] = tool

    lines = ["graph LR", '    client["Your application"]']

    for role in ROLES:
        tool = present.get(role.key)
        if tool is None:
            continue
        lines.append(f'    {role.key}["{_label(role.label)}<br/>{_label(tool.name)}"]')

    for chain in FLOW:
        drawn = [key for key in chain if key == "client" or key in present]
        for source, target in pairwise(drawn):
            lines.append(f"    {source} --> {target}")

    # These wrap the stack rather than sit in the request path; a dotted edge
    # says that without implying a call.
    for role_key, anchor in SUPPORTING:
        if role_key in present and anchor in present:
            lines.append(f"    {anchor} -.-> {role_key}")

    # Colour by role, and a brand mark per box. `client` is the caller rather
    # than a component, so it has a role and no tool — which is exactly the
    # case `decorate` leaves without a mark.
    roles = {"client": "client", **{key: key for key in present}}
    tools: dict[str, str | None] = {"client": None}
    tools.update({key: tool.slug for key, tool in present.items()})

    return "\n".join(diagram_theme.decorate(lines, roles=roles, tools=tools))


def document(
    *,
    components: list[ToolOut],
    requirements: Requirements,
    summary: str,
    diagram: str,
    score_rows: list[dict[str, object]],
    component_rows: list[dict[str, object]],
    roadmap: list[dict[str, object]],
) -> str:
    """The exportable architecture document.

    Everything on the result page, in one file someone can paste into a
    repository. Generated from the same rows the screen renders, so the
    document and the screen cannot disagree.
    """
    components_table = "\n".join(
        f"| {row['role']} | {row['name']} | {row['status']} | {row['why']} |"
        for row in component_rows
    )
    score_table = "\n".join(
        f"| {row['dimension']} | {row['score']}/10 | {row['weight_pct']}% | {row['contribution']} |"
        for row in score_rows
    )
    roadmap_section = (
        "\n".join(
            f"{index}. **{step.get('title', '')}** — {step.get('detail', '')} "
            f"_({step.get('effort', 'effort not estimated')})_"
            for index, step in enumerate(roadmap, 1)
        )
        or "_Roadmap unavailable for this run._"
    )

    return f"""# Stack architecture

{summary}

## Requirements

| Requirement | Value |
| --- | --- |
| Use case | {requirements.use_case} |
| Scale target | {requirements.scale_target} |
| Monthly budget | ${requirements.monthly_budget:,} |
| Team experience | {requirements.team_skill} |
| Latency target | {requirements.latency_ms} ms |
| Data sensitivity | {requirements.sensitivity} |
| Deployment | {requirements.deployment} |

## Diagram

```mermaid
{diagram}
```

## Components

| Role | Choice | Status | Why |
| --- | --- | --- | --- |
{components_table}

## Stack Score

| Dimension | Score | Weight | Contribution |
| --- | --- | --- | --- |
{score_table}

The contributions sum to the headline score. Every dimension is computed from
the catalog at read time, so this score changes when the catalog does — a
component deprecated tomorrow lowers this stack's score tomorrow.

## Implementation roadmap

{roadmap_section}

---

Generated by StackForge. Components were selected by hard constraint and
weighted scoring against the requirements above; nothing here was chosen by a
model.
"""
