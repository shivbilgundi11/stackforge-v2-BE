"""Colour and brand marks for every generated diagram.

Three services draw Mermaid — the stack architect, the RAG pipeline, and the
agent planner — and until now each emitted bare `graph LR` with grey boxes.
A stack diagram is the artefact people paste into a design review, and an
undifferentiated column of grey rectangles makes the reader do the sorting the
picture was supposed to do for them.

Two things are added here, and they are deliberately separate:

**Colour, in the source.** A `classDef` block plus one `class` line per node.
This is ordinary Mermaid, so the `.mmd` artefact someone downloads renders in
colour on GitHub, in Notion, in the VS Code preview — anywhere, with no help
from us. It carries meaning rather than decoration: the colour is the *role*,
so the model is always violet and the stores are always blue no matter which
tool filled the slot.

**Brand marks, as metadata.** A `%% brand:` comment per node naming the icon
and its colour. Comments are ignored by every Mermaid renderer, so the source
stays portable and the artefact stays readable; a renderer that knows about
them draws the logo, and one that does not loses nothing. That is why the logo
is not a node image: an `img` shape would need a data URI in the source, and
five kilobytes of base64 in the middle of a diagram someone is meant to read
and edit is not a trade worth making.

The mapping from a catalog slug to an icon is in `app.data.brands`. It is
partial on purpose — see the note there.
"""

from __future__ import annotations

from typing import Final

from app.data.brands import brand_for

#: Role key to the class name used in the diagram, and the colour that class
#: paints. One hue per role, chosen to stay legible on both a white page and
#: the dark app surface: these are mid-tone strokes, never a light fill that
#: would vanish on one of the two.
#:
#: The RAG pipeline and the agent DAG have their own vocabularies rather than
#: the stack's roles, and both are here: one table, so a colour means the same
#: thing wherever a reader meets it.
ROLE_COLOURS: Final[dict[str, str]] = {
    # The request path.
    "client": "#64748b",  # slate — the caller, deliberately neutral
    "framework": "#0ea5e9",  # sky
    "guardrails": "#f43f5e",  # rose — the one that inspects and can refuse
    "llm": "#8b5cf6",  # violet — generation, the centre of the picture
    # Stores.
    "vector_db": "#2563eb",  # blue
    "database": "#2563eb",
    "cache": "#0d9488",  # teal
    # Supporting.
    "orchestration": "#d97706",  # amber
    "observability": "#65a30d",  # lime
    "deployment": "#7c3aed",  # violet, darker
    "compute": "#c026d3",  # fuchsia — where the weights run
    # RAG pipeline stages.
    "loader": "#64748b",
    "splitter": "#0ea5e9",
    "embedder": "#8b5cf6",
    "store": "#2563eb",
    "retriever": "#0d9488",
    "reranker": "#d97706",
    "generator": "#8b5cf6",
    # Agent DAG. The vocabulary is `agent_planner_service`'s, not the stack's:
    # a topology is made of the parts that decide, the parts that do, and the
    # parts that check, and the colour says which is which before the label is
    # read.
    "planner": "#d97706",  # amber — decides
    "dispatcher": "#d97706",
    "supervisor": "#d97706",
    "triage": "#d97706",
    "worker": "#0ea5e9",  # sky — does
    "specialist": "#0ea5e9",
    "lead": "#0ea5e9",
    "reviewer": "#65a30d",  # lime — checks
    "resolver": "#65a30d",
    "aggregator": "#8b5cf6",  # violet — collects the answer
}

#: Every node that has no role of its own. Not an error case — the agent
#: planner names its nodes after the work they do, and most of those names are
#: not in the vocabulary above.
DEFAULT_COLOUR: Final = "#64748b"

#: The class prefix. Prefixed rather than bare so a role called `default` or
#: `node` cannot collide with something Mermaid already means.
CLASS_PREFIX: Final = "sf-"


def class_name(role: str) -> str:
    return f"{CLASS_PREFIX}{role.replace('_', '-')}"


def classdef_block(roles: list[str]) -> list[str]:
    """One `classDef` per role actually used.

    Only the roles present, rather than the whole table: a diagram carrying
    twenty definitions for four boxes is noise in a file someone reads.

    The fill is left to the renderer's theme and only the stroke is set. A
    fixed fill is the one thing that cannot work in both places — anything
    light enough to read black text on disappears against the dark app
    surface, and anything dark enough for the app is a hole in a printed page.
    The stroke carries the colour in both.
    """
    lines: list[str] = []
    for role in dict.fromkeys(roles):
        colour = ROLE_COLOURS.get(role, DEFAULT_COLOUR)
        lines.append(f"    classDef {class_name(role)} stroke:{colour},stroke-width:2px")
    return lines


def class_line(node_id: str, role: str) -> str:
    return f"    class {node_id} {class_name(role)}"


def brand_comment(node_id: str, tool_slug: str | None, role: str) -> str | None:
    """`%% brand:<node>:<icon>:<hex>`, or nothing.

    The hex is always present even when the icon is not: a renderer with no
    logo for this tool still draws a coloured monogram, and it should be the
    role's colour rather than a seventh grey circle.
    """
    if tool_slug is None:
        return None

    mark = brand_for(tool_slug)
    icon = mark.icon if mark else ""
    colour = mark.hex if mark else ROLE_COLOURS.get(role, DEFAULT_COLOUR).lstrip("#")
    return f"%% brand:{node_id}:{icon}:{colour}"


def decorate(lines: list[str], *, roles: dict[str, str], tools: dict[str, str | None]) -> list[str]:
    """Append the colour and brand metadata to a finished diagram.

    `roles` maps node id to role key; `tools` maps node id to the catalog slug
    that filled it, or None where nothing did. Both are keyed on the node id
    the caller already generated, so this never has to know how a diagram is
    laid out — only what each box turned out to be.
    """
    out = list(lines)
    out.extend(classdef_block([roles[node] for node in roles]))
    out.extend(class_line(node, role) for node, role in roles.items())

    comments = [
        comment
        for node, role in roles.items()
        if (comment := brand_comment(node, tools.get(node), role)) is not None
    ]
    if comments:
        # After the diagram body, not before it. A reader opening the `.mmd`
        # should meet the picture first; the metadata is for the renderer.
        out.append("")
        out.extend(f"    {comment}" for comment in comments)
    return out
