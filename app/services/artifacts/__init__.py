"""Artifact generation (M18).

One registry, one entry per artifact type. A generator takes a resolved source
and returns an `Artifact` — the same seven-key shape the tools already emit, so
the export layer, the artifact tray, and the share page all handle a generated
artifact and a tool-emitted one through the same path.

**Generators are pure and deterministic.** Same source, byte-identical output.
FR-11 requires idempotent exports; making that a property of the functions
rather than a promise about them turns it into a one-line unit test — which is
the only reason to believe it.

Two kinds of artifact reach a user, and the difference matters:

  * **Emitted** — the tool produced it during the run and it is stored on the
    row. These are returned verbatim. Regenerating a `docker-compose.yml` from
    the run's inputs would risk handing someone a file that differs from the
    one they were shown, and the difference would be invisible until it ran.
  * **Generated** — composed here from a stack or a run. These exist because
    the thing they describe has no natural moment of creation: nobody runs the
    "architecture document" tool.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

from app.core.errors import NotFound
from app.schemas.tools import Artifact
from app.services.artifacts import (
    architecture,
    business_case,
    cost_report,
    cursor_rules,
    deployment,
    roadmap,
)
from app.services.artifacts.sources import RunSource, Source, StackSource


class Generator(NamedTuple):
    type: str
    label: str
    description: str
    #: Declared rather than read back off a built artifact. Listing what a
    #: source *can* produce happens on every result page; building all eight
    #: documents to discover their filenames would make opening a page cost
    #: what exporting everything costs.
    filename: str
    format: str
    supports: Callable[[Source], bool]
    build: Callable[[Source], Artifact]


def _stack_only(builder: Callable[[StackSource], Artifact]) -> Callable[[Source], Artifact]:
    """Narrow a stack generator to the registry's uniform signature.

    The `supports` predicate has already established the type; this is what
    lets the registry hold one callable type without every generator taking a
    union it then has to re-narrow.
    """

    def build(source: Source) -> Artifact:
        assert isinstance(source, StackSource)
        return builder(source)

    return build


def _run_only(builder: Callable[[RunSource], Artifact]) -> Callable[[Source], Artifact]:
    def build(source: Source) -> Artifact:
        assert isinstance(source, RunSource)
        return builder(source)

    return build


#: Display order. This is the order the artifact tray shows and the order the
#: bundle README lists, so the most useful thing is first rather than whichever
#: generator was written first.
GENERATORS: tuple[Generator, ...] = (
    Generator(
        architecture.TYPE_DOCUMENT,
        "Architecture document",
        "The whole plan in one Markdown file: components, scores, compatibility, and the roadmap.",
        "architecture.md",
        "markdown",
        architecture.supports,
        _stack_only(architecture.document),
    ),
    Generator(
        architecture.TYPE_DIAGRAM,
        "Architecture diagram",
        "A Mermaid diagram of the request path through the stack.",
        "architecture.mmd",
        "mermaid",
        architecture.supports,
        _stack_only(architecture.diagram),
    ),
    Generator(
        cost_report.TYPE,
        "Cost estimate",
        "Cost shape against the stated budget, or the figures a cost tool produced.",
        "cost-estimate.md",
        "markdown",
        cost_report.supports,
        cost_report.generate,
    ),
    Generator(
        roadmap.TYPE,
        "Implementation roadmap",
        "Phases in dependency order, derived from the roles this stack fills.",
        "roadmap.md",
        "markdown",
        roadmap.supports,
        _stack_only(roadmap.generate),
    ),
    Generator(
        deployment.TYPE_COMPOSE,
        "Docker Compose",
        "A starter compose file built from this stack's own components.",
        "docker-compose.yml",
        "yaml",
        deployment.supports,
        _stack_only(deployment.compose),
    ),
    Generator(
        deployment.TYPE_ENV,
        ".env.example",
        "Every variable the compose file and the chosen SDKs read.",
        ".env.example",
        "text",
        deployment.supports,
        _stack_only(deployment.env_example),
    ),
    Generator(
        cursor_rules.TYPE,
        "Cursor rules",
        "A .cursorrules file pinning the stack and its gotchas for an assistant.",
        ".cursorrules",
        "text",
        cursor_rules.supports,
        _stack_only(cursor_rules.generate),
    ),
    Generator(
        business_case.TYPE,
        "Business case",
        "Payback, ROI, and the assumptions behind them.",
        "business-case.md",
        "markdown",
        business_case.supports,
        _run_only(business_case.generate),
    ),
)

BY_TYPE: dict[str, Generator] = {generator.type: generator for generator in GENERATORS}


class Descriptor(NamedTuple):
    """What the tray needs to offer an artifact before generating it."""

    type: str
    label: str
    description: str
    filename: str
    format: str
    #: True when the tool produced it during the run rather than composing it
    #: here. Surfaced so the UI can say "from this run" — a user who exported a
    #: compose file expects the one they saw.
    emitted: bool


def available(source: Source) -> list[Descriptor]:
    """Every artifact this source can produce, in display order.

    Emitted artifacts come first: they are the concrete output of the thing the
    user just ran, and burying them under generated documents would put the
    thing they came for second.
    """
    descriptors: list[Descriptor] = []
    seen: set[str] = set()

    if isinstance(source, RunSource):
        for artifact in source.artifacts:
            if artifact.type in seen:
                continue
            seen.add(artifact.type)
            descriptors.append(
                Descriptor(
                    type=artifact.type,
                    label=_label_for(artifact.type),
                    description=f"Produced by {source.tool_slug} during this run.",
                    filename=artifact.filename,
                    format=artifact.format,
                    emitted=True,
                )
            )

    for generator in GENERATORS:
        if generator.type in seen or not generator.supports(source):
            continue
        seen.add(generator.type)
        descriptors.append(
            Descriptor(
                type=generator.type,
                label=generator.label,
                description=generator.description,
                filename=generator.filename,
                format=generator.format,
                emitted=False,
            )
        )

    return descriptors


def _label_for(artifact_type: str) -> str:
    generator = BY_TYPE.get(artifact_type)
    if generator is not None:
        return generator.label
    return artifact_type.replace("-", " ").replace("_", " ").capitalize()


def generate(source: Source, artifact_type: str) -> Artifact:
    """One artifact by type. Emitted beats generated.

    The precedence is deliberate: if a run emitted a `compose` artifact and a
    generator could also build one, the user gets the file they were shown.
    """
    if isinstance(source, RunSource):
        for artifact in source.artifacts:
            if artifact.type == artifact_type:
                return artifact

    generator = BY_TYPE.get(artifact_type)
    if generator is None or not generator.supports(source):
        raise NotFound(f"No artifact of type '{artifact_type}' for this result.")
    return generator.build(source)


def generate_all(source: Source) -> list[Artifact]:
    """Everything the source can produce, in the same order `available` lists."""
    return [generate(source, descriptor.type) for descriptor in available(source)]


__all__ = [
    "BY_TYPE",
    "GENERATORS",
    "Descriptor",
    "Generator",
    "RunSource",
    "Source",
    "StackSource",
    "available",
    "generate",
    "generate_all",
]
