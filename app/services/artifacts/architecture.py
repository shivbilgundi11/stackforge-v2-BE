"""The architecture document and its diagram.

Both are built from the same resolved stack, so the prose and the picture
cannot disagree — which they did in the old build, where the document was
written from the saved component list and the diagram from the live one.

Nothing here calls a model. The shape of a stack is determined by its roles,
which the engine already knows; asking a model to draw it would add a failure
mode without adding information (D-06).

The document does carry written sections when a model wrote some, but it does
not fetch them: `StackSource.narrative` arrives already filled by the export
layer, which is the only place with a session and an identity to spend. That
keeps the generator a pure function of its source — the same source renders
the same bytes, with or without a narrative — and keeps FR-11's idempotency
requirement a unit test rather than a promise.
"""

from __future__ import annotations

from app.schemas.tools import Artifact
from app.services import stack_diagram_service
from app.services.artifacts.sources import Source, StackSource

TYPE_DOCUMENT = "architecture"
TYPE_DIAGRAM = "diagram"


def supports(source: Source) -> bool:
    return isinstance(source, StackSource)


def diagram(source: StackSource) -> Artifact:
    return Artifact(
        type=TYPE_DIAGRAM,
        format="mermaid",
        filename="architecture.mmd",
        content=stack_diagram_service.mermaid(source.components, source.requirements),
        language="mermaid",
    )


def document(source: StackSource) -> Artifact:
    from app.services.artifacts import cost_report, roadmap

    body = stack_diagram_service.document(
        components=source.components,
        requirements=source.requirements,
        summary=_summary(source),
        diagram=stack_diagram_service.mermaid(source.components, source.requirements),
        score_rows=source.score.breakdown(),
        component_rows=_component_rows(source),
        roadmap=roadmap.steps(source),
    )

    sections = [
        body,
        _narrative_section(source),
        _compatibility_section(source),
        cost_report.shape_section(source),
    ]
    if source.missing_slugs:
        sections.append(_missing_section(source))

    return Artifact(
        type=TYPE_DOCUMENT,
        format="markdown",
        filename="architecture.md",
        content="\n".join(part for part in sections if part),
        language="markdown",
    )


def _narrative_section(source: StackSource) -> str:
    """The written sections, or nothing at all.

    Absent rather than apologetic when no model ran. A heading followed by
    "commentary unavailable" is a hole someone has to explain to whoever they
    sent the document to; a document that simply does not have the section
    reads as complete, which it is — every figure above it is the engine's.
    """
    narrative = source.narrative
    if narrative is None:
        return ""

    parts = [
        ("Overview", narrative.overview),
        ("Design decisions", narrative.decisions),
        ("Operating this stack", narrative.operations),
    ]
    written = [f"## {heading}\n\n{body.strip()}\n" for heading, body in parts if body.strip()]
    return "\n".join(written)


def _summary(source: StackSource) -> str:
    """The document's opening paragraph.

    Written from the saved stack rather than from the Architect run that
    produced it: a stack that has been edited since is no longer described by
    the run's summary, and quoting it anyway would describe components that
    are no longer in the file.
    """
    names = [tool.name for tool in source.components]
    lead = names[0] if names else "no components"
    return (
        f"**{source.title}** — a {source.requirements.use_case} stack for "
        f"{source.requirements.scale_target} scale on a "
        f"${source.requirements.monthly_budget:,}/month budget, scoring "
        f"{source.score.total}/100 against today's catalog. Built around {lead}. "
        f"Version {source.version}."
    )


def _component_rows(source: StackSource) -> list[dict[str, object]]:
    """Role rows for the components table.

    Reuses the Architect's own row builder so the exported table has the same
    columns, the same `why` sentences, and the same graveyard status the
    result page shows.
    """
    from app.services import stack_architect_service

    candidate = stack_architect_service.Candidate(
        rank=1,
        components=source.components,
        score=source.score,
        compatibility=source.compatibility,
        deprecated=source.deprecated,
    )
    return stack_architect_service.component_rows(candidate, source.requirements)


def _compatibility_section(source: StackSource) -> str:
    if source.compatibility is None or not source.compatibility.pairs:
        return (
            "\n## Compatibility\n\n"
            "This stack has fewer than two components with reviewed pairings, so "
            "there is nothing to report. An unreported pairing is not a compatible "
            "one — it is one nobody has checked.\n"
        )

    rows = "\n".join(
        f"| {pair.tool_a} + {pair.tool_b} | {pair.score}/100 | {_status_of(pair.score)} | "
        f"{(pair.notes or '').replace('|', '/')} |"
        for pair in source.compatibility.pairs
    )
    unknown = ""
    if source.compatibility.missing_pairs:
        listed = ", ".join(" + ".join(pair) for pair in source.compatibility.missing_pairs)
        unknown = (
            f"\n{len(source.compatibility.missing_pairs)} pairings have no reviewed "
            f"score and are reported as unknown rather than assumed to work: "
            f"{listed}.\n"
        )

    return f"""
## Compatibility

Overall **{source.compatibility.overall}/100** — the *worst* pairing in the stack,
not the average. A stack is only as compatible as its weakest pair, and averaging
lets four good pairings hide the one combination that does not work.

| Pairing | Score | Status | Notes |
| --- | --- | --- | --- |
{rows}
{unknown}"""


def _status_of(score: int) -> str:
    """Value plus a word, never colour alone (M04)."""
    if score >= 85:
        return "Excellent"
    if score >= 70:
        return "Good"
    if score >= 50:
        return "Caution"
    return "Incompatible"


def _missing_section(source: StackSource) -> str:
    listed = "\n".join(f"- `{slug}`" for slug in source.missing_slugs)
    return f"""
## Components no longer in the catalog

This stack names {len(source.missing_slugs)} component(s) the catalog no longer
carries. They are listed rather than omitted — a plan that quietly drops a
component reads as a plan that never had one.

{listed}
"""
