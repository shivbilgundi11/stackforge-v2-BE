"""The business case.

WF5 already emits one for `model-roi`, because that tool computes payback and
NPV and the document is the whole point of running it. When the source run
carries that artifact this returns it **unchanged** — regenerating a document
the tool already wrote would produce two versions of the same numbers, and the
second one would be the one that drifts.

For the other three ROI tools it composes a case from what they did produce.
Those runs have real figures and no document, and "the ROI tool that does not
export a business case" is exactly the gap that makes an export feature feel
partial.
"""

from __future__ import annotations

from typing import Any

from app.schemas.tools import Artifact
from app.services.artifacts.sources import RunSource, Source, StackSource

TYPE = "business-case"

WORKFLOW = "roi"


def supports(source: Source) -> bool:
    if isinstance(source, StackSource):
        return False
    return source.workflow == WORKFLOW


def generate(source: RunSource) -> Artifact:
    existing = next(
        (artifact for artifact in source.artifacts if artifact.type == TYPE),
        None,
    )
    if existing is not None:
        return existing

    output = source.output
    figures = "\n".join(
        f"| {key.replace('_', ' ').capitalize()} | {value} |"
        for key, value in output.metrics.items()
        if key not in {"summary", "rationale", "confidence"}
    )
    assumptions = _assumptions(output.tables.get("assumptions") or [], source.input)
    caveats = _caveats(source)

    return Artifact(
        type=TYPE,
        format="markdown",
        filename="business-case.md",
        content=f"""# Business case — {source.title}

Prepared from StackForge run `{source.id}` on
{output.created_at.strftime("%Y-%m-%d")}.

## Figures

| Measure | Value |
| --- | --- |
{figures or "| — | no figures recorded |"}

## Stated assumptions

{assumptions}

## What this does not claim

{caveats}

---

Every figure above is arithmetic over the assumptions listed, not a forecast.
Change an assumption and the case changes with it — which is the point of
listing them rather than burying them.
""",
        language="markdown",
    )


def _assumptions(rows: list[dict[str, Any]], inputs: dict[str, Any]) -> str:
    """The tool's own assumptions table if it produced one, else the inputs.

    Falling back to the raw inputs rather than omitting the section: a business
    case whose assumptions are invisible gets challenged in the meeting and
    then discarded, and the inputs *are* the assumptions even when the tool did
    not label them.
    """
    if rows:
        return "\n".join(
            f"- **{row.get('assumption', '')}** — {row.get('value', '')}" for row in rows
        )
    if not inputs:
        return "_No assumptions were recorded for this run._"
    return "\n".join(
        f"- **{key.replace('_', ' ').capitalize()}** — {_scalar(value)}"
        for key, value in inputs.items()
    )


def _scalar(value: object) -> str:
    if isinstance(value, list):
        return ", ".join(str(item) for item in value) or "—"
    return "—" if value is None else str(value)


def _caveats(source: RunSource) -> str:
    lines = [
        "- Adoption is modelled, not observed. Every figure assumes the rollout lands as planned.",
        "- Running cost is charged in full from month one; the saving is not. A "
        "platform is paid for whether or not the rollout has finished.",
        "- Nothing here prices the work of maintaining the system after launch.",
    ]
    lines += [
        f"- {warning.message}"
        for warning in source.output.warnings
        if warning.level in {"warning", "critical"}
    ]
    return "\n".join(lines)
