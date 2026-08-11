"""Rendering, gating, and the export lifecycle (M18).

`PRD.md` §18 names artifact export as the **primary Pro conversion trigger**:
free users get real answers, they pay to act on them. So this module is both a
feature and the revenue mechanism, and the two pull in opposite directions.
The resolution is spelled out in three rules:

**Markdown is free and complete.** Not a teaser, not watermarked, not a
subset — the full answer in a form the user can paste anywhere. A crippled free
export teaches people the product is stingy rather than that the paid tier is
worth having.

**The paid formats are the ones you hand to someone else.** PDF, the zip
bundle, JSON and YAML for a pipeline, CSV for a spreadsheet. Those are the
formats where the work leaves the person who did it.

**Locked formats stay visible.** The gate is enforced here, server-side, and
surfaced with a plan badge rather than hidden. A user who never learns PDF
export exists never upgrades for it.

Everything is generated from `services/artifacts/`, which is pure and
deterministic, so re-exporting the same thing produces byte-identical output.
That is FR-11, and it is a unit test rather than a hope.
"""

from __future__ import annotations

import csv
import io
import json
import zipfile
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Final, cast

import yaml
from sqlalchemy import delete, select
from sqlalchemy.engine import CursorResult
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import Identity
from app.core.config import settings
from app.core.database import new_id, utcnow
from app.core.errors import NotFound, PlanRequired, ValidationFailed
from app.core.logging import get_logger
from app.models.export import Export, ExportFormat, ExportStatus, SourceType
from app.models.user import Plan
from app.schemas.tools import Artifact
from app.services import artifacts, pdf_service
from app.services.artifacts import markdown as md
from app.services.artifacts import result_document
from app.services.artifacts.sources import RunSource, Source, StackSource

logger = get_logger("exports")

#: Which plan each format needs. `PRD.md` §18. A dict rather than an `if`,
#: because "what does Pro get" is a pricing question and pricing questions
#: that live in branches get out of step with the pricing page.
FORMAT_PLANS: Final[dict[ExportFormat, Plan]] = {
    ExportFormat.MARKDOWN: Plan.FREE,
    ExportFormat.JSON: Plan.PRO,
    ExportFormat.YAML: Plan.PRO,
    ExportFormat.CSV: Plan.PRO,
    ExportFormat.PDF: Plan.PRO,
    ExportFormat.ZIP: Plan.PRO,
}

PLAN_RANK: Final[dict[Plan, int]] = {
    Plan.FREE: 0,
    Plan.PRO: 1,
    Plan.TEAM: 2,
    Plan.ENTERPRISE: 3,
}

CONTENT_TYPES: Final[dict[ExportFormat, str]] = {
    ExportFormat.MARKDOWN: "text/markdown; charset=utf-8",
    ExportFormat.JSON: "application/json",
    ExportFormat.YAML: "application/yaml",
    ExportFormat.CSV: "text/csv; charset=utf-8",
    ExportFormat.PDF: "application/pdf",
    ExportFormat.ZIP: "application/zip",
}

EXTENSIONS: Final[dict[ExportFormat, str]] = {
    ExportFormat.MARKDOWN: "md",
    ExportFormat.JSON: "json",
    ExportFormat.YAML: "yaml",
    ExportFormat.CSV: "csv",
    ExportFormat.PDF: "pdf",
    ExportFormat.ZIP: "zip",
}

#: Bytes of generated prose per stack component, used to predict a bundle's
#: size before building it. Deliberately generous — over-predicting queues a
#: job that did not need queueing, under-predicting builds a large zip inside
#: a request, and only one of those is a timeout.
BUNDLE_BYTES_PER_COMPONENT: Final = 3_500
BUNDLE_FIXED_BYTES: Final = 24_000


@dataclass(frozen=True)
class Rendered:
    filename: str
    content_type: str
    data: bytes


# ── plan gating ──────────────────────────────────────────────────────────────


def required_plan(export_format: ExportFormat) -> Plan:
    return FORMAT_PLANS.get(export_format, Plan.PRO)


def allowed(export_format: ExportFormat, identity: Identity) -> bool:
    return PLAN_RANK[identity.plan] >= PLAN_RANK[required_plan(export_format)]


def assert_allowed(export_format: ExportFormat, identity: Identity) -> None:
    """Raise the 402 the upgrade dialog is built against.

    `required_plan` and `current_plan` are in the details because a paywall
    with no figures is a dead end — the dialog has to be able to say what the
    user would be buying.
    """
    if allowed(export_format, identity):
        return
    minimum = required_plan(export_format)
    raise PlanRequired(
        f"{export_format.value.upper()} export requires the {minimum.value.title()} plan.",
        details={
            "required_plan": minimum.value,
            "current_plan": identity.plan.value,
            "format": export_format.value,
        },
    )


# ── rendering ────────────────────────────────────────────────────────────────


def _slug(text: str) -> str:
    cleaned = "".join(character if character.isalnum() else "-" for character in text.lower())
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    return cleaned.strip("-") or "export"


def filename_for(
    source: Source,
    *,
    artifact_type: str | None,
    export_format: ExportFormat,
) -> str:
    base = _slug(source.slug_basis)
    if export_format is ExportFormat.ZIP:
        return f"{base}-plan.zip"
    if artifact_type is not None:
        return f"{base}-{_slug(artifact_type)}.{EXTENSIONS[export_format]}"
    return f"{base}.{EXTENSIONS[export_format]}"


def render(
    source: Source,
    *,
    artifact_type: str | None = None,
    export_format: ExportFormat,
    table: str | None = None,
    share_url: str | None = None,
) -> Rendered:
    """One source, one format, one byte string. Deterministic."""
    if export_format is ExportFormat.ZIP:
        data = bundle(source)
    elif export_format is ExportFormat.CSV:
        data = _csv(source, table=table)
    elif export_format is ExportFormat.PDF:
        data = _pdf(source, artifact_type=artifact_type, share_url=share_url)
    elif export_format is ExportFormat.MARKDOWN:
        data = _markdown(source, artifact_type=artifact_type).encode("utf-8")
    else:
        payload = _payload(source, artifact_type=artifact_type)
        if export_format is ExportFormat.JSON:
            data = json.dumps(payload, indent=2, sort_keys=False, default=str).encode("utf-8")
        else:
            # Round-tripped through the same payload the JSON export uses, so
            # the two formats cannot describe different things.
            data = yaml.safe_dump(
                json.loads(json.dumps(payload, default=str)),
                sort_keys=False,
                default_flow_style=False,
                width=100,
                allow_unicode=True,
            ).encode("utf-8")

    return Rendered(
        filename=filename_for(source, artifact_type=artifact_type, export_format=export_format),
        content_type=CONTENT_TYPES[export_format],
        data=data,
    )


def _markdown(source: Source, *, artifact_type: str | None) -> str:
    if artifact_type is None:
        return result_document.render(source)

    artifact = artifacts.generate(source, artifact_type)
    if artifact.format == "markdown":
        return artifact.content

    # A YAML or text artifact is not Markdown, so it is wrapped rather than
    # emitted raw under a `.md` name. The heading gives the file a title, and
    # the fence keeps the content copy-pasteable.
    label = (
        artifacts.BY_TYPE[artifact_type].label
        if artifact_type in artifacts.BY_TYPE
        else (md.humanise(artifact_type))
    )
    return (
        f"# {label}\n\n"
        f"`{artifact.filename}` — generated by StackForge from {source.title}.\n\n"
        + md.fence(artifact.content, language=artifact.language or artifact.format)
        + "\n"
    )


def _payload(source: Source, *, artifact_type: str | None) -> dict[str, Any]:
    """The envelope every structured export carries.

    Versioned and self-describing: a JSON file found in a repository six
    months from now has to say what produced it and against which contract,
    or it is an anonymous blob someone has to reverse-engineer.

    There is no `generated_at`. A wall-clock stamp would make the second
    export of an unchanged stack differ from the first, which is exactly what
    FR-11 forbids. `source_updated_at` answers the question a reader actually
    has — how old is the thing this describes — and is stable.
    """
    envelope: dict[str, Any] = {
        "stackforge": {
            "schema": "stackforge.export/v1",
            "source_type": source.kind.value,
            "source_id": source.id,
            "source_updated_at": source.updated_at.isoformat(),
            "title": source.title,
        }
    }

    if artifact_type is not None:
        artifact = artifacts.generate(source, artifact_type)
        envelope["artifact"] = artifact.model_dump(mode="json")
        return envelope

    if isinstance(source, StackSource):
        envelope["stack"] = {
            "id": source.id,
            "name": source.title,
            "description": source.description,
            "version": source.version,
            "score": str(source.score.total),
            "score_breakdown": source.score.breakdown(),
            "requirements": source.requirements._asdict(),
            "components": [tool.model_dump(mode="json") for tool in source.components],
            "missing_components": source.missing_slugs,
            "deprecated_components": [tool.slug for tool in source.deprecated],
            "compatibility": (
                source.compatibility.model_dump(mode="json") if source.compatibility else None
            ),
        }
    else:
        envelope["result"] = source.output.model_dump(mode="json")
        envelope["input"] = source.input
        envelope["workflow"] = source.workflow

    return envelope


def tables_of(source: Source) -> dict[str, list[dict[str, Any]]]:
    """Everything in the source that is genuinely tabular.

    A stack has no `tables` key, so its rows are assembled here from the two
    things that are actually rectangular — the components and the score
    breakdown. Exporting a CSV of "the stack" without saying which of those it
    is would produce a file whose columns depend on an implementation detail.
    """
    if isinstance(source, StackSource):
        from app.services.artifacts.architecture import _component_rows

        rows: dict[str, list[dict[str, Any]]] = {
            "components": _component_rows(source),
            "score_breakdown": source.score.breakdown(),
        }
        if source.compatibility is not None and source.compatibility.pairs:
            rows["compatibility"] = [
                {
                    "pair": f"{pair.tool_a} + {pair.tool_b}",
                    "score": pair.score,
                    "notes": pair.notes or "",
                }
                for pair in source.compatibility.pairs
            ]
        return rows

    return {name: rows for name, rows in source.output.tables.items() if rows}


def _csv(source: Source, *, table: str | None) -> bytes:
    """One table per CSV file, named explicitly.

    Concatenating every table into one file with separator rows produces
    something no spreadsheet reads correctly, and picking one silently
    produces a file whose contents depend on dict ordering. So: if there is
    exactly one table it is used, and otherwise the caller names it.
    """
    available = tables_of(source)
    if not available:
        raise ValidationFailed.on_field(
            "format", "This result has no tabular data, so there is nothing to export as CSV."
        )

    if table is None:
        if len(available) > 1:
            raise ValidationFailed.on_field(
                "table",
                "Name the table to export: " + ", ".join(sorted(available)) + ".",
            )
        table = next(iter(available))

    rows = available.get(table)
    if not rows:
        raise ValidationFailed.on_field(
            "table",
            f"No table named '{table}'. Available: " + ", ".join(sorted(available)) + ".",
        )

    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)

    buffer = io.StringIO(newline="")
    # `\r\n` per RFC 4180, and written explicitly rather than left to the
    # platform — an export whose line endings depend on which machine rendered
    # it is not byte-identical across re-exports.
    writer = csv.DictWriter(
        buffer, fieldnames=columns, lineterminator="\r\n", extrasaction="ignore"
    )
    writer.writeheader()
    for row in rows:
        writer.writerow({key: md.scalar(row.get(key)) for key in columns})
    return buffer.getvalue().encode("utf-8")


def _pdf(source: Source, *, artifact_type: str | None, share_url: str | None) -> bytes:
    title, subtitle = result_document.title_of(source)
    if artifact_type is None:
        body = result_document.render(source)
    else:
        body = _markdown(source, artifact_type=artifact_type)
        label = (
            artifacts.BY_TYPE[artifact_type].label
            if artifact_type in artifacts.BY_TYPE
            else md.humanise(artifact_type)
        )
        subtitle = f"{label} · {subtitle}"

    return pdf_service.render(
        pdf_service.Document(
            title=title,
            subtitle=subtitle,
            markdown=body,
            share_url=share_url or settings.web_base_url,
            # The source's timestamp, not the clock. Same reason as `_payload`:
            # the cover date has to be a property of the plan, or every
            # re-export differs.
            generated_at=source.updated_at,
        )
    )


# ── the bundle ───────────────────────────────────────────────────────────────


def bundle_layout(source: Source) -> list[tuple[str, str]]:
    """`(path, content)` for every file in the zip, in write order.

    Computed separately from the zip so a test can assert the file list
    without unzipping, and so the README can list the real contents rather
    than a hardcoded description that drifts from them.
    """
    root = f"{_slug(source.slug_basis)}-plan"
    readme_path = f"{root}/README.md"
    files: list[tuple[str, str]] = []

    for artifact in artifacts.generate_all(source):
        files.append((f"{root}/{_bundle_path(artifact)}", artifact.content))

    if isinstance(source, RunSource):
        # The full result, so the bundle is self-contained even for a run whose
        # generators produced nothing but the emitted files.
        files.append((f"{root}/result.md", result_document.render(source)))

    # The README lists itself. A contents list that omits one of the files is
    # a contents list a reader stops trusting, and "except this one" is not a
    # rule anyone can hold in their head while scanning a zip.
    paths = [readme_path, *(path for path, _ in files)]
    files.insert(0, (readme_path, _bundle_readme(source, paths, root)))
    return files


#: Where each artifact type lands in the bundle. Deployment files go under
#: `deploy/` so the top level of the zip is documents a person reads and the
#: subdirectory is files a machine runs.
BUNDLE_PATHS: Final[dict[str, str]] = {
    "compose": "deploy/docker-compose.yml",
    "env": "deploy/.env.example",
    "k8s-deployment": "deploy/k8s/deployment.yaml",
    "k8s-service": "deploy/k8s/service.yaml",
    "k8s-hpa": "deploy/k8s/hpa.yaml",
    "k8s-pdb": "deploy/k8s/pdb.yaml",
}


def _bundle_path(artifact: Artifact) -> str:
    if artifact.type in BUNDLE_PATHS:
        return BUNDLE_PATHS[artifact.type]
    # The MCP generator emits a whole server tree, with paths already in the
    # filenames. Preserved rather than flattened — a bundle whose server files
    # all land in one directory is a bundle that does not run.
    if artifact.type.startswith("mcp") or "/" in artifact.filename:
        return f"mcp/{artifact.filename}" if "/" in artifact.filename else artifact.filename
    return artifact.filename


def _bundle_readme(source: Source, paths: list[str], root: str) -> str:
    listed = "\n".join(f"- `{path.removeprefix(root + '/')}`" for path in paths)
    title, subtitle = result_document.title_of(source)

    warning = ""
    if isinstance(source, StackSource) and source.deprecated:
        names = ", ".join(tool.name for tool in source.deprecated)
        warning = (
            f"\n> **{len(source.deprecated)} component(s) in this plan are marked "
            f"deprecated or caution in the catalog: {names}. "
            f"See `architecture.md` for the reason recorded against each.**\n"
        )

    return f"""# {title}

{subtitle}. Generated by StackForge from the state of
{source.updated_at.strftime("%d %B %Y")} — the date this plan last changed,
not the date you downloaded it.
{warning}
## What is in here

{listed}

## How to use it

1. Read `architecture.md` first — it is the whole plan, and every other file in
   this bundle is a piece of it.
2. `roadmap.md` is the order to build it in. The phases are derived from the
   roles this stack fills, so there is nothing in it to prune.
3. Everything under `deploy/` is a **starter template**, not a production
   deployment. Secrets come from `.env`, volumes are local, and the resource
   limits are a starting point rather than a measurement. Read them before you
   run them.
4. `.cursorrules` goes at the root of your repository.

## What this bundle is not

It is not a running system, and it is not an estimate of your bill. The figures
are computed from hand-verified catalog prices against the requirements stated
in `architecture.md`; change a requirement and they change with it.
"""


def bundle(source: Source) -> bytes:
    """A deterministic zip.

    Every entry gets a fixed timestamp and the archive is written store-only.
    Both matter for the same reason: `date_time=now` or a compression level
    that varies by zlib build makes the same input produce different bytes,
    and FR-11 requires that re-exporting produces byte-identical output.
    """
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path, content in bundle_layout(source):
            info = zipfile.ZipInfo(path, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, content.encode("utf-8"))
    return buffer.getvalue()


def predicted_bytes(source: Source, export_format: ExportFormat) -> int:
    """A size estimate that does not require building the thing.

    The threshold for queueing is size, not format (M18), so it has to be
    knowable before the work happens — otherwise "decide whether this is too
    big to do in a request" is itself done in the request.
    """
    if export_format is not ExportFormat.ZIP:
        return 0
    if isinstance(source, StackSource):
        return BUNDLE_FIXED_BYTES + BUNDLE_BYTES_PER_COMPONENT * len(source.components)
    emitted = sum(len(artifact.content) for artifact in source.artifacts)
    return BUNDLE_FIXED_BYTES + emitted


def should_queue(source: Source, export_format: ExportFormat) -> bool:
    return predicted_bytes(source, export_format) > settings.export_async_threshold_bytes


# ── the lifecycle ────────────────────────────────────────────────────────────


async def create(
    db: AsyncSession,
    identity: Identity,
    *,
    source: Source,
    source_type: SourceType,
    export_format: ExportFormat,
    artifact_type: str | None = None,
    table: str | None = None,
) -> Export:
    """Gate, then either render inline or queue.

    Both paths return the same row in the same shape, so the client polls and
    downloads identically whichever happened. A client that has to branch on
    "was this one fast?" grows two code paths, and the rare one rots.
    """
    assert_allowed(export_format, identity)

    now = utcnow()
    export = Export(
        id=new_id("exp"),
        user_id=identity.user.id if identity.user else None,
        anonymous_session_id=None if identity.user else identity.anonymous_id,
        source_type=source_type,
        source_id=source.id,
        artifact_type=artifact_type,
        format=export_format,
        status=ExportStatus.PENDING,
        filename=filename_for(source, artifact_type=artifact_type, export_format=export_format),
        content_type=CONTENT_TYPES[export_format],
        size_bytes=0,
        expires_at=now + timedelta(days=settings.export_ttl_days),
        created_at=now,
    )
    db.add(export)
    await db.flush()

    if should_queue(source, export_format):
        from app.workers.queue import enqueue

        queued = await enqueue("build_export", export.id)
        if queued:
            logger.info(
                "exports.queued",
                export_id=export.id,
                format=export_format.value,
                predicted_bytes=predicted_bytes(source, export_format),
            )
            return export
        # The queue is unreachable. Building inline is slower than queueing and
        # far better than a pending row nothing will ever pick up.
        logger.warning("exports.queue_unavailable", export_id=export.id, fallback="inline")

    rendered = render(
        source,
        artifact_type=artifact_type,
        export_format=export_format,
        table=table,
    )
    complete(export, rendered)
    await db.flush()

    logger.info(
        "exports.created",
        export_id=export.id,
        format=export_format.value,
        artifact_type=artifact_type,
        size_bytes=export.size_bytes,
        authenticated=identity.is_authenticated,
    )
    return export


def complete(export: Export, rendered: Rendered) -> None:
    export.filename = rendered.filename
    export.content_type = rendered.content_type
    export.content = rendered.data
    export.size_bytes = len(rendered.data)
    export.status = ExportStatus.READY
    export.completed_at = utcnow()
    export.error = None


def fail(export: Export, message: str) -> None:
    export.status = ExportStatus.FAILED
    export.error = message
    export.completed_at = utcnow()


async def get(db: AsyncSession, export_id: str, identity: Identity) -> Export:
    """Owner-scoped, and 404 for someone else's — an export id is not a
    capability. That is what share links are for."""
    export = await db.get(Export, export_id)
    if export is None:
        raise NotFound("No export with that id.")

    owned = (identity.user is not None and export.user_id == identity.user.id) or (
        identity.anonymous_id is not None and export.anonymous_session_id == identity.anonymous_id
    )
    if not owned:
        raise NotFound("No export with that id.")

    if export.expires_at <= utcnow():
        raise NotFound("This export has expired. Generate it again.")
    return export


async def list_for(db: AsyncSession, identity: Identity, *, limit: int = 25) -> list[Export]:
    statement = (
        select(Export)
        .where(Export.expires_at > utcnow())
        .order_by(Export.created_at.desc())
        .limit(limit)
    )
    if identity.user is not None:
        statement = statement.where(Export.user_id == identity.user.id)
    elif identity.anonymous_id is not None:
        statement = statement.where(Export.anonymous_session_id == identity.anonymous_id)
    else:
        return []
    return list((await db.execute(statement)).scalars().all())


async def purge_expired(db: AsyncSession) -> int:
    """Delete expired rows and, with them, the bytes.

    One statement because the bytes live in the row (see `models/export.py`).
    Storage cleanup and row cleanup being the same operation is the whole
    reason for that choice — there is no window in which one has happened and
    the other has not.
    """
    result = await db.execute(delete(Export).where(Export.expires_at <= utcnow()))
    removed = int(cast("CursorResult[Any]", result).rowcount or 0)
    if removed:
        logger.info("exports.purged", count=removed)
    return removed
