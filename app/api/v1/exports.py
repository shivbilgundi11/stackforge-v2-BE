"""Export endpoints (M18).

Anonymous callers reach every route here. Running and exporting are free; the
gate is on *format*, not on identity, and it is applied in `export_service`
where the pricing table lives rather than as a route dependency — a
`require_plan(PRO)` on the route would have to be repeated per format and
would gate Markdown along with the rest.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query, Response

from app.api.deps import CallerIdentity, Db
from app.core.errors import NotFound
from app.core.responses import Envelope, ok
from app.models.export import Export, ExportFormat, SourceType
from app.schemas.exports import (
    ArtifactOptionOut,
    ExportIn,
    ExportOptionsOut,
    ExportOut,
    FormatOptionOut,
)
from app.services import artifacts, export_service
from app.services.artifacts import sources

router = APIRouter(tags=["exports"])

FORMAT_LABELS = {
    ExportFormat.MARKDOWN: "Markdown",
    ExportFormat.JSON: "JSON",
    ExportFormat.YAML: "YAML",
    ExportFormat.CSV: "CSV",
    ExportFormat.PDF: "PDF",
    ExportFormat.ZIP: "Bundle (.zip)",
}


def _out(export: Export) -> ExportOut:
    return ExportOut(
        id=export.id,
        source_type=export.source_type.value,
        source_id=export.source_id,
        artifact_type=export.artifact_type,
        format=export.format.value,
        status=export.status.value,
        filename=export.filename,
        content_type=export.content_type,
        size_bytes=export.size_bytes,
        error=export.error,
        expires_at=export.expires_at,
        created_at=export.created_at,
        completed_at=export.completed_at,
        download_url=f"/api/v1/exports/{export.id}/download",
    )


@router.get("/options", response_model=Envelope[ExportOptionsOut], name="get_export_options")
async def get_export_options(
    db: Db,
    identity: CallerIdentity,
    source_type: str = Query(description="run or stack"),
    source_id: str = Query(min_length=1, max_length=64),
) -> dict[str, Any]:
    """What this result can produce, and what the caller's plan unlocks.

    One request rather than one per format. The tray needs every button and
    every lock state at once, and three round trips to render a toolbar is
    three chances for it to render half-populated.
    """
    resolved = _source_type(source_type)
    source = await sources.resolve(db, source_type=resolved, source_id=source_id, identity=identity)

    return ok(
        ExportOptionsOut(
            source_type=resolved.value,
            source_id=source.id,
            title=source.title,
            artifacts=[
                ArtifactOptionOut(
                    type=descriptor.type,
                    label=descriptor.label,
                    description=descriptor.description,
                    filename=descriptor.filename,
                    format=descriptor.format,
                    emitted=descriptor.emitted,
                )
                for descriptor in artifacts.available(source)
            ],
            formats=[
                FormatOptionOut(
                    format=export_format.value,
                    label=FORMAT_LABELS[export_format],
                    extension=export_service.EXTENSIONS[export_format],
                    required_plan=export_service.required_plan(export_format).value,
                    available=export_service.allowed(export_format, identity),
                )
                for export_format in ExportFormat
            ],
            tables=sorted(export_service.tables_of(source)),
        )
    )


def _source_type(raw: str) -> SourceType:
    try:
        return SourceType(raw)
    except ValueError:
        raise NotFound("Exports are available for runs and stacks.") from None


@router.post("", response_model=Envelope[ExportOut], name="create_export", status_code=201)
async def create_export(db: Db, identity: CallerIdentity, payload: ExportIn) -> dict[str, Any]:
    resolved = _source_type(payload.source_type)
    source = await sources.resolve(
        db, source_type=resolved, source_id=payload.source_id, identity=identity
    )
    # Written sections for the exports that show them, and the source
    # unchanged for the ones that do not. Attached before `create` so the
    # inline render and the queued build receive the same object.
    source = await export_service.narrated(
        db,
        identity,
        source,
        artifact_type=payload.artifact_type,
        export_format=ExportFormat(payload.format),
    )
    export = await export_service.create(
        db,
        identity,
        source=source,
        source_type=resolved,
        export_format=ExportFormat(payload.format),
        artifact_type=payload.artifact_type,
        table=payload.table,
    )
    return ok(_out(export))


@router.get("", response_model=Envelope[list[ExportOut]], name="list_exports")
async def list_exports(
    db: Db, identity: CallerIdentity, limit: int = Query(default=25, ge=1, le=100)
) -> dict[str, Any]:
    rows = await export_service.list_for(db, identity, limit=limit)
    return ok([_out(export) for export in rows])


@router.get("/{export_id}", response_model=Envelope[ExportOut], name="get_export")
async def get_export(db: Db, identity: CallerIdentity, export_id: str) -> dict[str, Any]:
    return ok(_out(await export_service.get(db, export_id, identity)))


@router.get(
    "/{export_id}/download",
    name="download_export",
    # The bytes are not an envelope. Declared explicitly so the generated
    # client does not type this as `Envelope<unknown>` and the browser gets a
    # real Content-Disposition rather than a JSON string.
    response_class=Response,
    responses={200: {"content": {"application/octet-stream": {}}}},
)
async def download_export(db: Db, identity: CallerIdentity, export_id: str) -> Response:
    export = await export_service.get(db, export_id, identity)
    if export.content is None:
        # Pending or failed. 404 rather than an empty 200 — a zero-byte file
        # landing in someone's downloads folder is a worse answer than an error
        # the client can show.
        raise NotFound(
            "This export is not ready yet."
            if export.error is None
            else f"This export failed: {export.error}"
        )

    return Response(
        content=export.content,
        media_type=export.content_type,
        headers={
            "Content-Disposition": f'attachment; filename="{export.filename}"',
            "Content-Length": str(export.size_bytes),
            # An export is a capability-free, owner-scoped download. Caching it
            # in a shared proxy would serve one user's plan to another.
            "Cache-Control": "private, max-age=0, no-store",
        },
    )


__all__ = ["router"]
