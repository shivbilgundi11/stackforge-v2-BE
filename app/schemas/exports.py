"""Export and share wire shapes (M18).

`ExportOut` carries `status` even when the export was rendered inside the
request that asked for it. The client polls and downloads through one path
whichever happened — a client that branched on "was this one fast?" would grow
a second code path for the rare case, and the rare case is the one that rots.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

ExportFormatName = Literal["markdown", "json", "yaml", "csv", "pdf", "zip"]
SourceTypeName = Literal["run", "stack"]


class ArtifactOptionOut(BaseModel):
    """One row in the artifact tray."""

    type: str
    label: str
    description: str
    filename: str
    format: str
    #: Produced by the tool during the run rather than composed on export. The
    #: tray says so, because a user exporting a compose file expects the exact
    #: one they were shown.
    emitted: bool


class FormatOptionOut(BaseModel):
    """A format button, with its lock state.

    `available: false` is rendered as a visible badge rather than a hidden
    button. The gate has to be seen to convert — a user who never learns PDF
    export exists never upgrades for it.
    """

    format: ExportFormatName
    label: str
    extension: str
    required_plan: str
    available: bool


class ExportOptionsOut(BaseModel):
    source_type: SourceTypeName
    source_id: str
    title: str
    artifacts: list[ArtifactOptionOut]
    formats: list[FormatOptionOut]
    #: Table names a CSV export can name. Empty when the result has nothing
    #: rectangular, which is why the CSV button can be present and still refuse.
    tables: list[str] = Field(default_factory=list)


class ExportIn(BaseModel):
    source_type: SourceTypeName
    source_id: str = Field(min_length=1, max_length=64)
    format: ExportFormatName
    #: Which artifact within the source. Omit for the whole result.
    artifact_type: str | None = Field(default=None, max_length=60)
    #: Required for CSV when the source has more than one table.
    table: str | None = Field(default=None, max_length=60)


class ExportOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    source_type: SourceTypeName
    source_id: str
    artifact_type: str | None
    format: ExportFormatName
    status: Literal["pending", "ready", "failed"]
    filename: str
    content_type: str
    size_bytes: int
    error: str | None
    expires_at: datetime
    created_at: datetime
    completed_at: datetime | None
    #: Where to GET the bytes. Present on every row including pending ones, so
    #: the client does not have to construct a URL from an id.
    download_url: str


# ── shares ───────────────────────────────────────────────────────────────────


class ShareIn(BaseModel):
    target_type: SourceTypeName
    target_id: str = Field(min_length=1, max_length=64)
    artifact_type: str | None = Field(default=None, max_length=60)
    #: Optional. A link with no expiry is normal — revocation is the primary
    #: control, and forcing an expiry on every link would make the common case
    #: an unnecessary decision.
    expires_in_days: int | None = Field(default=None, ge=1, le=365)


class ShareOut(BaseModel):
    """The owner's view. Includes the token, because the owner has to be able
    to re-copy the URL (D-14)."""

    id: str
    url: str
    title: str
    target_type: SourceTypeName
    target_id: str
    artifact_type: str | None
    view_count: int
    last_viewed_at: datetime | None
    expires_at: datetime | None
    revoked_at: datetime | None
    created_at: datetime


class SharePayloadOut(BaseModel):
    """The public view. No owner field exists on this model, by construction.

    Adding one would be a deliberate edit rather than an accident of
    serialising a row — which is the point of assembling it field by field in
    `share_service` instead of dumping the source.
    """

    title: str
    subtitle: str
    kind: str
    markdown: str
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    tables: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)
    warnings: list[dict[str, Any]] = Field(default_factory=list)
    provenance: dict[str, Any] = Field(default_factory=dict)
    created_at: str
    expires_at: str | None = None
    # No `view_count`. It is the owner's figure — "how many people opened
    # this" — and a reader who can see it learns something about the owner's
    # activity that sharing a document was never consent for.
