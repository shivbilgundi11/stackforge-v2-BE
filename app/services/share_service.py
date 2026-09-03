"""Public share links (M18).

`PRD.md` §24 lists shareable URLs as a retention mechanic: the recipient lands
on StackForge and is prompted to run their own session. That makes the share
page's CTA the whole reason for building it, and it makes the security rules
below non-negotiable — a link that leaks the owner's identity is a link people
stop sending.

Four rules, all from `Workflows/Phase-9`, all enforced here rather than in the
router:

**A revoked or expired token returns 404, not 403.** A 403 confirms the
resource exists, which turns token guessing into an enumeration oracle and
tells a former recipient that the thing they lost access to is still there.

**No owner identity, anywhere.** Not in the payload, not in the page, not in a
`created_by` field someone adds later "just for the UI". The payload model is
built here and contains no user column, so adding one would be a deliberate
act rather than an oversight.

**`noindex` by default, with no opt-in.** There is no public-index switch in
this program. A share link is a capability URL, and a capability URL in a
search index is a capability URL that is no longer a capability.

**Optional expiry, always revocable.** Revocation is the only protection that
survives the link being forwarded, so it can never be unavailable.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import Identity
from app.core.config import settings
from app.core.database import utcnow
from app.core.errors import NotFound, ValidationFailed
from app.core.logging import get_logger
from app.models.export import ShareLink, SourceType
from app.models.user import User
from app.services import artifacts
from app.services.artifacts import result_document
from app.services.artifacts.sources import Source, StackSource

logger = get_logger("shares")

#: 256 bits, URL-safe. 32 bytes of `secrets` becomes 43 characters, which is
#: short enough to paste into a chat message and far past guessing.
TOKEN_BYTES = 32

MAX_EXPIRY_DAYS = 365


def mint_token() -> str:
    return secrets.token_urlsafe(TOKEN_BYTES)


def url_for(link: ShareLink) -> str:
    return f"{settings.web_base_url.rstrip('/')}/s/{link.token}"


@dataclass(frozen=True)
class SharePayload:
    """What a logged-out visitor gets.

    Deliberately assembled rather than serialised from the source: building it
    field by field is what guarantees no owner column rides along when someone
    adds one to `Stack` next month.
    """

    title: str
    subtitle: str
    kind: str
    markdown: str
    artifacts: list[dict[str, Any]]
    metrics: dict[str, Any]
    tables: dict[str, list[dict[str, Any]]]
    warnings: list[dict[str, Any]]
    provenance: dict[str, Any]
    created_at: str
    expires_at: str | None
    view_count: int


async def create(
    db: AsyncSession,
    user: User,
    *,
    source: Source,
    target_type: SourceType,
    artifact_type: str | None = None,
    expires_in_days: int | None = None,
) -> ShareLink:
    """Mint a link over something the caller owns.

    Ownership was established by resolving `source`, which raises `NotFound`
    for anything the caller cannot see — so there is no second check here, and
    no way to reach this function with someone else's row.
    """
    if expires_in_days is not None and not 1 <= expires_in_days <= MAX_EXPIRY_DAYS:
        raise ValidationFailed.on_field(
            "expires_in_days", f"Expiry must be between 1 and {MAX_EXPIRY_DAYS} days."
        )

    if artifact_type is not None:
        # Fail now rather than on the public page. A link that 404s for the
        # person it was sent to is worse than a mint that refuses.
        artifacts.generate(source, artifact_type)

    title, _ = result_document.title_of(source)
    if artifact_type is not None and artifact_type in artifacts.BY_TYPE:
        title = f"{title} — {artifacts.BY_TYPE[artifact_type].label}"

    link = ShareLink(
        user_id=user.id,
        token=mint_token(),
        target_type=target_type,
        target_id=source.id,
        artifact_type=artifact_type,
        title=title[:200],
        expires_at=utcnow() + timedelta(days=expires_in_days) if expires_in_days else None,
        view_count=0,
    )
    db.add(link)
    await db.flush()
    await db.refresh(link)

    logger.info(
        "shares.created",
        share_id=link.id,
        target_type=target_type.value,
        artifact_type=artifact_type,
        expires=bool(link.expires_at),
    )
    return link


async def list_for(
    db: AsyncSession, user: User, *, include_revoked: bool = False
) -> list[ShareLink]:
    statement = (
        select(ShareLink).where(ShareLink.user_id == user.id).order_by(ShareLink.created_at.desc())
    )
    if not include_revoked:
        statement = statement.where(ShareLink.revoked_at.is_(None))
    return list((await db.execute(statement)).scalars().all())


async def revoke(db: AsyncSession, user: User, share_id: str) -> ShareLink:
    link = await db.scalar(
        select(ShareLink).where(ShareLink.id == share_id, ShareLink.user_id == user.id)
    )
    if link is None:
        raise NotFound("No share link with that id.")
    if link.revoked_at is None:
        link.revoked_at = utcnow()
        await db.flush()
        logger.info("shares.revoked", share_id=link.id)
    return link


async def revoke_all(db: AsyncSession, user: User) -> int:
    """Bulk revoke, for Settings → Shares.

    A row-at-a-time loop rather than one UPDATE so every revocation gets the
    same timestamp semantics as a single one and the count is the count of
    links that were actually live.
    """
    live = await list_for(db, user)
    now = utcnow()
    for link in live:
        link.revoked_at = now
    if live:
        await db.flush()
        logger.info("shares.revoked_all", user_id=user.id, count=len(live))
    return len(live)


async def resolve(db: AsyncSession, token: str) -> ShareLink:
    """A token to a live link, or `NotFound`.

    Every dead state — no such token, revoked, expired — produces the same
    404. That is the rule the whole module is arranged around, so it is one
    function with one exit rather than three call sites deciding for
    themselves.
    """
    link = await db.scalar(select(ShareLink).where(ShareLink.token == token))
    if link is None or not link.is_live:
        raise NotFound("This link is no longer available.")
    return link


async def record_view(db: AsyncSession, link: ShareLink) -> None:
    """Count the view for the owner.

    Nothing about the *viewer* is recorded — not the IP, not a fingerprint,
    not a referrer. The owner asked "is anyone reading this", which a count
    answers; anything more would be surveillance the recipient did not agree
    to when they opened a link.
    """
    link.view_count += 1
    link.last_viewed_at = utcnow()
    await db.flush()


async def payload_for(db: AsyncSession, link: ShareLink) -> SharePayload:
    """Render the shared thing for a logged-out reader.

    Resolved with a synthetic owner identity, because the public page is the
    one place where the reader is not the owner and the content still has to
    load. The identity never leaves this function.
    """
    from app.models.stack import Stack
    from app.services.artifacts import sources
    from app.services.tool_service import get_run

    if link.target_type is SourceType.STACK:
        stack = await db.get(Stack, link.target_id)
        if stack is None:
            raise NotFound("This link is no longer available.")
        source: Source = await sources.stack_source_of(db, stack)
    else:
        owner = await db.get(User, link.user_id)
        if owner is None:
            raise NotFound("This link is no longer available.")
        run = await get_run(db, link.target_id, Identity(user=owner, session_id=None))
        if run is None:
            raise NotFound("This link is no longer available.")
        source = sources.run_source_of(run)

    title, subtitle = result_document.title_of(source)

    if link.artifact_type is not None:
        artifact = artifacts.generate(source, link.artifact_type)
        return SharePayload(
            title=link.title,
            subtitle=subtitle,
            kind="artifact",
            markdown=(
                artifact.content
                if artifact.format == "markdown"
                else _fenced(artifact.content, artifact.language or artifact.format)
            ),
            artifacts=[artifact.model_dump(mode="json")],
            metrics={},
            tables={},
            warnings=[],
            provenance=_provenance_of(source),
            created_at=link.created_at.isoformat(),
            expires_at=link.expires_at.isoformat() if link.expires_at else None,
            view_count=link.view_count,
        )

    return SharePayload(
        title=title,
        subtitle=subtitle,
        kind=link.target_type.value,
        markdown=result_document.render(source),
        artifacts=[artifact.model_dump(mode="json") for artifact in _artifacts_of(source)],
        metrics=_metrics_of(source),
        tables=_tables_of(source),
        warnings=_warnings_of(source),
        provenance=_provenance_of(source),
        created_at=link.created_at.isoformat(),
        expires_at=link.expires_at.isoformat() if link.expires_at else None,
        view_count=link.view_count,
    )


def _fenced(content: str, language: str) -> str:
    from app.services.artifacts import markdown as md

    return md.fence(content, language=language)


def _artifacts_of(source: Source) -> list[Any]:
    if isinstance(source, StackSource):
        return artifacts.generate_all(source)
    return source.artifacts


def _metrics_of(source: Source) -> dict[str, Any]:
    if isinstance(source, StackSource):
        return {
            "score": str(source.score.total),
            "components": len(source.components),
            "deprecated_components": len(source.deprecated),
            "compatibility": source.compatibility.overall if source.compatibility else "unknown",
        }
    return {key: str(value) for key, value in source.output.metrics.items()}


def _tables_of(source: Source) -> dict[str, list[dict[str, Any]]]:
    from app.services import export_service

    return export_service.tables_of(source)


def _warnings_of(source: Source) -> list[dict[str, Any]]:
    """Warnings survive sharing.

    A shared plan that drops "this component is deprecated" is the version
    that gets forwarded to the person who will implement it.
    """
    if isinstance(source, StackSource):
        return [
            {
                "level": "critical",
                "message": (
                    f"{tool.name} is marked {tool.status} in the catalog: "
                    f"{tool.status_reason or 'no reason recorded'}."
                ),
            }
            for tool in source.deprecated
        ]
    return [warning.model_dump(mode="json") for warning in source.output.warnings]


def _provenance_of(source: Source) -> dict[str, Any]:
    if isinstance(source, StackSource):
        return {"oldest_verified_at": None, "variant": "fresh", "sources": []}
    return source.output.provenance.model_dump(mode="json")
