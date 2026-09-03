"""Share links, and the public page they open (M18).

Two audiences in one file, and the split is deliberate: everything under
`/shares` requires an account, and `/s/{token}` requires nothing at all. Having
them adjacent is what makes it obvious that the public handler returns a
different model from the owner's — `SharePayloadOut` has no owner field, and
`ShareOut` has a token, and neither is reachable from the other's route.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query, Response

from app.api.deps import CurrentUser, Db
from app.core.errors import NotFound
from app.core.responses import Envelope, ok
from app.models.export import ShareLink, SourceType
from app.schemas.exports import ShareIn, ShareOut, SharePayloadOut
from app.services import share_service
from app.services.artifacts import sources

router = APIRouter(tags=["shares"])

#: Belt and braces with the `<meta name="robots">` the public page renders.
#: A header covers the case where a crawler reads the API directly, which it
#: will, because the payload URL is in the page's network log.
NOINDEX = "noindex, nofollow, noarchive"


def _out(link: ShareLink) -> ShareOut:
    return ShareOut(
        id=link.id,
        url=share_service.url_for(link),
        title=link.title,
        target_type=link.target_type.value,
        target_id=link.target_id,
        artifact_type=link.artifact_type,
        view_count=link.view_count,
        last_viewed_at=link.last_viewed_at,
        expires_at=link.expires_at,
        revoked_at=link.revoked_at,
        created_at=link.created_at,
    )


@router.post("/shares", response_model=Envelope[ShareOut], name="create_share", status_code=201)
async def create_share(db: Db, user: CurrentUser, payload: ShareIn) -> dict[str, Any]:
    from app.api.deps import Identity

    try:
        target_type = SourceType(payload.target_type)
    except ValueError:
        raise NotFound("Runs and stacks can be shared.") from None

    source = await sources.resolve(
        db,
        source_type=target_type,
        source_id=payload.target_id,
        identity=Identity(user=user, session_id=None),
    )
    link = await share_service.create(
        db,
        user,
        source=source,
        target_type=target_type,
        artifact_type=payload.artifact_type,
        expires_in_days=payload.expires_in_days,
    )
    return ok(_out(link))


@router.get("/shares", response_model=Envelope[list[ShareOut]], name="list_shares")
async def list_shares(
    db: Db,
    user: CurrentUser,
    include_revoked: bool = Query(default=False),
) -> dict[str, Any]:
    links = await share_service.list_for(db, user, include_revoked=include_revoked)
    return ok([_out(link) for link in links])


@router.delete("/shares", response_model=Envelope[dict[str, int]], name="revoke_all_shares")
async def revoke_all_shares(db: Db, user: CurrentUser) -> dict[str, Any]:
    """Bulk revoke, for Settings → Shares.

    Returns the count rather than 204: "12 links revoked" is a confirmation
    the user can check against what they thought they had out there.
    """
    return ok({"revoked": await share_service.revoke_all(db, user)})


@router.delete("/shares/{share_id}", response_model=Envelope[ShareOut], name="revoke_share")
async def revoke_share(db: Db, user: CurrentUser, share_id: str) -> dict[str, Any]:
    """The row survives revocation, so the owner keeps the view count and the
    record that the link existed. Only the capability dies."""
    return ok(_out(await share_service.revoke(db, user, share_id)))


# ── the public page ─────────────────────────────────────────────────────────


@router.get("/s/{token}", response_model=Envelope[SharePayloadOut], name="get_shared")
async def get_shared(db: Db, response: Response, token: str) -> dict[str, Any]:
    """Unauthenticated, `noindex`, and 404 for anything that is not live.

    No dependency on `CurrentUser` or `CallerIdentity` at all: the handler has
    no way to know who is asking, which is the strongest form of "no owner
    identity is exposed" available — there is nothing here to leak.
    """
    response.headers["X-Robots-Tag"] = NOINDEX

    link = await share_service.resolve(db, token)
    payload = await share_service.payload_for(db, link)
    await share_service.record_view(db, link)

    return ok(
        SharePayloadOut(
            title=payload.title,
            subtitle=payload.subtitle,
            kind=payload.kind,
            markdown=payload.markdown,
            artifacts=payload.artifacts,
            metrics=payload.metrics,
            tables=payload.tables,
            warnings=payload.warnings,
            provenance=payload.provenance,
            created_at=payload.created_at,
            expires_at=payload.expires_at,
        )
    )
