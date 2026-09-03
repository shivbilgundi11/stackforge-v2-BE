"""Export and share endpoints, end to end (M18).

The centre of gravity is the share suite. Everything there is a security
property — 404 rather than 403, no owner identity, revoke takes effect
immediately — and each of them is the kind of thing that works in
development and leaks in production if nobody asserted it.
"""

from __future__ import annotations

import io
import json
import zipfile
from typing import Any

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import utcnow
from app.models.export import ShareLink
from app.models.user import Plan, User
from tests.conftest import GOOD_PASSWORD, register_and_verify

pytestmark = pytest.mark.usefixtures("seeded_catalog")

EXPORTS = "/api/v1/exports"
SHARES = "/api/v1/shares"
STACKS = "/api/v1/stacks"
LLM_PRICING = "/api/v1/tools/cost/llm-pricing"

BASE_RUN = {
    "model_id": "gpt-4o-mini",
    "input_tokens": 1000,
    "output_tokens": 500,
    "requests_per_day": 100,
}


async def _sign_in(
    client: AsyncClient, db: AsyncSession, email: str, *, plan: Plan = Plan.PRO
) -> User:
    user_id = await register_and_verify(client, db, email=email)
    user = await db.get(User, user_id)
    assert user is not None
    user.plan = plan
    await db.flush()

    response = await client.post(
        "/api/v1/auth/login", json={"email": email, "password": GOOD_PASSWORD}
    )
    client.headers["Authorization"] = f"Bearer {response.json()['data']['tokens']['access_token']}"
    return user


async def _a_run(client: AsyncClient) -> str:
    response = await client.post(LLM_PRICING, json=BASE_RUN)
    assert response.status_code == 200
    return str(response.json()["data"]["run_id"])


async def _a_stack(client: AsyncClient, name: str = "Client X RAG rollout") -> str:
    response = await client.post(
        STACKS,
        json={
            "name": name,
            "component_slugs": ["anthropic-api", "llamaindex", "qdrant", "postgresql"],
            "requirements": {"use_case": "rag", "monthly_budget": 2000},
        },
    )
    assert response.status_code == 201, response.text
    return str(response.json()["data"]["id"])


# ── options ──────────────────────────────────────────────────────────────────


async def test_the_tray_gets_every_artifact_and_format_in_one_request(
    client: AsyncClient, db: AsyncSession
) -> None:
    await _sign_in(client, db, "tray@example.com")
    stack_id = await _a_stack(client)

    response = await client.get(
        f"{EXPORTS}/options", params={"source_type": "stack", "source_id": stack_id}
    )
    assert response.status_code == 200
    data = response.json()["data"]

    types = {artifact["type"] for artifact in data["artifacts"]}
    assert {"architecture", "diagram", "roadmap", "compose", "cursor-rules"} <= types
    assert {option["format"] for option in data["formats"]} == {
        "markdown",
        "json",
        "yaml",
        "csv",
        "pdf",
        "zip",
    }
    assert "components" in data["tables"]


async def test_locked_formats_are_listed_rather_than_hidden(
    client: AsyncClient, db: AsyncSession
) -> None:
    """The gate has to be seen to convert. A user who never learns PDF export
    exists never upgrades for it."""
    await _sign_in(client, db, "free@example.com", plan=Plan.FREE)
    run_id = await _a_run(client)

    response = await client.get(
        f"{EXPORTS}/options", params={"source_type": "run", "source_id": run_id}
    )
    by_format = {option["format"]: option for option in response.json()["data"]["formats"]}

    assert by_format["markdown"]["available"] is True
    assert by_format["pdf"]["available"] is False
    assert by_format["pdf"]["required_plan"] == "pro"


# ── creating and downloading ─────────────────────────────────────────────────


async def test_an_export_is_ready_and_downloadable_in_one_round_trip(
    client: AsyncClient, db: AsyncSession
) -> None:
    await _sign_in(client, db, "download@example.com")
    stack_id = await _a_stack(client)

    created = await client.post(
        EXPORTS, json={"source_type": "stack", "source_id": stack_id, "format": "markdown"}
    )
    assert created.status_code == 201, created.text
    export = created.json()["data"]
    assert export["status"] == "ready"
    assert export["size_bytes"] > 0

    downloaded = await client.get(export["download_url"])
    assert downloaded.status_code == 200
    assert downloaded.headers["content-type"].startswith("text/markdown")
    assert "attachment" in downloaded.headers["content-disposition"]
    assert b"# Stack architecture" in downloaded.content


async def test_a_free_account_can_export_markdown(client: AsyncClient) -> None:
    run_id = await _a_run(client)

    created = await client.post(
        EXPORTS, json={"source_type": "run", "source_id": run_id, "format": "markdown"}
    )
    assert created.status_code == 201, created.text

    downloaded = await client.get(created.json()["data"]["download_url"])
    assert downloaded.status_code == 200
    assert b"Cost result" in downloaded.content


async def test_a_free_user_requesting_pdf_gets_402_with_the_required_plan(
    client: AsyncClient, db: AsyncSession
) -> None:
    await _sign_in(client, db, "gated@example.com", plan=Plan.FREE)
    run_id = await _a_run(client)

    response = await client.post(
        EXPORTS, json={"source_type": "run", "source_id": run_id, "format": "pdf"}
    )
    assert response.status_code == 402
    body = response.json()
    assert body["error"]["code"] == "PLAN_REQUIRED"
    assert body["error"]["details"]["required_plan"] == "pro"


async def test_a_pro_user_gets_a_real_pdf(client: AsyncClient, db: AsyncSession) -> None:
    await _sign_in(client, db, "pdf@example.com")
    stack_id = await _a_stack(client)

    created = await client.post(
        EXPORTS, json={"source_type": "stack", "source_id": stack_id, "format": "pdf"}
    )
    assert created.status_code == 201, created.text

    downloaded = await client.get(created.json()["data"]["download_url"])
    assert downloaded.headers["content-type"] == "application/pdf"
    assert downloaded.content.startswith(b"%PDF-")


async def test_the_bundle_unzips_and_holds_the_whole_plan(
    client: AsyncClient, db: AsyncSession
) -> None:
    await _sign_in(client, db, "bundle@example.com")
    stack_id = await _a_stack(client)

    created = await client.post(
        EXPORTS, json={"source_type": "stack", "source_id": stack_id, "format": "zip"}
    )
    downloaded = await client.get(created.json()["data"]["download_url"])

    with zipfile.ZipFile(io.BytesIO(downloaded.content)) as archive:
        assert archive.testzip() is None
        names = set(archive.namelist())

    assert "client-x-rag-rollout-plan/README.md" in names
    assert "client-x-rag-rollout-plan/deploy/docker-compose.yml" in names
    assert "client-x-rag-rollout-plan/.cursorrules" in names


async def test_json_export_round_trips(client: AsyncClient, db: AsyncSession) -> None:
    await _sign_in(client, db, "structured@example.com")
    run_id = await _a_run(client)

    created = await client.post(
        EXPORTS, json={"source_type": "run", "source_id": run_id, "format": "json"}
    )
    downloaded = await client.get(created.json()["data"]["download_url"])
    payload = json.loads(downloaded.content)

    assert payload["stackforge"]["schema"] == "stackforge.export/v1"
    assert payload["result"]["run_id"] == run_id


async def test_re_exporting_produces_identical_bytes(client: AsyncClient, db: AsyncSession) -> None:
    """FR-11, asserted through the API rather than only at the service layer."""
    await _sign_in(client, db, "idempotent@example.com")
    stack_id = await _a_stack(client)

    async def once() -> bytes:
        created = await client.post(
            EXPORTS, json={"source_type": "stack", "source_id": stack_id, "format": "zip"}
        )
        return (await client.get(created.json()["data"]["download_url"])).content

    assert await once() == await once()


# ── ownership ────────────────────────────────────────────────────────────────


async def test_an_export_id_is_not_a_capability(client: AsyncClient, db: AsyncSession) -> None:
    await _sign_in(client, db, "owner@example.com")
    stack_id = await _a_stack(client)
    created = await client.post(
        EXPORTS, json={"source_type": "stack", "source_id": stack_id, "format": "markdown"}
    )
    export_id = created.json()["data"]["id"]

    await _sign_in(client, db, "stranger@example.com")
    assert (await client.get(f"{EXPORTS}/{export_id}")).status_code == 404
    assert (await client.get(f"{EXPORTS}/{export_id}/download")).status_code == 404


async def test_exporting_someone_elses_stack_is_not_found(
    client: AsyncClient, db: AsyncSession
) -> None:
    await _sign_in(client, db, "author@example.com")
    stack_id = await _a_stack(client)

    await _sign_in(client, db, "intruder@example.com")
    response = await client.post(
        EXPORTS, json={"source_type": "stack", "source_id": stack_id, "format": "markdown"}
    )
    assert response.status_code == 404


# ── the purge ────────────────────────────────────────────────────────────────


async def _an_expired_export(client: AsyncClient, db: AsyncSession, stack_id: str) -> str:
    from datetime import timedelta

    from app.models.export import Export

    created = await client.post(
        EXPORTS, json={"source_type": "stack", "source_id": stack_id, "format": "markdown"}
    )
    export_id = str(created.json()["data"]["id"])

    export = await db.get(Export, export_id)
    assert export is not None
    export.expires_at = utcnow() - timedelta(seconds=1)
    await db.flush()
    return export_id


async def test_an_expired_export_is_unreadable_before_the_purge_runs(
    client: AsyncClient, db: AsyncSession
) -> None:
    """The expiry is honoured on read, not only by the nightly job. Otherwise
    an export is live for up to a day past the date it claims to die.

    One request per test: the 404 propagates out of the handler and the
    fixture's session override rolls the transaction back with it, which
    undoes the expiry this test just wrote. The download path is asserted
    separately below for the same reason.
    """
    await _sign_in(client, db, "expiring.read@example.com")
    export_id = await _an_expired_export(client, db, await _a_stack(client))

    assert (await client.get(f"{EXPORTS}/{export_id}")).status_code == 404


async def test_an_expired_export_cannot_be_downloaded(
    client: AsyncClient, db: AsyncSession
) -> None:
    await _sign_in(client, db, "expiring.download@example.com")
    export_id = await _an_expired_export(client, db, await _a_stack(client))

    assert (await client.get(f"{EXPORTS}/{export_id}/download")).status_code == 404


async def test_the_purge_deletes_expired_rows_and_their_bytes(
    client: AsyncClient, db: AsyncSession
) -> None:
    from app.models.export import Export
    from app.services import export_service

    await _sign_in(client, db, "expiring.purge@example.com")
    export_id = await _an_expired_export(client, db, await _a_stack(client))

    assert await export_service.purge_expired(db) >= 1
    # The bytes live on the row, so one delete is the whole cleanup — there is
    # no window where the row is gone and an object is still being paid for.
    assert await db.get(Export, export_id) is None


# ── shares ───────────────────────────────────────────────────────────────────


async def _share(client: AsyncClient, **payload: Any) -> dict[str, Any]:
    response = await client.post(SHARES, json=payload)
    assert response.status_code == 201, response.text
    return dict(response.json()["data"])


def _token_of(share: dict[str, Any]) -> str:
    return share["url"].rsplit("/", 1)[-1]


async def test_a_share_link_opens_logged_out(client: AsyncClient, db: AsyncSession) -> None:
    await _sign_in(client, db, "sharer@example.com")
    stack_id = await _a_stack(client)
    share = await _share(client, target_type="stack", target_id=stack_id)

    client.headers.pop("Authorization")
    response = await client.get(f"/api/v1/s/{_token_of(share)}")

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["title"] == "Client X RAG rollout"
    assert "# Stack architecture" in data["markdown"]


async def test_the_share_page_carries_noindex(client: AsyncClient, db: AsyncSession) -> None:
    await _sign_in(client, db, "noindex@example.com")
    share = await _share(client, target_type="stack", target_id=await _a_stack(client))

    client.headers.pop("Authorization")
    response = await client.get(f"/api/v1/s/{_token_of(share)}")

    assert response.headers["x-robots-tag"] == "noindex, nofollow, noarchive"


async def test_the_share_payload_exposes_no_owner_identity(
    client: AsyncClient, db: AsyncSession
) -> None:
    """A link that leaks who made it is a link people stop sending."""
    user = await _sign_in(client, db, "private.person@example.com")
    share = await _share(client, target_type="stack", target_id=await _a_stack(client))

    client.headers.pop("Authorization")
    body = (await client.get(f"/api/v1/s/{_token_of(share)}")).text

    assert user.id not in body
    assert "private.person@example.com" not in body
    assert "Ada Lovelace" not in body
    for forbidden in ("user_id", "owner", "email", "view_count"):
        assert forbidden not in body


async def test_revoking_breaks_the_link_with_404_not_403(
    client: AsyncClient, db: AsyncSession
) -> None:
    """A 403 would confirm the resource still exists, which is exactly what a
    former recipient must not learn."""
    await _sign_in(client, db, "revoker@example.com")
    share = await _share(client, target_type="stack", target_id=await _a_stack(client))
    token = _token_of(share)

    assert (await client.delete(f"{SHARES}/{share['id']}")).status_code == 200

    response = await client.get(f"/api/v1/s/{token}")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "NOT_FOUND"


async def test_an_expired_token_returns_404(client: AsyncClient, db: AsyncSession) -> None:
    from datetime import timedelta

    await _sign_in(client, db, "expired@example.com")
    share = await _share(
        client, target_type="stack", target_id=await _a_stack(client), expires_in_days=1
    )

    link = await db.get(ShareLink, share["id"])
    assert link is not None
    link.expires_at = utcnow() - timedelta(seconds=1)
    await db.flush()

    assert (await client.get(f"/api/v1/s/{_token_of(share)}")).status_code == 404


async def test_an_unknown_token_returns_404(client: AsyncClient) -> None:
    assert (await client.get("/api/v1/s/not-a-real-token")).status_code == 404


async def test_a_view_is_counted_for_the_owner(client: AsyncClient, db: AsyncSession) -> None:
    await _sign_in(client, db, "counter@example.com")
    share = await _share(client, target_type="stack", target_id=await _a_stack(client))
    token = _token_of(share)
    authorization = client.headers["Authorization"]

    client.headers.pop("Authorization")
    await client.get(f"/api/v1/s/{token}")
    await client.get(f"/api/v1/s/{token}")

    client.headers["Authorization"] = authorization
    listed = (await client.get(SHARES)).json()["data"]
    assert listed[0]["view_count"] == 2
    assert listed[0]["last_viewed_at"] is not None


async def test_a_share_can_target_one_artifact(client: AsyncClient, db: AsyncSession) -> None:
    await _sign_in(client, db, "artifact.sharer@example.com")
    share = await _share(
        client,
        target_type="stack",
        target_id=await _a_stack(client),
        artifact_type="cursor-rules",
    )

    client.headers.pop("Authorization")
    data = (await client.get(f"/api/v1/s/{_token_of(share)}")).json()["data"]

    assert data["kind"] == "artifact"
    assert "Cursor rules" in data["title"]
    assert "Do not invent pricing" in data["markdown"]


async def test_sharing_an_artifact_the_source_cannot_produce_is_refused(
    client: AsyncClient, db: AsyncSession
) -> None:
    """Fail at mint time rather than on the page. A link that 404s for the
    person it was sent to is worse than a mint that refuses."""
    await _sign_in(client, db, "bad.artifact@example.com")
    response = await client.post(
        SHARES,
        json={
            "target_type": "run",
            "target_id": await _a_run(client),
            "artifact_type": "cursor-rules",
        },
    )
    assert response.status_code == 404


async def test_sharing_requires_an_account(anon_client: AsyncClient) -> None:
    """A caller with no session cannot revoke later, and revocation is the only
    protection that survives forwarding.

    Every route needs an account now, so this is no longer the *first* gate a
    sharer meets — but it is still the one that matters, and a share endpoint
    that ever stopped checking would be the worst possible place to find out.
    """
    response = await anon_client.post(
        SHARES, json={"target_type": "run", "target_id": "run_whatever"}
    )
    assert response.status_code == 401


async def test_bulk_revoke_kills_every_live_link(client: AsyncClient, db: AsyncSession) -> None:
    await _sign_in(client, db, "bulk@example.com")
    first = await _share(client, target_type="stack", target_id=await _a_stack(client, "One"))
    second = await _share(client, target_type="stack", target_id=await _a_stack(client, "Two"))

    response = await client.delete(SHARES)
    assert response.json()["data"] == {"revoked": 2}

    client.headers.pop("Authorization")
    for share in (first, second):
        assert (await client.get(f"/api/v1/s/{_token_of(share)}")).status_code == 404


async def test_revoking_keeps_the_row_and_the_count(client: AsyncClient, db: AsyncSession) -> None:
    """Only the capability dies. The owner keeps the record that the link
    existed and how often it was opened."""
    await _sign_in(client, db, "history@example.com")
    share = await _share(client, target_type="stack", target_id=await _a_stack(client))
    await client.delete(f"{SHARES}/{share['id']}")

    listed = (await client.get(SHARES, params={"include_revoked": True})).json()["data"]
    assert len(listed) == 1
    assert listed[0]["revoked_at"] is not None

    assert (await client.get(SHARES)).json()["data"] == []


async def test_a_stranger_cannot_revoke_someone_elses_link(
    client: AsyncClient, db: AsyncSession
) -> None:
    await _sign_in(client, db, "victim@example.com")
    share = await _share(client, target_type="stack", target_id=await _a_stack(client))

    await _sign_in(client, db, "attacker@example.com")
    assert (await client.delete(f"{SHARES}/{share['id']}")).status_code == 404

    client.headers.pop("Authorization")
    assert (await client.get(f"/api/v1/s/{_token_of(share)}")).status_code == 200


# ── projects hold artifacts now ──────────────────────────────────────────────


async def test_an_export_can_be_saved_into_a_project(client: AsyncClient, db: AsyncSession) -> None:
    """M17 refused `artifact` items because nothing resolved them. M18 gives
    them a table, so the tray's "save to project" works."""
    await _sign_in(client, db, "filer@example.com")
    stack_id = await _a_stack(client)
    export = (
        await client.post(
            EXPORTS, json={"source_type": "stack", "source_id": stack_id, "format": "markdown"}
        )
    ).json()["data"]

    project = (await client.post("/api/v1/projects", json={"name": "Client X"})).json()["data"]
    added = await client.post(
        f"/api/v1/projects/{project['id']}/items",
        json={"item_type": "artifact", "item_id": export["id"]},
    )

    assert added.status_code == 201, added.text
    item = added.json()["data"]
    assert item["title"] == export["filename"]
    assert item["href"] == export["download_url"]


async def test_a_project_cannot_hold_someone_elses_export(
    client: AsyncClient, db: AsyncSession
) -> None:
    await _sign_in(client, db, "export.owner@example.com")
    export = (
        await client.post(
            EXPORTS,
            json={
                "source_type": "stack",
                "source_id": await _a_stack(client),
                "format": "markdown",
            },
        )
    ).json()["data"]

    await _sign_in(client, db, "project.owner@example.com")
    project = (await client.post("/api/v1/projects", json={"name": "Not mine"})).json()["data"]
    response = await client.post(
        f"/api/v1/projects/{project['id']}/items",
        json={"item_type": "artifact", "item_id": export["id"]},
    )

    assert response.status_code == 404
