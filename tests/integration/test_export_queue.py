"""The queued-bundle path (M18).

The fallback test is the one that matters operationally. `enqueue` returns
False when Redis is unreachable, and it really is unreachable here — the call
goes out to a closed port rather than to a mock. A queue being down has to make
the product slower, never lossy: a `pending` row nothing will ever pick up is a
download button that spins forever.

What is *not* covered here, deliberately: `build_export` running end to end
under a real worker. The job opens its own `SessionLocal`, which cannot see a
test's uncommitted transaction, so invoking it would mean either committing
fixtures outside the rollback or rebinding the session maker — both of which
test the harness rather than the job. Its two steps, `render` and `complete`,
are asserted directly instead, and its error path is asserted through `fail`.
"""

from __future__ import annotations

import io
import zipfile

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import Identity
from app.core.config import settings
from app.models.export import Export, ExportFormat, ExportStatus, SourceType
from app.models.user import Plan, User
from app.services import export_service
from app.services.artifacts import sources
from app.workers import queue
from tests.conftest import GOOD_PASSWORD, register_and_verify

pytestmark = pytest.mark.usefixtures("seeded_catalog")

STACKS = "/api/v1/stacks"


async def _pro_user(client: AsyncClient, db: AsyncSession, email: str) -> User:
    user_id = await register_and_verify(client, db, email=email)
    user = await db.get(User, user_id)
    assert user is not None
    user.plan = Plan.PRO
    await db.flush()

    response = await client.post(
        "/api/v1/auth/login", json={"email": email, "password": GOOD_PASSWORD}
    )
    client.headers["Authorization"] = f"Bearer {response.json()['data']['tokens']['access_token']}"
    return user


async def _a_stack(client: AsyncClient) -> str:
    response = await client.post(
        STACKS,
        json={
            "name": "Queued plan",
            "component_slugs": ["anthropic-api", "llamaindex", "qdrant", "postgresql"],
            "requirements": {"use_case": "rag"},
        },
    )
    assert response.status_code == 201, response.text
    return str(response.json()["data"]["id"])


async def test_a_bundle_over_the_threshold_is_built_when_the_queue_is_down(
    client: AsyncClient, db: AsyncSession
) -> None:
    """Slower, never lossy. This is the branch a real outage takes."""
    from app.models.stack import Stack

    user = await _pro_user(client, db, "queue.down@example.com")
    stack = await db.get(Stack, await _a_stack(client))
    assert stack is not None

    identity = Identity(user=user, anonymous_id=None, session_id=None)
    source = await sources.stack_source_of(db, stack)

    original = settings.export_async_threshold_bytes
    settings.export_async_threshold_bytes = 1
    try:
        assert export_service.should_queue(source, ExportFormat.ZIP)
        export = await export_service.create(
            db,
            identity,
            source=source,
            source_type=SourceType.STACK,
            export_format=ExportFormat.ZIP,
        )
    finally:
        settings.export_async_threshold_bytes = original

    assert export.status is ExportStatus.READY
    assert export.content is not None
    with zipfile.ZipFile(io.BytesIO(export.content)) as archive:
        assert "queued-plan-plan/README.md" in archive.namelist()


async def test_completing_a_pending_row_stores_the_bytes(
    client: AsyncClient, db: AsyncSession
) -> None:
    """The two steps `build_export` performs, over the test's transaction.

    The job itself is not invoked here: it opens its own `SessionLocal`, which
    would not see this test's uncommitted rows. What it does *inside* that
    session is `render` then `complete`, and those are what this asserts.
    """
    from app.models.stack import Stack

    user = await _pro_user(client, db, "queue.job@example.com")
    stack_id = await _a_stack(client)
    stack = await db.get(Stack, stack_id)
    assert stack is not None
    source = await sources.stack_source_of(db, stack)

    export = Export(
        user_id=user.id,
        source_type=SourceType.STACK,
        source_id=stack_id,
        format=ExportFormat.ZIP,
        status=ExportStatus.PENDING,
        filename="queued-plan-plan.zip",
        content_type="application/zip",
        size_bytes=0,
        expires_at=export_service.utcnow(),
        created_at=export_service.utcnow(),
    )
    db.add(export)
    await db.flush()

    rendered = export_service.render(source, export_format=ExportFormat.ZIP)
    export_service.complete(export, rendered)
    await db.flush()

    assert export.status is ExportStatus.READY
    assert export.size_bytes == len(rendered.data)
    assert export.error is None


async def test_a_failed_build_records_the_reason_on_the_row() -> None:
    """A job that vanished is a spinner that never stops. The row is the error
    channel, so the UI has something to show."""
    export = Export(
        source_type=SourceType.STACK,
        source_id="stk_gone",
        format=ExportFormat.ZIP,
        status=ExportStatus.PENDING,
        filename="x.zip",
        content_type="application/zip",
        size_bytes=0,
        expires_at=export_service.utcnow(),
        created_at=export_service.utcnow(),
    )
    export_service.fail(export, "No stack with that id.")

    assert export.status is ExportStatus.FAILED
    assert export.error == "No stack with that id."
    assert export.completed_at is not None


async def test_enqueue_reports_failure_rather_than_raising() -> None:
    """Every caller has an inline fallback. One that had to wrap this in a
    try/except would eventually be written without one."""
    original = settings.redis_url
    settings.redis_url = "redis://127.0.0.1:6399/0"  # nothing listens here
    try:
        assert await queue.enqueue(queue.BUILD_EXPORT, "exp_nope") is False
    finally:
        settings.redis_url = original
