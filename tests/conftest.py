"""Test fixtures.

The shape of these determines how pleasant every later module's tests are, so
they get more care than the count of them suggests.

  * `db` runs each test inside a transaction that is rolled back — no
    truncation between tests, no cross-test pollution, fast.
  * `client` is an httpx AsyncClient bound to the app with the session
    dependency overridden onto that same transaction, so a request and the
    assertions after it see one consistent view.
  * Redis is fakeredis and email is the console sender, so the unit and
    integration suites need no containers beyond Postgres.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

os.environ.setdefault("HIBP_ENABLED", "false")  # no network in tests
os.environ.setdefault("EMAIL_PROVIDER", "console")

from app.core.config import settings
from app.core.database import Base, get_session
from app.core.redis import set_redis
from app.integrations import email as email_integration
from app.main import app
from app.services import token_service

TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL",
    settings.database_url.rsplit("/", 1)[0] + "/stackforge_test",
)


def pytest_configure() -> None:
    """Ephemeral signing key, so the suite never depends on a developer's .env
    and never signs anything with a real key.

    The AI key is cleared for the same reason, and a stronger one: a developer
    with `ANTHROPIC_API_KEY` in their `.env` would otherwise have `pytest`
    make live, billable, non-deterministic model calls — and would be running
    a different suite from CI. Tests that exercise the AI layer set a fake key
    themselves, which is also what keeps "the whole suite passes with the key
    unset" true by construction rather than by hoping.
    """
    settings.anthropic_api_key = ""
    if not settings.auth_private_key:
        private_pem, public_pem = token_service.generate_keypair()
        settings.auth_private_key = private_pem
        settings.auth_public_key = public_pem


@pytest.fixture(scope="session")
def anyio_backend() -> str:
    return "asyncio"


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def engine() -> AsyncIterator[object]:
    test_engine = create_async_engine(TEST_DATABASE_URL, poolclass=None)
    async with test_engine.begin() as conn:
        # citext is required by the users.email column.
        from sqlalchemy import text

        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS citext"))
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    yield test_engine
    await test_engine.dispose()


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def seeded_catalog(engine: object) -> None:
    """The real catalog seed, committed once for the whole session.

    Committed outside the per-test transaction on purpose: catalog data is
    reference data, every test reads the same rows, and re-seeding 2,200 rows
    per test would dominate the suite's runtime. Tests that mutate catalog rows
    do so inside their own transaction and are rolled back as usual.

    Tests assert against *real seeded values* — `gpt-4o-mini` really is
    $0.00015 per 1k input — because a test that only checks a status code
    passes just as happily when the seed loads nothing.
    """
    from app.services.seed_service import seed_all

    maker = async_sessionmaker(bind=engine, expire_on_commit=False)  # type: ignore[arg-type]
    async with maker() as session:
        await seed_all(session)
        await session.commit()


@pytest_asyncio.fixture(loop_scope="session")
async def db(engine: object) -> AsyncIterator[AsyncSession]:
    """One transaction per test, rolled back at the end.

    `join_transaction_mode="create_savepoint"` is what makes this work with a
    request handler that commits: the session's commit lands on a SAVEPOINT
    inside the fixture's transaction, so the data is visible to assertions but
    the outer rollback still wipes it. Without it the test would either see no
    data (never flushed) or leak between tests (really committed).
    """
    connection = await engine.connect()  # type: ignore[attr-defined]
    transaction = await connection.begin()
    maker = async_sessionmaker(
        bind=connection,
        expire_on_commit=False,
        autoflush=False,
        join_transaction_mode="create_savepoint",
    )
    session = maker()

    try:
        yield session
    finally:
        await session.close()
        await transaction.rollback()
        await connection.close()


@pytest.fixture(autouse=True)
def fake_redis() -> AsyncIterator[None]:
    from fakeredis import aioredis

    set_redis(aioredis.FakeRedis(decode_responses=True))
    yield
    set_redis(None)


@pytest.fixture(autouse=True)
def outbox() -> AsyncIterator[email_integration.ConsoleSender]:
    """Captures every email so a test can read a verification link out of it
    without a mail server."""
    sender = email_integration.ConsoleSender()
    email_integration.set_sender(sender)
    yield sender
    email_integration.set_sender(None)


@pytest_asyncio.fixture(loop_scope="session")
async def client(db: AsyncSession) -> AsyncIterator[AsyncClient]:
    async def override() -> AsyncIterator[AsyncSession]:
        # Mirrors production get_session: commit on success, roll back on
        # error. Exercising the same commit path is the point — a test that
        # skips it would not catch a handler that forgets to persist.
        try:
            yield db
        except Exception:
            await db.rollback()
            raise
        else:
            await db.commit()

    app.dependency_overrides[get_session] = override
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
        follow_redirects=False,
    ) as http:
        yield http
    app.dependency_overrides.clear()


# ── Helpers ─────────────────────────────────────────────────────────────────

GOOD_PASSWORD = "correct-horse-battery-staple-42"


@pytest.fixture
def register_payload() -> dict[str, str]:
    return {
        "email": "ada@example.com",
        "password": GOOD_PASSWORD,
        "name": "Ada Lovelace",
    }


async def register_and_verify(
    client: AsyncClient,
    db: AsyncSession,
    email: str = "ada@example.com",
    password: str = GOOD_PASSWORD,
) -> str:
    """Register, verify, and return the user id."""
    from sqlalchemy import select

    from app.models.user import User

    await client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": password, "name": "Ada Lovelace"},
    )
    user = (await db.execute(select(User).where(User.email == email))).scalar_one()
    from app.core.database import utcnow

    user.email_verified_at = utcnow()
    await db.flush()
    return user.id
