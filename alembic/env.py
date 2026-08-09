from __future__ import annotations

import asyncio
from logging.config import fileConfig

from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from alembic import context
from app.core.config import settings
from app.core.database import Base

# Importing the package registers every model on Base.metadata. Without it,
# autogenerate produces a migration that drops tables it cannot see.
import app.models  # noqa: F401  isort:skip

config = context.config

# The URL is NOT written into the ini config. configparser applies `%`
# interpolation to every value it stores, so a URL-encoded password — `%40`
# for `@`, `%23` for `#` — raises before the engine is ever built. Passing it
# straight to the engine keeps the raw string intact.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _configure(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        # Both on: without them a column changing type or default is silently
        # absent from the diff, and the schema drifts from the models.
        compare_type=True,
        compare_server_default=True,
        render_as_batch=False,
    )


def run_migrations_offline() -> None:
    context.configure(
        url=settings.database_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def _run(connection: Connection) -> None:
    _configure(connection)
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    engine = async_engine_from_config(
        {
            **config.get_section(config.config_ini_section, {}),
            "sqlalchemy.url": settings.database_url,
        },
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    async with engine.connect() as connection:
        await connection.run_sync(_run)
    await engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
