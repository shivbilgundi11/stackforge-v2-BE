"""Async SQLAlchemy engine, session lifecycle, declarative base."""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime

from sqlalchemy import DateTime, MetaData, func
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from uuid_extensions import uuid7

from app.core.config import settings

# Explicit constraint naming. Without this, Alembic autogenerate produces
# machine-named indexes that differ between environments and make a migration
# diff unreadable.
NAMING_CONVENTION = {
    "ix": "ix_%(table_name)s_%(column_0_N_name)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class SoftDeleteMixin:
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)


def new_id(prefix: str) -> str:
    """Prefixed UUIDv7.

    Time-ordered, so index locality is good and cursor pagination can sort on
    the primary key. The prefix means an id in a log line or a support ticket
    identifies its own table without a lookup.
    """
    return f"{prefix}_{uuid7(as_type='str').replace('-', '')}"


engine: AsyncEngine = create_async_engine(
    settings.database_url,
    echo=settings.database_echo,
    pool_size=settings.database_pool_size,
    max_overflow=settings.database_max_overflow,
    pool_pre_ping=True,
    pool_recycle=1800,
)

SessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)


async def get_session() -> AsyncIterator[AsyncSession]:
    """Request-scoped session: commit on success, roll back on exception.

    Routers never commit. A handler that raises leaves nothing half-written.
    """
    async with SessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        else:
            await session.commit()


async def check_database() -> bool:
    """Readiness probe. Never raises — reports."""
    from sqlalchemy import text

    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
    except Exception:
        return False
    return True


async def dispose_engine() -> None:
    await engine.dispose()


def utcnow() -> datetime:
    """Timezone-aware now.

    Every timestamp in this codebase is aware. A naive datetime compared
    against an aware one raises, and finding that at runtime is worse than
    typing this.
    """
    return datetime.now(UTC)


__all__ = [
    "Base",
    "SessionLocal",
    "SoftDeleteMixin",
    "TimestampMixin",
    "check_database",
    "dispose_engine",
    "engine",
    "get_session",
    "new_id",
    "utcnow",
]
