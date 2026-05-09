"""Async SQLAlchemy engine and session plumbing.

The engine is constructed lazily — call :func:`make_engine` once at
process start (typically from the CLI or pipeline entrypoint) and pass
the resulting :class:`AsyncEngine` into :func:`make_session_factory`.
The :func:`session` context manager is a convenience for one-off scripts
and tests; long-running services should hold onto a session factory and
open sessions per unit of work.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from midas.config import settings


def make_engine(database_url: str | None = None) -> AsyncEngine:
    """Build an :class:`AsyncEngine`, defaulting to ``settings.database_url``."""
    url = database_url if database_url is not None else settings.database_url
    return create_async_engine(url, future=True)


def make_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """Return a session factory bound to ``engine``.

    ``expire_on_commit=False`` keeps already-loaded attributes accessible
    after ``commit()`` — important when callers return ORM instances out
    of the session scope.
    """
    return async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


@asynccontextmanager
async def session(engine: AsyncEngine | None = None) -> AsyncIterator[AsyncSession]:
    """Yield a single :class:`AsyncSession`, disposing the engine if we created it.

    If ``engine`` is omitted a fresh one is built from settings and
    disposed on exit. Pass an existing engine in long-lived processes to
    reuse its connection pool.
    """
    owns_engine = engine is None
    eng = engine if engine is not None else make_engine()
    factory = make_session_factory(eng)
    try:
        async with factory() as s:
            yield s
    finally:
        if owns_engine:
            await eng.dispose()
