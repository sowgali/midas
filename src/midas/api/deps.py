"""FastAPI dependency providers.

The engine is built once at app startup (see :func:`midas.api.app.create_app`'s
``lifespan``) and stored on ``app.state``; :func:`get_session` opens one
session per request from a process-wide ``async_sessionmaker``.

Tests override ``get_session`` directly via
``app.dependency_overrides[get_session]`` rather than touching the engine
state, so the override surface stays small.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker


async def get_session(request: Request) -> AsyncIterator[AsyncSession]:
    """Yield one :class:`AsyncSession` per request, scoped to the request lifecycle.

    The ``async_sessionmaker`` is attached to ``app.state.session_factory``
    by the lifespan handler. We open a session, hand it to the route, and
    close it when the route returns.
    """
    factory: async_sessionmaker[AsyncSession] = request.app.state.session_factory
    async with factory() as session:
        yield session


# Tidy alias so route handlers can write ``session: Session`` instead of
# repeating the full ``Annotated[AsyncSession, Depends(get_session)]``.
Session = Annotated[AsyncSession, Depends(get_session)]
