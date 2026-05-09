"""FastAPI factory + module-level ``app``.

The app owns one :class:`AsyncEngine` for its whole lifetime: it's built
in the ``lifespan`` ``startup`` step and disposed at shutdown. Each
request gets its own ``AsyncSession`` from a process-wide
``async_sessionmaker`` (see :mod:`midas.api.deps`).

CORS is opened up to the two dev-server origins we expect a React
frontend to use locally — Vite's default (5173) and CRA's default
(3000). Production origins should be added explicitly when the
deployment story exists.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from midas import __version__
from midas.storage.db import make_engine, make_session_factory

from .routers import deals as deals_router
from .routers import entities as entities_router
from .routers import graph as graph_router

CORS_ORIGINS: tuple[str, ...] = (
    "http://localhost:5173",
    "http://localhost:3000",
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Build the engine + session factory once; dispose at shutdown."""
    engine = make_engine()
    app.state.engine = engine
    app.state.session_factory = make_session_factory(engine)
    try:
        yield
    finally:
        await engine.dispose()


def create_app() -> FastAPI:
    """Construct the FastAPI app and wire routers + middleware + lifespan."""
    app = FastAPI(
        title="midas",
        version=__version__,
        description="Read-only HTTP API over the midas cash-flow graph.",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(CORS_ORIGINS),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health", tags=["meta"])
    async def health() -> dict[str, str]:
        """Liveness probe — also echoes the installed package version."""
        return {"status": "ok", "version": __version__}

    app.include_router(entities_router.router, prefix="/api")
    app.include_router(deals_router.router, prefix="/api")
    app.include_router(graph_router.router, prefix="/api")
    return app


app = create_app()
