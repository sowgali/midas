"""HTTP API package — read-only FastAPI surface for the React frontend.

Exposes :func:`create_app` (the FastAPI factory) and the module-level
``app`` instance so ``uvicorn midas.api.app:app`` works out of the box.
"""

from __future__ import annotations

from .app import app, create_app

__all__ = ["app", "create_app"]
