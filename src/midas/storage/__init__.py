"""Async persistence layer: engine, session, repositories.

The package exposes a small public surface — an engine factory, a session
context manager, and one repository class per aggregate. Anything that
talks to the database from elsewhere in the codebase should go through
these types rather than building its own session.
"""

from __future__ import annotations

from .db import make_engine, make_session_factory, session
from .repository import (
    DealRepository,
    EntityRepository,
    EvidenceRepository,
    SourceRepository,
)

__all__ = [
    "DealRepository",
    "EntityRepository",
    "EvidenceRepository",
    "SourceRepository",
    "make_engine",
    "make_session_factory",
    "session",
]
