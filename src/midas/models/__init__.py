"""Data models for midas: Entity, Source, Deal, EvidenceSpan.

Importing this package also registers all tables on
``SQLModel.metadata`` so :func:`SQLModel.metadata.create_all` and Alembic
autogenerate see them.

Validation contract
-------------------
SQLModel ``table=True`` classes intentionally skip pydantic validation in
``__init__`` so the ORM can hydrate partial rows. That means a direct
``Deal(...)`` call **does not** enforce ``Field`` constraints (``ge``,
``le``, required-ness, type coercion).

Always validate untrusted input — anything coming from an LLM extractor,
HTTP request, or YAML registry — through
:meth:`SQLModel.model_validate`::

    deal = Deal.model_validate(extracted_dict)   # raises ValidationError

Direct construction is reserved for code that has already validated its
inputs (tests, repository internals).
"""

from .deal import Deal
from .discovered_source import DiscoveredSource
from .entity import Entity
from .evidence import EvidenceSpan
from .source import Source
from .types import DealStatus, DealType, EntityType, SourceType

__all__ = [
    "Deal",
    "DealStatus",
    "DealType",
    "DiscoveredSource",
    "Entity",
    "EntityType",
    "EvidenceSpan",
    "Source",
    "SourceType",
]
