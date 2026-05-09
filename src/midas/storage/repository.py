"""Async repositories — one per aggregate root.

Each repository wraps an :class:`AsyncSession` and offers a small,
purpose-built API rather than a generic CRUD surface. Callers pass
already-validated SQLModel instances; repositories don't re-validate (see
the validation contract in ``midas.models.__init__``).

Repositories ``add`` rows but **do not** commit — the unit-of-work
boundary belongs to the caller, who decides when a logical operation is
done. Use ``await session.commit()`` (or rely on a context-managed
transaction) at the call site.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from datetime import date

from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col, select

from midas.models import Deal, Entity, EvidenceSpan, Source


class EntityRepository:
    """Read/write access to :class:`Entity` rows."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def add(self, entity: Entity) -> Entity:
        self._session.add(entity)
        await self._session.flush()
        return entity

    async def get(self, entity_id: uuid.UUID) -> Entity | None:
        return await self._session.get(Entity, entity_id)

    async def get_by_canonical_name(self, name: str) -> Entity | None:
        stmt = select(Entity).where(col(Entity.canonical_name) == name)
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_by_ticker(self, ticker: str) -> Entity | None:
        stmt = select(Entity).where(col(Entity.ticker) == ticker)
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def list_by_sector(self, tag: str) -> Sequence[Entity]:
        # JSON ``contains`` is dialect-specific; portable approach is to
        # pull candidates and filter in Python. The seed registry is small
        # (~50 entities) so this is fine for V1.
        stmt = select(Entity)
        result = await self._session.execute(stmt)
        return [e for e in result.scalars().all() if tag in e.sector_tags]


class SourceRepository:
    """Read/write access to :class:`Source` rows.

    ``content_sha256`` is the dedup key — the same document fetched twice
    collapses to one row via :meth:`upsert`.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def add(self, source: Source) -> Source:
        self._session.add(source)
        await self._session.flush()
        return source

    async def get_by_content_sha256(self, content_sha256: str) -> Source | None:
        stmt = select(Source).where(col(Source.content_sha256) == content_sha256)
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def upsert(self, source: Source) -> Source:
        """Return the existing row for this hash, or insert ``source`` and return it."""
        existing = await self.get_by_content_sha256(source.content_sha256)
        if existing is not None:
            return existing
        return await self.add(source)


class DealRepository:
    """Read/write access to :class:`Deal` rows."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def add(self, deal: Deal) -> Deal:
        self._session.add(deal)
        await self._session.flush()
        return deal

    async def list_outgoing(
        self,
        entity_id: uuid.UUID,
        as_of: date | None = None,
    ) -> Sequence[Deal]:
        """Deals where ``entity_id`` is the payer.

        When ``as_of`` is provided, restrict to deals already announced by
        that date (``announced_at <= as_of``); deals with no
        ``announced_at`` are excluded since we can't place them in time.
        """
        stmt = select(Deal).where(col(Deal.from_entity_id) == entity_id)
        if as_of is not None:
            stmt = stmt.where(col(Deal.announced_at).is_not(None)).where(
                col(Deal.announced_at) <= as_of,
            )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def list_incoming(
        self,
        entity_id: uuid.UUID,
        as_of: date | None = None,
    ) -> Sequence[Deal]:
        """Deals where ``entity_id`` is the recipient. See :meth:`list_outgoing`."""
        stmt = select(Deal).where(col(Deal.to_entity_id) == entity_id)
        if as_of is not None:
            stmt = stmt.where(col(Deal.announced_at).is_not(None)).where(
                col(Deal.announced_at) <= as_of,
            )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())


class EvidenceRepository:
    """Bulk-insert :class:`EvidenceSpan` rows for a freshly-extracted deal."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def add_many(self, spans: Sequence[EvidenceSpan]) -> Sequence[EvidenceSpan]:
        self._session.add_all(list(spans))
        await self._session.flush()
        return list(spans)
