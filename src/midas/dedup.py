"""V1.6 cross-source deal deduplication.

The pipeline can extract the same real-world transaction from multiple
sources (a 10-K announcement + the close-stage 10-Q + N quarterly
"subsequent events" disclosures), and pre-V1.6 each extraction created
a fresh :class:`midas.models.Deal` row. This module is the
match-and-merge logic that collapses them at ingest time so the schema
matches the design promise: **1 Deal = 1 real-world transaction; N
EvidenceSpans = N filings that disclosed it.**

Match policy ("same deal" predicate)
------------------------------------
A new candidate matches an existing :class:`Deal` row when **all** of:

1. ``from_entity_id`` matches.
2. ``to_entity_id`` matches.
3. ``deal_type`` matches.
4. *Temporal compatibility* — ``announced_at`` within
   :data:`DATE_WINDOW_DAYS` (default 90), OR one side is None.
5. *Amount compatibility* — relative difference of ``amount_usd`` ≤
   :data:`AMOUNT_TOLERANCE` (default 15%), OR one side is None.

The temporal window accommodates the typical announce → close lifecycle
(quarterly filings between are within 90 days). The amount tolerance
accommodates post-close adjustments (Wiz: $32.0B announced → $29.5B at
close = 7.8% drop, comfortably inside).

Field-merge rules when a match fires
------------------------------------
=========================  ==============================================
``status``                 ``TERMINATED`` is authoritative if either side
                           has it; otherwise CLOSED > ANNOUNCED > RUMORED.
``amount_usd``             A new ``CLOSED`` amount wins (post-adjustment
                           is more accurate); else the larger known amount.
``amount_native``          Same rule as ``amount_usd``.
``currency``               Existing wins if set; else take new.
``announced_at``           Earliest non-None wins (first announcement is
                           "the" date).
``closes_at``              When both are set, latest wins (later filings
                           know more); else whichever is set.
``confidence``             Max.
``description``            Longer wins.
``updated_at``             Stamped to ``now(UTC)``.
=========================  ==============================================

Always-append: every reconciliation produces an :class:`EvidenceSpan`
attached to the surviving Deal row, regardless of whether the candidate
was a match or a fresh insert.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal
from typing import TYPE_CHECKING

from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col, select

from midas.models import Deal, DealStatus

if TYPE_CHECKING:
    import uuid

    from midas.extractors.base import ExtractedDeal


DATE_WINDOW_DAYS: int = 90
AMOUNT_TOLERANCE: float = 0.15

_STATUS_RANK: dict[DealStatus, int] = {
    DealStatus.RUMORED: 0,
    DealStatus.ANNOUNCED: 1,
    DealStatus.CLOSED: 2,
}


# ---------- predicates ----------


def temporal_compatible(
    a: date | None, b: date | None, window_days: int = DATE_WINDOW_DAYS
) -> bool:
    if a is None or b is None:
        return True
    return abs((a - b).days) <= window_days


def amount_compatible(
    a: Decimal | None, b: Decimal | None, tolerance: float = AMOUNT_TOLERANCE
) -> bool:
    if a is None or b is None:
        return True
    if a == b:
        return True
    denom = max(abs(a), abs(b))
    if denom == 0:
        return True
    return abs(a - b) / denom <= Decimal(str(tolerance))


def deals_match(
    existing: Deal,
    *,
    from_entity_id: uuid.UUID,
    to_entity_id: uuid.UUID,
    deal_type: str,
    announced_at: date | None,
    amount_usd: Decimal | None,
    window_days: int = DATE_WINDOW_DAYS,
    tolerance: float = AMOUNT_TOLERANCE,
) -> bool:
    """Pure predicate — no DB. Used by both the live path and the
    backfill reconciler so they share one definition of "same deal".
    """
    if existing.from_entity_id != from_entity_id:
        return False
    if existing.to_entity_id != to_entity_id:
        return False
    if existing.deal_type != deal_type:
        return False
    if not temporal_compatible(existing.announced_at, announced_at, window_days):
        return False
    return amount_compatible(existing.amount_usd, amount_usd, tolerance)


# ---------- match lookup ----------


async def find_matching_deal(
    session: AsyncSession,
    *,
    from_entity_id: uuid.UUID,
    to_entity_id: uuid.UUID,
    deal_type: str,
    announced_at: date | None,
    amount_usd: Decimal | None,
) -> Deal | None:
    """Find a Deal row in the DB that this candidate should absorb.

    When several rows match, the highest-confidence one wins. Caller
    is expected to log a warning at the multiple-match site.
    """
    stmt = select(Deal).where(
        col(Deal.from_entity_id) == from_entity_id,
        col(Deal.to_entity_id) == to_entity_id,
        col(Deal.deal_type) == deal_type,
    )
    result = await session.execute(stmt)
    candidates = list(result.scalars().all())

    matches = [
        d
        for d in candidates
        if temporal_compatible(d.announced_at, announced_at)
        and amount_compatible(d.amount_usd, amount_usd)
    ]
    if not matches:
        return None
    matches.sort(key=lambda d: d.confidence, reverse=True)
    return matches[0]


# ---------- field-level merge primitives ----------


def merge_status(existing: DealStatus, candidate: DealStatus) -> DealStatus:
    """``TERMINATED`` is authoritative; else most-progressed wins."""
    if existing == DealStatus.TERMINATED or candidate == DealStatus.TERMINATED:
        return DealStatus.TERMINATED
    return max((existing, candidate), key=lambda s: _STATUS_RANK[s])


def merge_amount(
    existing_amount: Decimal | None,
    existing_status: DealStatus,
    candidate_amount: Decimal | None,
    candidate_status: DealStatus,
) -> Decimal | None:
    """Closed amounts win (they're post-adjustment)."""
    if candidate_status == DealStatus.CLOSED and candidate_amount is not None:
        return candidate_amount
    if existing_status == DealStatus.CLOSED and existing_amount is not None:
        return existing_amount
    if existing_amount is None:
        return candidate_amount
    if candidate_amount is None:
        return existing_amount
    return max(existing_amount, candidate_amount)


def merge_announced_at(a: date | None, b: date | None) -> date | None:
    """Earliest known announcement wins."""
    if a is None:
        return b
    if b is None:
        return a
    return min(a, b)


def merge_closes_at(a: date | None, b: date | None) -> date | None:
    """Latest known close date wins (later filings disclose more)."""
    if a is None:
        return b
    if b is None:
        return a
    return max(a, b)


def merge_description(a: str, b: str) -> str:
    return a if len(a) >= len(b) else b


# ---------- top-level merge ----------


def apply_merge(existing: Deal, candidate: ExtractedDeal) -> None:
    """Mutate ``existing`` in place to absorb ``candidate``.

    Does NOT touch the DB session, does NOT add the EvidenceSpan
    (caller owns those — keeps this function easy to test).
    """
    existing.amount_usd = merge_amount(
        existing.amount_usd,
        existing.status,
        candidate.amount_usd,
        candidate.status,
    )
    existing.amount_native = merge_amount(
        existing.amount_native,
        existing.status,
        candidate.amount_native,
        candidate.status,
    )
    if existing.currency is None:
        existing.currency = candidate.currency
    existing.announced_at = merge_announced_at(existing.announced_at, candidate.announced_at)
    existing.closes_at = merge_closes_at(existing.closes_at, candidate.closes_at)
    existing.status = merge_status(existing.status, candidate.status)
    existing.confidence = max(existing.confidence, candidate.confidence)
    existing.description = merge_description(existing.description, candidate.description)
    existing.updated_at = datetime.now(UTC)


def merge_duplicate_into(canonical: Deal, duplicate: Deal) -> None:
    """Absorb ``duplicate`` Deal row into ``canonical``, in place.

    Same field-merge rules as :func:`apply_merge` but operating on two
    persisted :class:`Deal` rows instead of (Deal, ExtractedDeal). Used
    by ``midas reconcile`` to collapse rows that pre-date V1.6 dedup.
    Caller is responsible for migrating EvidenceSpans + deleting
    ``duplicate``.
    """
    canonical.amount_usd = merge_amount(
        canonical.amount_usd,
        canonical.status,
        duplicate.amount_usd,
        duplicate.status,
    )
    canonical.amount_native = merge_amount(
        canonical.amount_native,
        canonical.status,
        duplicate.amount_native,
        duplicate.status,
    )
    if canonical.currency is None:
        canonical.currency = duplicate.currency
    canonical.announced_at = merge_announced_at(canonical.announced_at, duplicate.announced_at)
    canonical.closes_at = merge_closes_at(canonical.closes_at, duplicate.closes_at)
    canonical.status = merge_status(canonical.status, duplicate.status)
    canonical.confidence = max(canonical.confidence, duplicate.confidence)
    canonical.description = merge_description(canonical.description, duplicate.description)
    canonical.updated_at = datetime.now(UTC)
