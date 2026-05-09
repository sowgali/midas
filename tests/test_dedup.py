"""Unit tests for the V1.6 dedup match policy + merge primitives.

These are pure-function tests against the policy. The end-to-end
behavior (pipeline reconciliation; Wiz-lifecycle collapse) is covered
in ``tests/test_pipeline.py``.
"""

from __future__ import annotations

import uuid
from datetime import UTC, date
from decimal import Decimal

from midas.dedup import (
    AMOUNT_TOLERANCE,
    DATE_WINDOW_DAYS,
    amount_compatible,
    apply_merge,
    deals_match,
    merge_amount,
    merge_announced_at,
    merge_closes_at,
    merge_description,
    merge_status,
    temporal_compatible,
)
from midas.extractors.base import ExtractedDeal
from midas.models import Deal, DealStatus, DealType

# ---------- temporal_compatible ----------


def test_temporal_compatible_handles_none_on_either_side() -> None:
    assert temporal_compatible(None, date(2025, 1, 1))
    assert temporal_compatible(date(2025, 1, 1), None)
    assert temporal_compatible(None, None)


def test_temporal_compatible_inside_window() -> None:
    assert temporal_compatible(date(2025, 1, 1), date(2025, 1, 1))
    assert temporal_compatible(date(2025, 1, 1), date(2025, 3, 31))  # 89d


def test_temporal_compatible_outside_window() -> None:
    assert not temporal_compatible(date(2025, 1, 1), date(2025, 5, 1))  # 120d


def test_temporal_compatible_window_param() -> None:
    assert not temporal_compatible(date(2025, 1, 1), date(2025, 2, 1), window_days=15)


def test_temporal_window_default_is_90() -> None:
    assert DATE_WINDOW_DAYS == 90


# ---------- amount_compatible ----------


def test_amount_compatible_handles_none() -> None:
    assert amount_compatible(None, Decimal("32_000_000_000"))
    assert amount_compatible(Decimal("32_000_000_000"), None)
    assert amount_compatible(None, None)


def test_amount_compatible_exact_match() -> None:
    assert amount_compatible(Decimal("100"), Decimal("100"))


def test_amount_compatible_within_default_tolerance() -> None:
    # Wiz: $32B announced -> $29.5B closed (7.8% drop)
    assert amount_compatible(Decimal("32_000_000_000"), Decimal("29_500_000_000"))


def test_amount_compatible_outside_default_tolerance() -> None:
    # $16B vs $5.6B (Waymo case) — 65% gap, definitely different deals.
    assert not amount_compatible(Decimal("16_000_000_000"), Decimal("5_600_000_000"))


def test_amount_tolerance_default_is_15pct() -> None:
    assert AMOUNT_TOLERANCE == 0.15


def test_amount_compatible_exactly_on_boundary() -> None:
    # 15.0% exact — accept (≤).
    assert amount_compatible(Decimal("100"), Decimal("85"))
    # 16% — reject.
    assert not amount_compatible(Decimal("100"), Decimal("84"))


# ---------- deals_match ----------


def _existing(
    *,
    from_id: uuid.UUID,
    to_id: uuid.UUID,
    deal_type: DealType = DealType.ACQUISITION,
    announced_at: date | None = date(2025, 3, 1),
    amount: Decimal | None = Decimal("32_000_000_000"),
    status: DealStatus = DealStatus.ANNOUNCED,
) -> Deal:
    return Deal(
        from_entity_id=from_id,
        to_entity_id=to_id,
        deal_type=deal_type,
        announced_at=announced_at,
        amount_usd=amount,
        status=status,
        confidence=0.9,
        description="x",
    )


def test_deals_match_full_positive() -> None:
    f, t = uuid.uuid4(), uuid.uuid4()
    existing = _existing(from_id=f, to_id=t)
    assert deals_match(
        existing,
        from_entity_id=f,
        to_entity_id=t,
        deal_type=DealType.ACQUISITION,
        announced_at=date(2025, 3, 1),
        amount_usd=Decimal("32_000_000_000"),
    )


def test_deals_dont_match_when_party_differs() -> None:
    f, t = uuid.uuid4(), uuid.uuid4()
    existing = _existing(from_id=f, to_id=t)
    other = uuid.uuid4()
    assert not deals_match(
        existing,
        from_entity_id=other,
        to_entity_id=t,
        deal_type=DealType.ACQUISITION,
        announced_at=date(2025, 3, 1),
        amount_usd=Decimal("32_000_000_000"),
    )


def test_deals_dont_match_when_deal_type_differs() -> None:
    f, t = uuid.uuid4(), uuid.uuid4()
    existing = _existing(from_id=f, to_id=t, deal_type=DealType.INVESTMENT)
    assert not deals_match(
        existing,
        from_entity_id=f,
        to_entity_id=t,
        deal_type=DealType.ACQUISITION,
        announced_at=date(2025, 3, 1),
        amount_usd=Decimal("32_000_000_000"),
    )


def test_deals_match_with_one_null_announced_date() -> None:
    """The Wiz close-row case: closed disclosure has no announced_at."""
    f, t = uuid.uuid4(), uuid.uuid4()
    existing = _existing(from_id=f, to_id=t, amount=Decimal("32_000_000_000"))
    assert deals_match(
        existing,
        from_entity_id=f,
        to_entity_id=t,
        deal_type=DealType.ACQUISITION,
        announced_at=None,
        amount_usd=Decimal("29_500_000_000"),  # post-close adjustment, 7.8% gap
    )


# ---------- merge_status ----------


def test_merge_status_progression() -> None:
    assert merge_status(DealStatus.RUMORED, DealStatus.ANNOUNCED) == DealStatus.ANNOUNCED
    assert merge_status(DealStatus.ANNOUNCED, DealStatus.CLOSED) == DealStatus.CLOSED
    assert merge_status(DealStatus.RUMORED, DealStatus.CLOSED) == DealStatus.CLOSED


def test_merge_status_terminated_is_authoritative() -> None:
    assert merge_status(DealStatus.CLOSED, DealStatus.TERMINATED) == DealStatus.TERMINATED
    assert merge_status(DealStatus.TERMINATED, DealStatus.CLOSED) == DealStatus.TERMINATED
    assert merge_status(DealStatus.RUMORED, DealStatus.TERMINATED) == DealStatus.TERMINATED


def test_merge_status_idempotent() -> None:
    for s in DealStatus:
        assert merge_status(s, s) == s


# ---------- merge_amount ----------


def test_merge_amount_closed_wins() -> None:
    """Post-close adjustments are the most accurate amount."""
    out = merge_amount(
        Decimal("32_000_000_000"),
        DealStatus.ANNOUNCED,
        Decimal("29_500_000_000"),
        DealStatus.CLOSED,
    )
    assert out == Decimal("29_500_000_000")


def test_merge_amount_existing_closed_beats_new_announced() -> None:
    out = merge_amount(
        Decimal("29_500_000_000"),
        DealStatus.CLOSED,
        Decimal("32_000_000_000"),
        DealStatus.ANNOUNCED,
    )
    assert out == Decimal("29_500_000_000")


def test_merge_amount_neither_closed_keeps_larger() -> None:
    out = merge_amount(
        Decimal("100"),
        DealStatus.ANNOUNCED,
        Decimal("110"),
        DealStatus.ANNOUNCED,
    )
    assert out == Decimal("110")


def test_merge_amount_handles_none_sides() -> None:
    assert merge_amount(
        None, DealStatus.ANNOUNCED, Decimal("100"), DealStatus.ANNOUNCED
    ) == Decimal("100")
    assert merge_amount(
        Decimal("100"), DealStatus.ANNOUNCED, None, DealStatus.ANNOUNCED
    ) == Decimal("100")
    assert merge_amount(None, DealStatus.ANNOUNCED, None, DealStatus.ANNOUNCED) is None


# ---------- date merges ----------


def test_merge_announced_at_takes_earliest() -> None:
    assert merge_announced_at(date(2025, 3, 1), date(2025, 4, 1)) == date(2025, 3, 1)
    assert merge_announced_at(None, date(2025, 4, 1)) == date(2025, 4, 1)
    assert merge_announced_at(date(2025, 3, 1), None) == date(2025, 3, 1)
    assert merge_announced_at(None, None) is None


def test_merge_closes_at_takes_latest() -> None:
    assert merge_closes_at(date(2026, 3, 11), date(2026, 4, 1)) == date(2026, 4, 1)
    assert merge_closes_at(None, date(2026, 4, 1)) == date(2026, 4, 1)
    assert merge_closes_at(date(2026, 3, 11), None) == date(2026, 3, 11)


# ---------- description ----------


def test_merge_description_longer_wins() -> None:
    assert merge_description("short", "a much longer string with more detail") == (
        "a much longer string with more detail"
    )
    # Equal length: existing wins (deterministic tiebreak).
    assert merge_description("aaaa", "bbbb") == "aaaa"


# ---------- apply_merge integration ----------


def _candidate(
    *,
    deal_type: DealType = DealType.ACQUISITION,
    status: DealStatus = DealStatus.CLOSED,
    amount_usd: Decimal | None = Decimal("29_500_000_000"),
    announced_at: date | None = None,
    closes_at: date | None = date(2026, 3, 11),
    confidence: float = 0.98,
    description: str = "Alphabet completed acquisition of Wiz for $29.5 billion.",
) -> ExtractedDeal:
    return ExtractedDeal(
        source_party_name="Alphabet Inc.",
        target_party_name="Wiz, Inc.",
        deal_type=deal_type,
        status=status,
        amount_usd=amount_usd,
        announced_at=announced_at,
        closes_at=closes_at,
        confidence=confidence,
        description=description,
        evidence_text_snippet="...",
        char_start=0,
        char_end=10,
        extractor_name="test",
    )


def test_apply_merge_wiz_close_into_announce() -> None:
    """End-to-end shape of the Wiz close absorbing the announce row."""
    f, t = uuid.uuid4(), uuid.uuid4()
    existing = _existing(
        from_id=f,
        to_id=t,
        announced_at=date(2025, 3, 1),
        amount=Decimal("32_000_000_000"),
        status=DealStatus.ANNOUNCED,
    )
    existing.confidence = 0.97
    existing.description = (
        "Alphabet entered into agreement to acquire Wiz, a cloud security platform"
    )
    pre_updated = existing.updated_at

    apply_merge(existing, _candidate())

    assert existing.status == DealStatus.CLOSED
    assert existing.amount_usd == Decimal("29_500_000_000")
    assert existing.announced_at == date(2025, 3, 1)  # earliest preserved
    assert existing.closes_at == date(2026, 3, 11)
    assert existing.confidence == 0.98  # max
    assert existing.updated_at >= pre_updated
    assert existing.updated_at.tzinfo == UTC
    # description: existing was longer, keeps existing.
    assert "cloud security" in existing.description


def test_apply_merge_does_not_overwrite_with_lesser_amount() -> None:
    """Re-ingesting the same announce row shouldn't lower confidence/amount."""
    f, t = uuid.uuid4(), uuid.uuid4()
    existing = _existing(
        from_id=f,
        to_id=t,
        amount=Decimal("32_000_000_000"),
        status=DealStatus.ANNOUNCED,
    )
    existing.confidence = 0.95

    apply_merge(
        existing,
        _candidate(
            status=DealStatus.ANNOUNCED,
            amount_usd=Decimal("32_000_000_000"),
            announced_at=date(2025, 3, 1),
            closes_at=None,
            confidence=0.90,
            description="short",
        ),
    )

    assert existing.amount_usd == Decimal("32_000_000_000")
    assert existing.status == DealStatus.ANNOUNCED
    assert existing.confidence == 0.95  # max preserved
