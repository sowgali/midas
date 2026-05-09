"""Validation and round-trip tests for the data models.

These tests are pure pydantic / SQLAlchemy — no live DB required. They
guard:

* default factories (UUIDs, timestamps, empty list defaults),
* enum coercion (StrEnum acceptance and rejection of bogus values),
* numeric typing (Decimal, not float, for money),
* range constraints (``confidence`` clamped to [0, 1]),
* metadata registration (every model is on ``SQLModel.metadata`` and
  the schema can be materialized against an in-memory SQLite engine).

End-to-end Postgres tests live alongside the migrations in a later slice.
"""

from __future__ import annotations

import uuid
from datetime import UTC, date, datetime
from decimal import Decimal

import pytest
from pydantic import ValidationError
from sqlalchemy import create_engine
from sqlmodel import SQLModel

from midas.models import (
    Deal,
    DealStatus,
    DealType,
    Entity,
    EntityType,
    EvidenceSpan,
    Source,
    SourceType,
)

# ---------- Entity ----------


def test_entity_defaults_uuid_and_empty_lists() -> None:
    e = Entity(canonical_name="Alphabet Inc.", entity_type=EntityType.PUBLIC_COMPANY)
    assert isinstance(e.id, uuid.UUID)
    assert e.aliases == []
    assert e.sector_tags == []
    assert e.created_at.tzinfo is not None


def test_entity_aliases_round_trip() -> None:
    e = Entity(
        canonical_name="Alphabet Inc.",
        aliases=["Google", "GOOG", "GOOGL"],
        ticker="GOOGL",
        cik="0001652044",
        entity_type=EntityType.PUBLIC_COMPANY,
        sector_tags=["ai", "cloud"],
        country="US",
    )
    assert "Google" in e.aliases
    assert e.entity_type == EntityType.PUBLIC_COMPANY


def test_entity_rejects_unknown_entity_type() -> None:
    # Untrusted input goes through model_validate; SQLModel table=True skips
    # validation in __init__, so direct construction is intentionally lax.
    with pytest.raises(ValidationError):
        Entity.model_validate(
            {"canonical_name": "X", "entity_type": "not_a_type"},
        )


# ---------- Source ----------


def test_source_requires_content_hash() -> None:
    with pytest.raises(ValidationError):
        Source.model_validate(
            {
                "url": "https://example.com/x",
                "source_type": SourceType.PRESS_RELEASE,
                "publisher": "Example Co.",
                # content_sha256 omitted on purpose
            },
        )


def test_source_accepts_form_codes() -> None:
    s = Source(
        url="https://www.sec.gov/Archives/edgar/data/0/0.htm",
        source_type=SourceType.FORM_10K,
        publisher="SEC",
        content_sha256="a" * 64,
    )
    assert s.source_type == SourceType.FORM_10K
    assert s.fetched_at.tzinfo is not None


# ---------- Deal ----------


def _make_deal(**overrides: object) -> Deal:
    base: dict[str, object] = {
        "from_entity_id": uuid.uuid4(),
        "to_entity_id": uuid.uuid4(),
        "deal_type": DealType.INVESTMENT,
        "status": DealStatus.ANNOUNCED,
        "confidence": 0.9,
        "description": "Test deal.",
    }
    base.update(overrides)
    return Deal(**base)


def test_deal_is_directional() -> None:
    src, dst = uuid.uuid4(), uuid.uuid4()
    d = _make_deal(from_entity_id=src, to_entity_id=dst)
    assert d.from_entity_id == src
    assert d.to_entity_id == dst
    assert d.from_entity_id != d.to_entity_id


def test_deal_amount_is_decimal() -> None:
    d = _make_deal(amount_usd=Decimal("1000000.50"))
    assert isinstance(d.amount_usd, Decimal)
    assert d.amount_usd == Decimal("1000000.50")


def test_deal_amount_accepts_string_for_decimal_safety() -> None:
    # Strings are the safe way to construct Decimals; model_validate coerces.
    d = Deal.model_validate(
        {
            "from_entity_id": uuid.uuid4(),
            "to_entity_id": uuid.uuid4(),
            "deal_type": DealType.INVESTMENT,
            "status": DealStatus.ANNOUNCED,
            "confidence": 0.9,
            "description": "Test deal.",
            "amount_usd": "999999999999.99",
        },
    )
    assert isinstance(d.amount_usd, Decimal)
    assert d.amount_usd == Decimal("999999999999.99")


def test_deal_confidence_clamped_to_unit_interval() -> None:
    base = {
        "from_entity_id": uuid.uuid4(),
        "to_entity_id": uuid.uuid4(),
        "deal_type": DealType.INVESTMENT,
        "status": DealStatus.ANNOUNCED,
        "description": "Test deal.",
    }
    with pytest.raises(ValidationError):
        Deal.model_validate({**base, "confidence": 1.5})
    with pytest.raises(ValidationError):
        Deal.model_validate({**base, "confidence": -0.1})


def test_deal_temporal_fields() -> None:
    d = _make_deal(
        announced_at=date(2025, 9, 15),
        closes_at=date(2025, 12, 31),
    )
    assert d.announced_at == date(2025, 9, 15)
    assert d.closes_at == date(2025, 12, 31)


# ---------- EvidenceSpan ----------


def test_evidence_span_links_deal_and_source() -> None:
    deal_id, source_id = uuid.uuid4(), uuid.uuid4()
    span = EvidenceSpan(
        deal_id=deal_id,
        source_id=source_id,
        text_snippet="Microsoft will invest an additional $10 billion in OpenAI.",
        char_start=120,
        char_end=180,
        extractor="claude:opus-4-7",
    )
    assert span.deal_id == deal_id
    assert span.source_id == source_id
    assert "OpenAI" in span.text_snippet


def test_evidence_span_rejects_negative_offsets() -> None:
    with pytest.raises(ValidationError):
        EvidenceSpan.model_validate(
            {
                "deal_id": uuid.uuid4(),
                "source_id": uuid.uuid4(),
                "text_snippet": "x",
                "char_start": -1,
                "char_end": 10,
                "extractor": "regex",
            },
        )


# ---------- Schema sanity ----------


def test_all_tables_registered_on_metadata() -> None:
    expected = {"entity", "source", "deal", "evidence_span"}
    assert expected.issubset(SQLModel.metadata.tables.keys())


def test_schema_materializes_against_sqlite() -> None:
    """Smoke test: the full schema can be created on a generic backend.

    Doesn't validate Postgres-specific behavior (JSONB, partial indexes),
    just that nothing in our column declarations is illegal SQL.
    """
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)
    inspector_tables = SQLModel.metadata.tables
    assert {"entity", "source", "deal", "evidence_span"}.issubset(inspector_tables.keys())


def test_datetime_defaults_are_utc_aware() -> None:
    e = Entity(canonical_name="X Inc.", entity_type=EntityType.PRIVATE_COMPANY)
    s = Source(
        url="https://example.com",
        source_type=SourceType.NEWS,
        publisher="Example",
        content_sha256="b" * 64,
    )
    assert e.created_at.tzinfo == UTC
    assert s.fetched_at.tzinfo == UTC
    # Round-trip with ISO format keeps tz info.
    iso = e.created_at.isoformat()
    assert datetime.fromisoformat(iso).tzinfo is not None
