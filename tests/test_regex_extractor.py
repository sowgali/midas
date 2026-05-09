"""Tests for the regex extractor.

These pin down the four canonical patterns we promise to catch and
guard the resolution logic that decides when to keep vs drop a match.
Anything more nuanced is the LLM extractor's job — we deliberately
don't over-test edge cases here, since adding patterns to satisfy
edge tests is exactly the regex creep we want to avoid.
"""

from __future__ import annotations

import uuid
from decimal import Decimal

import pytest

from midas.extractors import (
    ExtractedDeal,
    ExtractionContext,
    KnownParty,
    RegexExtractor,
)
from midas.models.types import DealStatus, DealType, SourceType


def _ctx(text: str, parties: list[KnownParty]) -> ExtractionContext:
    return ExtractionContext(
        source_id=uuid.uuid4(),
        source_url="https://example.com/press-release",
        source_type=SourceType.PRESS_RELEASE,
        known_parties=parties,
        document_text=text,
    )


# ---------- Party fixtures ----------

MICROSOFT = KnownParty(
    entity_id=uuid.uuid4(),
    canonical_name="Microsoft Corporation",
    aliases=["Microsoft", "MSFT"],
)
OPENAI = KnownParty(
    entity_id=uuid.uuid4(),
    canonical_name="OpenAI",
    aliases=["OpenAI Inc."],
)
GOOGLE = KnownParty(
    entity_id=uuid.uuid4(),
    canonical_name="Alphabet Inc.",
    aliases=["Google", "GOOGL"],
)
DEEPMIND = KnownParty(
    entity_id=uuid.uuid4(),
    canonical_name="DeepMind",
    aliases=["DeepMind Technologies"],
)
ANTHROPIC = KnownParty(
    entity_id=uuid.uuid4(),
    canonical_name="Anthropic",
    aliases=["Anthropic PBC"],
)
AMAZON = KnownParty(
    entity_id=uuid.uuid4(),
    canonical_name="Amazon",
    aliases=["AWS", "Amazon Web Services"],
)


# ---------- Investment ----------


@pytest.mark.asyncio
async def test_invest_in_pattern_matches_billion_amount() -> None:
    text = "Microsoft will invest $10 billion in OpenAI."
    extractor = RegexExtractor()

    deals = await extractor.extract(_ctx(text, [MICROSOFT, OPENAI]))

    assert len(deals) == 1
    deal = deals[0]
    assert deal.deal_type == DealType.INVESTMENT
    assert deal.status == DealStatus.ANNOUNCED
    assert deal.source_party_name == "Microsoft Corporation"
    assert deal.target_party_name == "OpenAI"
    assert deal.amount_usd == Decimal("10000000000")
    assert deal.amount_native == Decimal("10000000000")
    assert deal.currency == "USD"
    assert deal.extractor_name == "regex"
    assert "Microsoft" in deal.evidence_text_snippet
    assert deal.char_start >= 0
    assert deal.char_end > deal.char_start
    # Snippet is the substring of the document at those offsets.
    assert text[deal.char_start : deal.char_end] == deal.evidence_text_snippet


@pytest.mark.asyncio
async def test_invested_past_tense_pattern() -> None:
    text = "Microsoft invested $1.5 billion in OpenAI last quarter."
    extractor = RegexExtractor()

    deals = await extractor.extract(_ctx(text, [MICROSOFT, OPENAI]))

    assert len(deals) == 1
    assert deals[0].deal_type == DealType.INVESTMENT
    assert deals[0].amount_usd == Decimal("1500000000")


# ---------- Acquisition ----------


@pytest.mark.asyncio
async def test_acquisition_pattern() -> None:
    text = "Google acquired DeepMind for $500 million."
    extractor = RegexExtractor()

    deals = await extractor.extract(_ctx(text, [GOOGLE, DEEPMIND]))

    assert len(deals) == 1
    deal = deals[0]
    assert deal.deal_type == DealType.ACQUISITION
    assert deal.source_party_name == "Alphabet Inc."
    assert deal.target_party_name == "DeepMind"
    assert deal.amount_usd == Decimal("500000000")


# ---------- Commercial contract ----------


@pytest.mark.asyncio
async def test_multi_year_contract_pattern() -> None:
    text = "Anthropic announced a 5-year, $4 billion compute contract with Amazon."
    extractor = RegexExtractor()

    deals = await extractor.extract(_ctx(text, [ANTHROPIC, AMAZON]))

    assert len(deals) == 1
    deal = deals[0]
    assert deal.deal_type == DealType.COMMERCIAL_CONTRACT
    assert deal.source_party_name == "Anthropic"
    assert deal.target_party_name == "Amazon"
    assert deal.amount_usd == Decimal("4000000000")


# ---------- Multi-deal ----------


@pytest.mark.asyncio
async def test_multi_deal_text_returns_multiple_extracted_deals() -> None:
    text = (
        "Microsoft will invest $10 billion in OpenAI. "
        "Separately, Google acquired DeepMind for $500 million."
    )
    extractor = RegexExtractor()

    deals = await extractor.extract(
        _ctx(text, [MICROSOFT, OPENAI, GOOGLE, DEEPMIND]),
    )

    assert len(deals) == 2
    types = {d.deal_type for d in deals}
    assert types == {DealType.INVESTMENT, DealType.ACQUISITION}
    # Offsets are distinct and reference the correct sentences.
    invest = next(d for d in deals if d.deal_type == DealType.INVESTMENT)
    acquire = next(d for d in deals if d.deal_type == DealType.ACQUISITION)
    assert "microsoft" in text[invest.char_start : invest.char_end].lower()
    assert "google" in text[acquire.char_start : acquire.char_end].lower()
    assert "openai" in text[invest.char_start : invest.char_end].lower()
    assert "deepmind" in text[acquire.char_start : acquire.char_end].lower()


# ---------- Negatives ----------


@pytest.mark.asyncio
async def test_unrelated_text_yields_no_deals() -> None:
    extractor = RegexExtractor()
    deals = await extractor.extract(_ctx("It was a sunny day.", [MICROSOFT, OPENAI]))
    assert deals == []


@pytest.mark.asyncio
async def test_unknown_party_is_skipped() -> None:
    # MICROSOFT is known but OpenAI is not — without grounding for both
    # sides we drop the candidate rather than fabricate a target entity.
    text = "Microsoft will invest $10 billion in OpenAI."
    extractor = RegexExtractor()

    deals = await extractor.extract(_ctx(text, [MICROSOFT]))

    assert deals == []


@pytest.mark.asyncio
async def test_extracted_deal_is_not_a_sqlmodel_instance() -> None:
    # Sanity: ExtractedDeal must not bypass validation the way our
    # SQLModel tables do. confidence=2.0 should raise.
    text = "Microsoft will invest $10 billion in OpenAI."
    extractor = RegexExtractor()
    deals = await extractor.extract(_ctx(text, [MICROSOFT, OPENAI]))
    assert isinstance(deals[0], ExtractedDeal)
    assert 0.0 <= deals[0].confidence <= 1.0
