"""Tests for the Claude (Anthropic) LLM extractor.

Every test injects a mocked ``anthropic.AsyncAnthropic`` client so we
never make real API calls. The tests pin down:

* the happy path — tool_use blocks parse into ExtractedDeals
* the multi-call path — multiple tool_use blocks all surface
* the no-claim path — text-only responses return []
* the missing-API-key path — RuntimeError with the right message
* the validation path — bad tool_use is logged + skipped, valid
  siblings in the same response still surface
* the default-status path — model omitting status defaults to ANNOUNCED
"""

from __future__ import annotations

import uuid
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from midas.extractors import ClaudeExtractor, ExtractionContext, KnownParty
from midas.models.types import SourceType


def _ctx(text: str = "Microsoft will invest $10 billion in OpenAI.") -> ExtractionContext:
    return ExtractionContext(
        source_id=uuid.uuid4(),
        source_url="https://example.com/pr",
        source_type=SourceType.PRESS_RELEASE,
        known_parties=[
            KnownParty(
                entity_id=uuid.uuid4(),
                canonical_name="Microsoft Corporation",
                aliases=["Microsoft"],
            ),
            KnownParty(
                entity_id=uuid.uuid4(),
                canonical_name="OpenAI",
                aliases=[],
            ),
        ],
        document_text=text,
    )


def _tool_use_block(name: str, payload: dict[str, object]) -> SimpleNamespace:
    """Mimic an ``anthropic.types.ToolUseBlock``.

    The extractor only touches ``.type``, ``.name``, and ``.input`` so a
    SimpleNamespace stands in cleanly without pulling in anthropic's
    internal types or mocking out the whole pydantic class.
    """
    return SimpleNamespace(type="tool_use", name=name, input=payload)


def _text_block(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


def _mock_client_returning(content: list[SimpleNamespace]) -> AsyncMock:
    client = AsyncMock()
    response = SimpleNamespace(content=content)
    client.messages.create = AsyncMock(return_value=response)
    return client


# ---------- Happy paths ----------


@pytest.mark.asyncio
async def test_single_tool_use_block_yields_one_extracted_deal() -> None:
    payload = {
        "source_party_name": "Microsoft Corporation",
        "target_party_name": "OpenAI",
        "deal_type": "investment",
        "status": "announced",
        "amount_usd": 10000000000,
        "amount_native": 10000000000,
        "currency": "USD",
        "announced_at": None,
        "closes_at": None,
        "confidence": 0.92,
        "description": "Microsoft is investing $10B in OpenAI.",
        "evidence_text_snippet": "Microsoft will invest $10 billion in OpenAI.",
        "char_start": 0,
        "char_end": 44,
    }
    client = _mock_client_returning([_tool_use_block("record_deal", payload)])
    extractor = ClaudeExtractor(client=client)

    deals = await extractor.extract(_ctx())

    assert len(deals) == 1
    deal = deals[0]
    assert deal.source_party_name == "Microsoft Corporation"
    assert deal.target_party_name == "OpenAI"
    assert deal.amount_usd == Decimal("10000000000")
    assert deal.confidence == pytest.approx(0.92)
    # Extractor stamps its own name; the model isn't asked to.
    assert deal.extractor_name == "claude:opus-4-7"

    # System prompt and tool definition both carry cache_control.
    call_kwargs = client.messages.create.call_args.kwargs
    assert call_kwargs["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert call_kwargs["tools"][0]["cache_control"] == {"type": "ephemeral"}
    assert call_kwargs["model"] == "claude-opus-4-7"


@pytest.mark.asyncio
async def test_two_tool_use_blocks_yield_two_extracted_deals() -> None:
    payload_a = {
        "source_party_name": "Microsoft Corporation",
        "target_party_name": "OpenAI",
        "deal_type": "investment",
        "status": "announced",
        "amount_usd": 10000000000,
        "amount_native": None,
        "currency": "USD",
        "announced_at": None,
        "closes_at": None,
        "confidence": 0.9,
        "description": "Microsoft → OpenAI investment",
        "evidence_text_snippet": "Microsoft will invest $10 billion in OpenAI.",
        "char_start": 0,
        "char_end": 44,
    }
    payload_b = {
        "source_party_name": "Alphabet Inc.",
        "target_party_name": "DeepMind",
        "deal_type": "acquisition",
        "status": "closed",
        "amount_usd": 500000000,
        "amount_native": None,
        "currency": "USD",
        "announced_at": None,
        "closes_at": None,
        "confidence": 0.95,
        "description": "Google acquired DeepMind",
        "evidence_text_snippet": "Google acquired DeepMind for $500 million.",
        "char_start": 50,
        "char_end": 92,
    }
    client = _mock_client_returning(
        [
            _text_block("I found two deals."),
            _tool_use_block("record_deal", payload_a),
            _tool_use_block("record_deal", payload_b),
        ]
    )
    extractor = ClaudeExtractor(client=client)

    deals = await extractor.extract(_ctx())

    assert len(deals) == 2
    assert {d.target_party_name for d in deals} == {"OpenAI", "DeepMind"}


@pytest.mark.asyncio
async def test_text_only_response_yields_no_deals() -> None:
    client = _mock_client_returning([_text_block("This document doesn't mention any deals.")])
    extractor = ClaudeExtractor(client=client)

    deals = await extractor.extract(_ctx("It was a sunny day."))

    assert deals == []
    client.messages.create.assert_awaited_once()


# ---------- Configuration / error paths ----------


@pytest.mark.asyncio
async def test_missing_api_key_with_no_injected_client_raises() -> None:
    # Force settings.anthropic_api_key to None for this test.
    from midas import config as cfg

    extractor = ClaudeExtractor()  # no injected client
    original = cfg.settings.anthropic_api_key
    cfg.settings.anthropic_api_key = None
    try:
        with pytest.raises(RuntimeError, match="MIDAS_ANTHROPIC_API_KEY not set"):
            await extractor.extract(_ctx())
    finally:
        cfg.settings.anthropic_api_key = original


@pytest.mark.asyncio
async def test_invalid_tool_input_logged_and_skipped() -> None:
    # Decision: per-tool_use validation failures don't tank the whole
    # document. The bad block is dropped (and logged); valid neighbors
    # in the same response still come through.
    bad_payload = {
        "source_party_name": "Microsoft Corporation",
        "target_party_name": "OpenAI",
        "deal_type": "investment",
        "status": "announced",
        "amount_usd": 10000000000,
        "amount_native": None,
        "currency": "USD",
        "announced_at": None,
        "closes_at": None,
        "confidence": 2.0,  # out of range — pydantic should reject
        "description": "Microsoft → OpenAI",
        "evidence_text_snippet": "Microsoft will invest $10 billion in OpenAI.",
        "char_start": 0,
        "char_end": 44,
    }
    good_payload = {
        "source_party_name": "Anthropic",
        "target_party_name": "Amazon.com, Inc.",
        "deal_type": "investment",
        "status": "announced",
        "amount_usd": 4000000000,
        "amount_native": None,
        "currency": "USD",
        "announced_at": None,
        "closes_at": None,
        "confidence": 0.95,
        "description": "Amazon invests in Anthropic",
        "evidence_text_snippet": "Amazon to invest up to $4 billion in Anthropic.",
        "char_start": 100,
        "char_end": 146,
    }
    client = _mock_client_returning(
        [
            _tool_use_block("record_deal", bad_payload),
            _tool_use_block("record_deal", good_payload),
        ],
    )
    extractor = ClaudeExtractor(client=client)

    deals = await extractor.extract(_ctx())
    assert len(deals) == 1
    assert deals[0].source_party_name == "Anthropic"


@pytest.mark.asyncio
async def test_status_defaults_when_omitted_by_model() -> None:
    """Real-world Claude responses sometimes omit ``status`` — the model
    defaults to ``ANNOUNCED`` (matches the dominant 8-K case).
    """
    payload_no_status = {
        "source_party_name": "Microsoft Corporation",
        "target_party_name": "OpenAI",
        "deal_type": "investment",
        "amount_usd": 10000000000,
        "currency": "USD",
        "confidence": 0.9,
        "description": "Microsoft → OpenAI",
        "evidence_text_snippet": "Microsoft invests $10B in OpenAI.",
        "char_start": 0,
        "char_end": 33,
    }
    client = _mock_client_returning([_tool_use_block("record_deal", payload_no_status)])
    extractor = ClaudeExtractor(client=client)
    deals = await extractor.extract(_ctx())
    assert len(deals) == 1
    assert deals[0].status.value == "announced"


@pytest.mark.asyncio
async def test_unrecognized_tool_name_is_ignored() -> None:
    # If the model invents a tool we didn't define, drop it on the
    # floor. record_deal is the only tool we accept.
    client = _mock_client_returning(
        [
            _tool_use_block("invent_a_tool", {"foo": "bar"}),
            _text_block("done"),
        ]
    )
    extractor = ClaudeExtractor(client=client)

    deals = await extractor.extract(_ctx())

    assert deals == []
