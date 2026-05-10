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


# ---------- Pure chunking helper ----------


def test_split_chunks_preserves_total_coverage() -> None:
    """Every character of the input must appear in at least one chunk
    after concatenation (with overlap allowed)."""
    from midas.extractors.claude import _split_into_chunks

    text = "Microsoft will invest $10 billion in OpenAI.\n\n" * 30_000
    chunks = _split_into_chunks(text)
    assert len(chunks) >= 2
    # The last chunk reaches the end of the document.
    last_text, last_offset = chunks[-1]
    assert last_offset + len(last_text) == len(text)
    # Doc offsets are monotonically increasing.
    offsets = [o for _, o in chunks]
    assert offsets == sorted(offsets)


def test_split_chunks_returns_single_chunk_for_small_doc() -> None:
    from midas.extractors.claude import _split_into_chunks

    text = "Microsoft will invest $10B in OpenAI."
    assert _split_into_chunks(text) == [(text, 0)]


def test_split_chunks_overlap_creates_redundancy_at_boundary() -> None:
    """Successive chunks share ``overlap_chars`` of common content so a
    sentence straddling the boundary appears in both."""
    from midas.extractors.claude import _split_into_chunks

    text = "x" * 1_200_000
    chunks = _split_into_chunks(text)
    assert len(chunks) >= 2
    # Second chunk starts inside the first chunk's span.
    first_text, first_offset = chunks[0]
    _, second_offset = chunks[1]
    assert second_offset < first_offset + len(first_text)


def test_split_chunks_breaks_at_paragraph_when_possible() -> None:
    """When a paragraph break is within the tail half of the chunk
    window, the split happens there instead of mid-sentence.
    """
    from midas.extractors.claude import _CHUNK_CHARS, _split_into_chunks

    # Build a doc with a clear paragraph break in the tail half of the
    # first chunk window. After 400 KB of filler + "\n\n" + more filler,
    # we expect the first chunk's end to align with the break.
    head = "a" * (_CHUNK_CHARS - 50_000)  # well inside the first chunk
    middle_break_pos = len(head)
    tail = "b" * 1_000_000
    text = head + "\n\n" + tail
    chunks = _split_into_chunks(text)

    first_text, _ = chunks[0]
    # First chunk should end at the paragraph break (+ the included
    # blank-line bytes), within +/- 5 KB of the break position.
    assert abs(len(first_text) - (middle_break_pos + 2)) < 5_000


def test_split_chunks_caps_at_max_chunks() -> None:
    """Pathologically huge docs stop chunking at the safety cap."""
    from midas.extractors.claude import (
        _CHUNK_CHARS,
        _MAX_CHUNKS_PER_DOC,
        _split_into_chunks,
    )

    # Big enough to exceed the cap.
    text = "x" * (_CHUNK_CHARS * (_MAX_CHUNKS_PER_DOC + 4))
    chunks = _split_into_chunks(text)
    assert len(chunks) == _MAX_CHUNKS_PER_DOC


# ---------- Oversize doc + bad-request handling (V1.9.3 fixes) ----------


@pytest.mark.asyncio
async def test_oversize_document_is_chunked_and_multipassed() -> None:
    """Documents past one chunk's worth go through multi-pass extraction.

    Each chunk gets its own API call; no transactional info is dropped.
    """
    call_count = {"n": 0}
    captured_messages: list[str] = []
    client = AsyncMock()

    async def fake_create(**kwargs: object) -> SimpleNamespace:
        call_count["n"] += 1
        captured_messages.append(kwargs["messages"][0]["content"])  # type: ignore[index]
        return SimpleNamespace(content=[])

    client.messages.create = fake_create
    extractor = ClaudeExtractor(client=client)

    # ~3 MB doc → 7+ chunks at the default 500 KB chunk size.
    huge_text = "Microsoft will invest $10 billion in OpenAI. " * 70_000
    assert len(huge_text) > 2_000_000

    await extractor.extract(_ctx(text=huge_text))

    # Multi-pass: multiple API calls, one per chunk.
    assert call_count["n"] > 1
    # Every chunk got the "chunk N of M" disclosure.
    for msg in captured_messages:
        assert "chunk " in msg and " of " in msg


@pytest.mark.asyncio
async def test_chunk_offsets_translate_back_to_document_coordinates() -> None:
    """Char offsets emitted by the model are relative to the chunk; the
    extractor must add the chunk's doc_offset before returning so each
    ExtractedDeal points at the canonical document text.
    """
    from midas.extractors.claude import _split_into_chunks

    # Pick a doc large enough to force 2 chunks.
    repeat = "Microsoft will invest $10 billion in OpenAI. " * 30_000
    chunks = _split_into_chunks(repeat)
    assert len(chunks) >= 2

    # The model will be told to extract; we make it return a tool_use
    # with chunk-relative offsets (0..44) on each chunk. After
    # translation, deals from chunk[1] should have offsets >= chunks[1]'s
    # doc_offset.
    chunk_index = {"i": 0}

    async def fake_create(**kwargs: object) -> SimpleNamespace:
        i = chunk_index["i"]
        chunk_index["i"] += 1
        payload = {
            "source_party_name": "Microsoft Corporation",
            "target_party_name": "OpenAI",
            "deal_type": "investment",
            "status": "announced",
            "amount_usd": 1.0e10,
            "currency": "USD",
            "confidence": 0.95,
            "description": f"chunk {i}",
            "evidence_text_snippet": "Microsoft will invest $10 billion in OpenAI.",
            "char_start": 0,
            "char_end": 44,
        }
        return SimpleNamespace(content=[_tool_use_block("record_deal", payload)])

    client = AsyncMock()
    client.messages.create = fake_create
    extractor = ClaudeExtractor(client=client)

    deals = await extractor.extract(_ctx(text=repeat))

    # After dedup the (Microsoft, OpenAI, $10B, investment, announced)
    # tuple collapses to one row even though every chunk returned it.
    # First-seen wins, so the surviving deal carries chunk-0's
    # (translated) offsets = (0, 44).
    assert len(deals) == 1
    assert deals[0].char_start == 0
    assert deals[0].char_end == 44


@pytest.mark.asyncio
async def test_dedup_collapses_duplicates_from_overlap_region() -> None:
    """A deal that lands inside the overlap between two chunks gets
    surfaced twice by extraction. The post-pass dedup must collapse it.
    """
    async def fake_create(**kwargs: object) -> SimpleNamespace:
        msg = kwargs["messages"][0]["content"]  # type: ignore[index]
        # Always returns the same deal, regardless of which chunk we're on.
        payload = {
            "source_party_name": "Microsoft",
            "target_party_name": "OpenAI",
            "deal_type": "investment",
            "amount_usd": 1.0e10,
            "currency": "USD",
            "confidence": 0.9,
            "description": "Microsoft -> OpenAI $10B",
            "evidence_text_snippet": "$10 billion",
            "char_start": 50,
            "char_end": 61,
        }
        _ = msg
        return SimpleNamespace(content=[_tool_use_block("record_deal", payload)])

    client = AsyncMock()
    client.messages.create = fake_create
    extractor = ClaudeExtractor(client=client)

    huge = "x" * 1_500_000
    deals = await extractor.extract(_ctx(text=huge))

    # Exactly one deal survives — the overlap-introduced duplicates
    # are collapsed in dedup.
    assert len(deals) == 1


@pytest.mark.asyncio
async def test_single_pass_when_document_fits_in_one_chunk() -> None:
    """Backward compat: docs that fit in one chunk make one API call,
    no chunk-note in the prompt, no dedup pass overhead.
    """
    captured: dict[str, object] = {}
    call_count = {"n": 0}

    async def fake_create(**kwargs: object) -> SimpleNamespace:
        call_count["n"] += 1
        captured.update(kwargs)
        return SimpleNamespace(content=[])

    client = AsyncMock()
    client.messages.create = fake_create
    extractor = ClaudeExtractor(client=client)

    # Normal-size doc.
    await extractor.extract(_ctx(text="Microsoft will invest $10B in OpenAI."))
    assert call_count["n"] == 1
    msg = captured["messages"][0]["content"]  # type: ignore[index]
    # No multi-chunk disclosure on single-pass.
    assert "chunk " not in msg


@pytest.mark.asyncio
async def test_chunk_bad_request_doesnt_kill_other_chunks() -> None:
    """If one chunk raises BadRequestError, the others still contribute."""
    import anthropic

    state = {"i": 0}

    async def fake_create(**kwargs: object) -> SimpleNamespace:
        i = state["i"]
        state["i"] += 1
        if i == 0:
            # First chunk: API rejects.
            raise anthropic.BadRequestError(
                "prompt is too long",
                response=SimpleNamespace(  # type: ignore[arg-type]
                    status_code=400, headers={}, request=SimpleNamespace(),
                ),
                body=None,
            )
        # Later chunks: return a real deal.
        payload = {
            "source_party_name": "Anthropic",
            "target_party_name": "Amazon.com, Inc.",
            "deal_type": "commercial_contract",
            "amount_usd": 1.0e11,
            "currency": "USD",
            "confidence": 0.95,
            "description": "Anthropic -> AWS $100B",
            "evidence_text_snippet": "$100 billion",
            "char_start": 10,
            "char_end": 21,
        }
        return SimpleNamespace(content=[_tool_use_block("record_deal", payload)])

    client = AsyncMock()
    client.messages.create = fake_create
    extractor = ClaudeExtractor(client=client)

    deals = await extractor.extract(_ctx(text="z" * 1_500_000))

    # First chunk crashed but the remaining chunks all returned the
    # same deal; dedup collapses to one.
    assert len(deals) == 1
    assert deals[0].source_party_name == "Anthropic"


@pytest.mark.asyncio
async def test_bad_request_error_returns_empty_instead_of_crashing() -> None:
    """If the API rejects the prompt (token cap, schema drift, etc.),
    the extractor logs and returns []; the surrounding pipeline keeps
    processing other documents.
    """
    import anthropic

    client = AsyncMock()
    # Build a plausible BadRequestError. The SDK's constructor wants a
    # message + a response + a body; SimpleNamespace stand-ins are fine
    # because the extractor only str()s the exception.
    fake_response = SimpleNamespace(
        status_code=400,
        headers={},
        request=SimpleNamespace(),
    )
    err = anthropic.BadRequestError(
        "prompt is too long: 3043747 tokens > 1000000 maximum",
        response=fake_response,  # type: ignore[arg-type]
        body=None,
    )
    client.messages.create = AsyncMock(side_effect=err)
    extractor = ClaudeExtractor(client=client)

    # Doesn't raise — returns no deals so the pipeline moves on.
    deals = await extractor.extract(_ctx(text="anything"))
    assert deals == []
