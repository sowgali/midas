"""Tests for the V1.9.4 batch-mode Claude extractor.

The Anthropic Message Batches API is stateful (submit → poll →
results) so the mock here implements all three sides via a small
fake-batches helper. No real API calls are ever made.

Coverage:

* submit + poll loop reaches ``ended`` and stops
* per-request results route back to the right context via custom_id
* multi-chunk docs translate offsets correctly and dedupe duplicates
* per-request failures don't kill other contexts' results
* a wall-clock cap fires when a batch never ends
* extract([]) returns [] without calling the API
* extract(single) wraps extract_many transparently
"""

from __future__ import annotations

import asyncio
import uuid
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from midas.extractors import BatchClaudeExtractor, ExtractionContext, KnownParty
from midas.extractors.claude_batch import _decode_custom_id, _encode_custom_id
from midas.models.types import SourceType

# ---------- Fixtures ----------


def _ctx(text: str = "Microsoft will invest $10 billion in OpenAI.") -> ExtractionContext:
    return ExtractionContext(
        source_id=uuid.uuid4(),
        source_url=f"https://example.com/{uuid.uuid4()}",
        source_type=SourceType.PRESS_RELEASE,
        known_parties=[
            KnownParty(entity_id=uuid.uuid4(), canonical_name="Microsoft Corporation"),
            KnownParty(entity_id=uuid.uuid4(), canonical_name="OpenAI"),
        ],
        document_text=text,
    )


def _tool_use_block(name: str, payload: dict[str, object]) -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", name=name, input=payload)


def _payload(
    src: str = "Microsoft Corporation",
    tgt: str = "OpenAI",
    amount: float | None = 1.0e10,
) -> dict[str, object]:
    return {
        "source_party_name": src,
        "target_party_name": tgt,
        "deal_type": "investment",
        "status": "announced",
        "amount_usd": amount,
        "currency": "USD",
        "confidence": 0.92,
        "description": f"{src} -> {tgt}",
        "evidence_text_snippet": "Microsoft will invest $10 billion in OpenAI.",
        "char_start": 0,
        "char_end": 44,
    }


def _result_row(
    custom_id: str,
    *,
    result_type: str = "succeeded",
    content_blocks: list[Any] | None = None,
) -> SimpleNamespace:
    """Mimic one row in Anthropic's batch-results stream."""
    if content_blocks is None:
        content_blocks = []
    inner_message = SimpleNamespace(content=content_blocks)
    inner = SimpleNamespace(type=result_type, message=inner_message)
    return SimpleNamespace(custom_id=custom_id, result=inner)


class _AsyncResultIterator:
    """Minimal async-iterable over a list of result rows.

    ``client.messages.batches.results(batch_id)`` is expected to be
    awaited and the awaited value is an async iterator; this matches.
    """

    def __init__(self, rows: list[Any]) -> None:
        self._rows = rows

    def __await__(self) -> Any:
        async def _passthrough() -> _AsyncResultIterator:
            return self

        return _passthrough().__await__()

    def __aiter__(self) -> _AsyncResultIterator:
        return self

    async def __anext__(self) -> Any:
        if not self._rows:
            raise StopAsyncIteration
        return self._rows.pop(0)


def _mock_client(
    *,
    submit_id: str = "batch-test-001",
    poll_transitions: list[str] | None = None,
    results: list[Any] | None = None,
) -> AsyncMock:
    """Build a mock anthropic client that pretends to run a batch.

    ``poll_transitions`` lists the sequence of ``processing_status``
    values that ``retrieve`` will return on successive calls. End the
    list with ``"ended"`` so the poll loop terminates.
    """
    poll_transitions = poll_transitions or ["ended"]
    poll_iter = iter(poll_transitions)

    client = AsyncMock()

    async def fake_create(*, requests: list[Any]) -> SimpleNamespace:
        # Counters used by the test to assert what was sent.
        client._submitted_requests = requests  # type: ignore[attr-defined]
        return SimpleNamespace(id=submit_id)

    async def fake_retrieve(batch_id: str) -> SimpleNamespace:
        try:
            status = next(poll_iter)
        except StopIteration:
            status = "ended"
        return SimpleNamespace(
            id=batch_id,
            processing_status=status,
            request_counts=SimpleNamespace(
                succeeded=1, errored=0, processing=0, canceled=0, expired=0,
            ),
        )

    def fake_results(batch_id: str) -> _AsyncResultIterator:
        return _AsyncResultIterator(list(results or []))

    client.messages.batches.create = fake_create
    client.messages.batches.retrieve = fake_retrieve
    client.messages.batches.results = fake_results
    return client


# ---------- custom_id codec ----------


def test_custom_id_round_trip() -> None:
    assert _decode_custom_id(_encode_custom_id(0, 0)) == (0, 0)
    assert _decode_custom_id(_encode_custom_id(42, 7)) == (42, 7)
    assert _decode_custom_id(_encode_custom_id(99999, 999)) == (99999, 999)


def test_custom_id_raises_on_malformed() -> None:
    with pytest.raises(ValueError):
        _decode_custom_id("not-a-real-id")


# ---------- happy paths ----------


@pytest.mark.asyncio
async def test_empty_contexts_returns_empty_without_api_call() -> None:
    """Passing [] short-circuits — no batch submission, no poll, no results."""
    submit_called = {"n": 0}

    async def fake_create(*, requests: list[Any]) -> SimpleNamespace:
        submit_called["n"] += 1
        return SimpleNamespace(id="should-not-happen")

    client = AsyncMock()
    client.messages.batches.create = fake_create
    extractor = BatchClaudeExtractor(client=client, poll_initial_s=0.0)
    assert await extractor.extract_many([]) == []
    assert submit_called["n"] == 0


@pytest.mark.asyncio
async def test_single_context_routes_back_correctly() -> None:
    """One context, one chunk, one deal — the simplest batch path."""
    client = _mock_client(
        results=[
            _result_row(
                _encode_custom_id(0, 0),
                content_blocks=[_tool_use_block("record_deal", _payload())],
            ),
        ],
    )
    extractor = BatchClaudeExtractor(client=client, poll_initial_s=0.0)
    results = await extractor.extract_many([_ctx()])
    assert len(results) == 1
    assert len(results[0]) == 1
    assert results[0][0].source_party_name == "Microsoft Corporation"
    # Extractor stamped its (batch) name onto the deal.
    assert results[0][0].extractor_name == "claude:opus-4-7:batch"
    # And the API actually got one request submitted.
    assert len(client._submitted_requests) == 1  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_multiple_contexts_partition_results_by_custom_id() -> None:
    """N contexts with one chunk each → N results with distinct custom_ids."""
    contexts = [_ctx(), _ctx(), _ctx()]
    client = _mock_client(
        results=[
            # Reversed order — the extractor must route by custom_id, not insertion order.
            _result_row(
                _encode_custom_id(2, 0),
                content_blocks=[_tool_use_block(
                    "record_deal",
                    _payload(src="Anthropic", tgt="Amazon.com, Inc."),
                )],
            ),
            _result_row(
                _encode_custom_id(0, 0),
                content_blocks=[_tool_use_block(
                    "record_deal",
                    _payload(src="Microsoft Corporation", tgt="OpenAI"),
                )],
            ),
            _result_row(
                _encode_custom_id(1, 0),
                content_blocks=[_tool_use_block(
                    "record_deal",
                    _payload(src="Alphabet Inc.", tgt="DeepMind"),
                )],
            ),
        ],
    )
    extractor = BatchClaudeExtractor(client=client, poll_initial_s=0.0)
    results = await extractor.extract_many(contexts)
    assert len(results) == 3
    assert results[0][0].source_party_name == "Microsoft Corporation"
    assert results[1][0].source_party_name == "Alphabet Inc."
    assert results[2][0].source_party_name == "Anthropic"


# ---------- chunking interaction ----------


@pytest.mark.asyncio
async def test_chunked_context_dispatches_one_request_per_chunk() -> None:
    """A document large enough to chunk produces multiple batch requests."""
    huge = "Microsoft will invest $10 billion in OpenAI. " * 30_000
    client = _mock_client(results=[])  # don't care about results here
    extractor = BatchClaudeExtractor(client=client, poll_initial_s=0.0)
    await extractor.extract_many([_ctx(text=huge)])
    submitted = client._submitted_requests  # type: ignore[attr-defined]
    assert len(submitted) > 1
    # Every request points back to context 0 with an increasing chunk index.
    ctx_chunks = [_decode_custom_id(r["custom_id"]) for r in submitted]
    assert all(c == 0 for c, _ in ctx_chunks)
    chunk_indices = [k for _, k in ctx_chunks]
    assert chunk_indices == list(range(len(ctx_chunks)))


@pytest.mark.asyncio
async def test_chunked_context_dedupes_overlap_duplicates() -> None:
    """Same deal extracted from multiple chunks of one doc collapses to one."""
    huge = "Microsoft will invest $10 billion in OpenAI. " * 30_000
    ctx = _ctx(text=huge)
    # Pre-mocked results: every chunk surfaces the same deal.
    # We don't know how many chunks until extract_many splits, so we
    # cheat by counting the requests after submit and producing N
    # identical results. That requires inverting control: pre-build a
    # large pool of results and let _AsyncResultIterator drain what it
    # needs. The extractor walks results until StopAsyncIteration.
    same_payload = _payload()
    pool = [
        _result_row(
            _encode_custom_id(0, i),
            content_blocks=[_tool_use_block("record_deal", same_payload)],
        )
        for i in range(20)  # plenty of slots
    ]
    client = _mock_client(results=pool)
    extractor = BatchClaudeExtractor(client=client, poll_initial_s=0.0)
    results = await extractor.extract_many([ctx])
    assert len(results) == 1
    # Despite 20 mocked results, the (src, tgt, type, amount, status)
    # dedup key collapses all to one ExtractedDeal.
    assert len(results[0]) == 1


# ---------- failure isolation ----------


@pytest.mark.asyncio
async def test_per_request_errored_result_isolated_to_its_context() -> None:
    """A failed request on context #0 doesn't affect contexts #1 / #2."""
    contexts = [_ctx(), _ctx(), _ctx()]
    client = _mock_client(
        results=[
            _result_row(_encode_custom_id(0, 0), result_type="errored"),
            _result_row(
                _encode_custom_id(1, 0),
                content_blocks=[_tool_use_block(
                    "record_deal", _payload(src="Anthropic", tgt="AWS"),
                )],
            ),
            _result_row(
                _encode_custom_id(2, 0),
                content_blocks=[_tool_use_block(
                    "record_deal", _payload(src="Meta Platforms, Inc.", tgt="CoreWeave"),
                )],
            ),
        ],
    )
    extractor = BatchClaudeExtractor(client=client, poll_initial_s=0.0)
    results = await extractor.extract_many(contexts)
    assert results[0] == []
    assert len(results[1]) == 1 and results[1][0].source_party_name == "Anthropic"
    assert len(results[2]) == 1 and results[2][0].source_party_name == "Meta Platforms, Inc."


@pytest.mark.asyncio
async def test_unknown_custom_id_logged_and_skipped() -> None:
    """A result with a custom_id we never submitted gets dropped, others survive."""
    client = _mock_client(
        results=[
            _result_row(
                _encode_custom_id(0, 0),
                content_blocks=[_tool_use_block("record_deal", _payload())],
            ),
            _result_row(
                "ctx-99999-chunk-999",  # never submitted
                content_blocks=[_tool_use_block("record_deal", _payload())],
            ),
        ],
    )
    extractor = BatchClaudeExtractor(client=client, poll_initial_s=0.0)
    results = await extractor.extract_many([_ctx()])
    assert len(results[0]) == 1


# ---------- polling ----------


@pytest.mark.asyncio
async def test_poll_loop_waits_for_ended_status() -> None:
    """Two non-terminal polls, then ended — extractor must wait through both."""
    poll_calls = {"n": 0}

    async def counting_retrieve(batch_id: str) -> SimpleNamespace:
        poll_calls["n"] += 1
        status = ["in_progress", "in_progress", "ended"][poll_calls["n"] - 1]
        return SimpleNamespace(
            id=batch_id,
            processing_status=status,
            request_counts=SimpleNamespace(
                succeeded=1, errored=0, processing=0, canceled=0, expired=0,
            ),
        )

    client = _mock_client(results=[])
    client.messages.batches.retrieve = counting_retrieve  # type: ignore[assignment]

    extractor = BatchClaudeExtractor(
        client=client,
        poll_initial_s=0.001,  # essentially zero
        poll_max_s=0.001,
        poll_backoff=1.0,
    )
    await extractor.extract_many([_ctx()])
    assert poll_calls["n"] == 3


@pytest.mark.asyncio
async def test_poll_loop_times_out_when_batch_never_ends() -> None:
    """If processing never reaches ended within max_wait_s, raise TimeoutError."""

    async def stuck_retrieve(batch_id: str) -> SimpleNamespace:
        return SimpleNamespace(
            id=batch_id,
            processing_status="in_progress",
            request_counts=SimpleNamespace(
                succeeded=0, errored=0, processing=1, canceled=0, expired=0,
            ),
        )

    client = _mock_client()
    client.messages.batches.retrieve = stuck_retrieve  # type: ignore[assignment]

    extractor = BatchClaudeExtractor(
        client=client,
        poll_initial_s=0.005,
        poll_max_s=0.005,
        poll_backoff=1.0,
        max_wait_s=0.05,  # ~10 polls
    )
    with pytest.raises(TimeoutError, match="still 'in_progress'"):
        await extractor.extract_many([_ctx()])


# ---------- single-doc convenience ----------


@pytest.mark.asyncio
async def test_extract_wraps_extract_many() -> None:
    """The single-doc extract() returns the unwrapped per-context list."""
    client = _mock_client(
        results=[
            _result_row(
                _encode_custom_id(0, 0),
                content_blocks=[_tool_use_block("record_deal", _payload())],
            ),
        ],
    )
    extractor = BatchClaudeExtractor(client=client, poll_initial_s=0.0)
    deals = await extractor.extract(_ctx())
    assert len(deals) == 1
    assert deals[0].extractor_name == "claude:opus-4-7:batch"


# Touch asyncio so the import is real for type-checking parity.
_ = asyncio
