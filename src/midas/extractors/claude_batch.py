"""Batch-mode Claude extractor (50% cheaper, async, same contract).

Anthropic's Message Batches API processes up to 100K async requests in
one job at half the per-token cost of real-time. For midas, where every
extraction call is independent and an ingest run can spawn thousands
of them, this is a near-pure win — minutes of latency in exchange for
~50% cost savings.

This module exposes :class:`BatchClaudeExtractor` with the same
:class:`midas.extractors.base.Extractor` interface as the real-time
``ClaudeExtractor`` plus a bulk :meth:`extract_many` method that the
pipeline calls when ``--batch`` is set. The single-doc :meth:`extract`
is a thin convenience that wraps a one-element ``extract_many``.

Pipeline-shape implication
--------------------------
The real-time path updates the entity resolver between articles in a
feed, so article #3 extracts with knowledge of entities discovered by
articles #1 and #2. In batch mode every article in a batch fires
together, so the resolver doesn't update mid-batch. In practice this
costs little because:

1. ``normalize_entity_name`` already collapses corporate-suffix
   variants without an LLM hint.
2. The :mod:`midas.dedup` layer absorbs near-duplicate deals even when
   the LLM names a party slightly differently in two articles.
3. Cross-feed entity resolution still happens at the resolver layer
   because we process feed-by-feed.

What it shares with the real-time extractor
-------------------------------------------
Prompt shape, chunking, and parsing are imported from
:mod:`midas.extractors.claude` so the two paths can't drift. Each
chunk becomes one batch request with a ``custom_id`` that encodes
``(context_index, chunk_index)`` for stitching results back per-doc.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections import defaultdict
from typing import TYPE_CHECKING, Any, cast

import anthropic
import structlog

from midas.config import settings

from .base import ExtractedDeal, ExtractionContext
from .claude import (
    _MODEL_ID,
    _build_request_params,
    _build_user_message,
    _dedupe_within_document,
    _parse_tool_uses,
    _split_into_chunks,
)

if TYPE_CHECKING:
    from anthropic.types.messages import MessageBatch
    from anthropic.types.messages.batch_create_params import Request as BatchRequestParam

log = structlog.get_logger(__name__)


# Poll cadence. Anthropic's docs say most batches finish in under an
# hour; a 24-hour SLA is the worst case. We start with a fast poll and
# back off so we don't hammer the API on quick batches but don't sit
# idle on slower ones.
_POLL_INITIAL_S = 10.0
_POLL_MAX_S = 120.0
_POLL_BACKOFF = 1.5

# Wall-clock cap. Anthropic's batches guarantee 24h; we bail earlier so
# an ingest run doesn't hang indefinitely if something goes sideways.
_DEFAULT_MAX_WAIT_S = 4 * 60 * 60.0


def _encode_custom_id(context_index: int, chunk_index: int) -> str:
    """Encode (context, chunk) into the per-request id we get back later.

    Anthropic constrains ``custom_id`` to ASCII / ≤64 chars. The encoding
    is deliberately readable for log-spelunking.
    """
    return f"ctx-{context_index:05d}-chunk-{chunk_index:03d}"


def _decode_custom_id(custom_id: str) -> tuple[int, int]:
    """Parse a custom_id back into ``(context_index, chunk_index)``.

    Raises :class:`ValueError` on a malformed id — caller logs and skips
    so a single weird result doesn't tank the whole batch.
    """
    parts = custom_id.split("-")
    if len(parts) != 4 or parts[0] != "ctx" or parts[2] != "chunk":
        raise ValueError(f"unparseable custom_id: {custom_id!r}")
    return int(parts[1]), int(parts[3])


class BatchClaudeExtractor:
    """Claude extractor that submits requests via the Message Batches API.

    Half the per-token cost of real-time at the price of async latency.
    Use when extracting from many independent documents (an ingest run,
    a frontier-loop round) — not for interactive single-doc queries.
    """

    name: str = "claude:opus-4-7:batch"

    def __init__(
        self,
        client: anthropic.AsyncAnthropic | None = None,
        model: str = _MODEL_ID,
        *,
        poll_initial_s: float = _POLL_INITIAL_S,
        poll_max_s: float = _POLL_MAX_S,
        poll_backoff: float = _POLL_BACKOFF,
        max_wait_s: float = _DEFAULT_MAX_WAIT_S,
    ) -> None:
        self._injected_client = client
        self._model = model
        self._poll_initial_s = poll_initial_s
        self._poll_max_s = poll_max_s
        self._poll_backoff = poll_backoff
        self._max_wait_s = max_wait_s

    def _get_client(self) -> anthropic.AsyncAnthropic:
        if self._injected_client is not None:
            return self._injected_client
        if settings.anthropic_api_key is None:
            raise RuntimeError("MIDAS_ANTHROPIC_API_KEY not set")
        return anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key.get_secret_value())

    async def extract(self, context: ExtractionContext) -> list[ExtractedDeal]:
        """Single-doc convenience that wraps :meth:`extract_many`.

        For one document there's no batching benefit — using real-time
        would be cheaper *for that doc* — but callers that have only
        a single context still get a working extractor with the same
        interface as :class:`ClaudeExtractor`.
        """
        results = await self.extract_many([context])
        return results[0] if results else []

    async def extract_many(
        self,
        contexts: list[ExtractionContext],
    ) -> list[list[ExtractedDeal]]:
        """Submit all contexts as one Anthropic batch; return per-context deal lists.

        Each context's document_text is chunked the same way the
        real-time extractor chunks. Every chunk becomes one batch
        request. After the batch completes the results are grouped by
        context_index, char offsets are translated back to
        document-relative, and :func:`_dedupe_within_document` collapses
        duplicates from overlap regions.

        Per-context output order matches input order, so the caller can
        zip the result list with the original contexts.
        """
        if not contexts:
            return []

        client = self._get_client()

        requests: list[BatchRequestParam] = []
        # Map: custom_id → (context_index, doc_offset). We don't need
        # the chunk_index after dispatch — only the doc_offset matters
        # for offset translation when results come back.
        chunk_meta: dict[str, tuple[int, int]] = {}
        # Track each context's chunk count so we can log a useful summary.
        chunks_per_context: list[int] = []

        for ctx_idx, context in enumerate(contexts):
            chunks = _split_into_chunks(context.document_text)
            chunks_per_context.append(len(chunks))
            for chunk_idx, (chunk_text, doc_offset) in enumerate(chunks):
                custom_id = _encode_custom_id(ctx_idx, chunk_idx)
                chunk_meta[custom_id] = (ctx_idx, doc_offset)
                user_message = _build_user_message(
                    context=context,
                    chunk_text=chunk_text,
                    chunk_index=chunk_idx,
                    chunk_count=len(chunks),
                )
                params = _build_request_params(model=self._model, user_message=user_message)
                # Anthropic's TypedDict for the batch request expects a
                # MessageCreateParamsNonStreaming for ``params``; ours is
                # structurally compatible (same keys + types) but mypy
                # can't see that through a plain dict[str, Any] — cast.
                requests.append(
                    cast(
                        "BatchRequestParam",
                        {"custom_id": custom_id, "params": cast("Any", params)},
                    ),
                )

        batch_id = await self._submit_batch(client, requests)
        log.info(
            "claude.batch.submitted",
            batch_id=batch_id,
            request_count=len(requests),
            context_count=len(contexts),
            multipass_contexts=sum(1 for n in chunks_per_context if n > 1),
        )

        await self._poll_until_ended(client, batch_id)

        return await self._collect_results(
            client,
            batch_id=batch_id,
            chunk_meta=chunk_meta,
            context_count=len(contexts),
        )

    # ---------- internal: submit / poll / collect ----------

    async def _submit_batch(
        self,
        client: anthropic.AsyncAnthropic,
        requests: list[BatchRequestParam],
    ) -> str:
        batch: MessageBatch = await client.messages.batches.create(requests=requests)
        return batch.id

    async def _poll_until_ended(
        self,
        client: anthropic.AsyncAnthropic,
        batch_id: str,
    ) -> None:
        """Poll the batch with exponential backoff until processing ends.

        Anthropic's processing_status transitions in_progress →
        canceling → ended; ``ended`` is the terminal happy path. We
        don't differentiate between "ended with all-succeeded" and
        "ended with some failures" here — per-request failures show up
        when we stream results.
        """
        start = time.monotonic()
        delay = self._poll_initial_s
        attempts = 0
        while True:
            await asyncio.sleep(delay)
            attempts += 1
            batch: MessageBatch = await client.messages.batches.retrieve(batch_id)
            status = batch.processing_status
            log.debug(
                "claude.batch.poll",
                batch_id=batch_id,
                attempts=attempts,
                status=status,
                succeeded=getattr(batch.request_counts, "succeeded", None),
                errored=getattr(batch.request_counts, "errored", None),
                processing=getattr(batch.request_counts, "processing", None),
            )
            if status == "ended":
                elapsed = time.monotonic() - start
                log.info(
                    "claude.batch.ended",
                    batch_id=batch_id,
                    elapsed_s=round(elapsed, 1),
                    succeeded=getattr(batch.request_counts, "succeeded", 0),
                    errored=getattr(batch.request_counts, "errored", 0),
                    expired=getattr(batch.request_counts, "expired", 0),
                    canceled=getattr(batch.request_counts, "canceled", 0),
                )
                return
            elapsed = time.monotonic() - start
            if elapsed >= self._max_wait_s:
                raise TimeoutError(
                    f"Batch {batch_id} still {status!r} after "
                    f"{elapsed:.0f}s (cap={self._max_wait_s:.0f}s)",
                )
            delay = min(delay * self._poll_backoff, self._poll_max_s)

    async def _collect_results(
        self,
        client: anthropic.AsyncAnthropic,
        *,
        batch_id: str,
        chunk_meta: dict[str, tuple[int, int]],
        context_count: int,
    ) -> list[list[ExtractedDeal]]:
        """Stream batch results, parse tool_uses, dedupe per-context.

        Each result row carries the ``custom_id`` we set on submit so
        we can route it back to the right context. Anthropic's
        ``result.result.type`` is one of ``succeeded`` / ``errored`` /
        ``expired`` / ``canceled``; only the first carries a message.
        """
        per_ctx: dict[int, list[ExtractedDeal]] = defaultdict(list)
        succeeded = 0
        failed = 0

        async for result in await client.messages.batches.results(batch_id):
            cid: str | None = getattr(result, "custom_id", None)
            if cid is None:
                failed += 1
                continue
            try:
                ctx_idx, doc_offset = chunk_meta[cid]
            except (KeyError, ValueError):
                log.warning("claude.batch.unknown_custom_id", custom_id=cid)
                failed += 1
                continue

            inner = getattr(result, "result", None)
            inner_type = getattr(inner, "type", None) if inner is not None else None
            if inner_type != "succeeded":
                log.warning(
                    "claude.batch.request_failed",
                    custom_id=cid,
                    result_type=inner_type,
                )
                failed += 1
                continue

            message = getattr(inner, "message", None)
            content = getattr(message, "content", None) if message is not None else None
            if not content:
                succeeded += 1
                continue

            deals = _parse_tool_uses(
                list(content),
                doc_offset=doc_offset,
                extractor_name=self.name,
            )
            per_ctx[ctx_idx].extend(deals)
            succeeded += 1

        log.info(
            "claude.batch.collected",
            batch_id=batch_id,
            results_succeeded=succeeded,
            results_failed=failed,
        )

        return [_dedupe_within_document(per_ctx.get(i, [])) for i in range(context_count)]


# Silence unused-import warning for the uuid alias kept around for API parity.
_ = uuid
