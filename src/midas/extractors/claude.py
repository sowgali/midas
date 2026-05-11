"""Claude (Anthropic) LLM extractor.

The LLM extractor handles everything the regex pass misses: indirect
references, multi-clause sentences, partnerships without an obvious
payer, etc. Implementation strategy:

* **Tool use, not free-form JSON.** The model emits one ``record_deal``
  ``tool_use`` block per claim. The SDK already validates the tool call
  shape against ``input_schema``; we run the input through pydantic for
  the constraints JSON Schema can't express (Decimal coercion, ranges,
  enum membership against our :class:`DealType` / :class:`DealStatus`).
* **Prompt caching on the stable prefix.** The system prompt and the
  tool definition list never change between calls — they're the right
  things to cache. We mark ``cache_control`` on the last system block
  and on the last tool definition so the SDK caches both. The user
  message (document text + known parties) is per-request and stays
  uncached.

Default model is ``claude-opus-4-7`` (per the claude-api skill's
guidance). Adaptive thinking is left off here: extraction is a
relatively shallow structured-output task and the latency hit isn't
worth the marginal recall.
"""

from __future__ import annotations

from typing import Any

import anthropic
import structlog
from pydantic import ValidationError

from midas.config import settings

from .base import ExtractedDeal, ExtractionContext, KnownParty

log = structlog.get_logger(__name__)

_MODEL_ID = "claude-opus-4-7"

# Chunking budget for multi-pass extraction. Opus 4.7 caps at 1M input
# tokens; we use ~500K chars per chunk (~150K tokens at the conservative
# 3-chars-per-token ratio) so a single chunk plus system prompt + tool
# def + known-parties + response leaves comfortable headroom — and so a
# document up to ~8 MB still fits in a reasonable number of API calls.
# Larger chunks = fewer calls but more risk of edge-case overflow from
# tokenizer-heavy text (CJK, dense numerics, base64 blobs in HTML).
_CHUNK_CHARS = 500_000

# Overlap between successive chunks. A money-flow sentence that
# straddles a chunk boundary would otherwise vanish into the gap;
# duplicating ~10 KB on each side guarantees any single deal-bearing
# span lands wholly inside at least one chunk. Within-document
# dedup collapses the duplicates produced by overlap regions.
_CHUNK_OVERLAP_CHARS = 10_000

# Sanity cap on number of chunks per document. Beyond ~8 MB a doc is
# almost certainly an HTML render of a PDF or a multi-article
# concatenation — at that point we'd rather log loudly than make
# dozens of API calls. 16 chunks of 500 KB = 8 MB document.
_MAX_CHUNKS_PER_DOC = 16

_SYSTEM_PROMPT = """\
You extract directional money-flow claims from press releases, filings, \
and news text. For every distinct claim about money moving from one \
named party to another, emit exactly one `record_deal` tool call.

Rules:
1. The deal is directional. `source_party_name` is the payer; \
`target_party_name` is the recipient.
2. Cite the EXACT verbatim snippet from the input as `evidence_text_snippet`. \
`char_start` and `char_end` MUST index into the document text the user provides.
3. Resolve party names to one of the known parties when possible, using \
the canonical name. If a party isn't in the known list, use the \
mention as it appears in the text.
4. `amount_usd` is the USD-normalized amount as a number (no currency \
symbol). Leave it null if the source doesn't state an amount.
5. `confidence` is 0..1. Use ≥0.85 only when both parties and the \
amount are explicit and unambiguous. Use ≤0.5 for vague or indirect claims.
6. Do not emit a tool call for non-monetary partnerships, hires, \
product launches, or general commentary."""

# JSON Schema for the record_deal tool. Hand-written rather than derived
# from ExtractedDeal.model_json_schema() so the model sees a clean,
# focused contract — no $defs indirection, no internal-only fields.
_RECORD_DEAL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "source_party_name": {
            "type": "string",
            "description": "Canonical name of the payer (money source).",
        },
        "target_party_name": {
            "type": "string",
            "description": "Canonical name of the recipient (money target).",
        },
        "deal_type": {
            "type": "string",
            "enum": [
                "investment",
                "acquisition",
                "commercial_contract",
                "partnership",
                "licensing",
                "debt",
                "grant",
            ],
        },
        "status": {
            "type": "string",
            "enum": ["announced", "closed", "rumored", "terminated"],
        },
        "amount_usd": {
            "type": ["number", "null"],
            "description": "USD-normalized amount. Null if not stated.",
        },
        "amount_native": {
            "type": ["number", "null"],
            "description": "Amount in the native currency. Null if not stated.",
        },
        "currency": {
            "type": ["string", "null"],
            "description": "ISO 4217 currency code, or null.",
        },
        "announced_at": {
            "type": ["string", "null"],
            "description": "ISO 8601 date (YYYY-MM-DD), or null.",
        },
        "closes_at": {
            "type": ["string", "null"],
            "description": "ISO 8601 date (YYYY-MM-DD), or null.",
        },
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
        },
        "description": {
            "type": "string",
            "description": "Short human-readable summary (≤2 sentences).",
        },
        "evidence_text_snippet": {
            "type": "string",
            "description": "EXACT verbatim quote from the document.",
        },
        "char_start": {
            "type": "integer",
            "minimum": 0,
            "description": "Inclusive start offset of the snippet in the document.",
        },
        "char_end": {
            "type": "integer",
            "minimum": 0,
            "description": "Exclusive end offset of the snippet in the document.",
        },
    },
    "required": [
        "source_party_name",
        "target_party_name",
        "deal_type",
        # status is intentionally NOT required — pydantic defaults to
        # ANNOUNCED, which matches the dominant case for 8-K / press
        # release sources where the deal is being announced as the
        # filing is published.
        "confidence",
        "description",
        "evidence_text_snippet",
        "char_start",
        "char_end",
    ],
}


def _split_into_chunks(
    text: str,
    *,
    chunk_chars: int = _CHUNK_CHARS,
    overlap_chars: int = _CHUNK_OVERLAP_CHARS,
) -> list[tuple[str, int]]:
    """Split ``text`` into overlapping chunks for multi-pass extraction.

    Returns a list of ``(chunk_text, doc_offset)`` tuples where
    ``doc_offset`` is the index of ``chunk_text[0]`` in the original
    document — used by the extractor to translate model-emitted
    ``char_start`` / ``char_end`` offsets back into document space.

    A boundary search looks for the nearest ``\\n\\n`` paragraph break
    in the tail half of the window so chunks don't routinely cut a
    sentence in half (the overlap absorbs the residual cases).

    >>> [(c, o) for c, o in _split_into_chunks("abcdef", chunk_chars=10)]
    [('abcdef', 0)]
    """
    if not text:
        return [("", 0)]
    if len(text) <= chunk_chars:
        return [(text, 0)]
    if overlap_chars >= chunk_chars:
        raise ValueError("overlap_chars must be smaller than chunk_chars")

    chunks: list[tuple[str, int]] = []
    start = 0
    n = len(text)
    while start < n:
        target_end = min(start + chunk_chars, n)
        end = target_end
        if end < n:
            # Search the tail half of the window for a paragraph break;
            # falls back to the target if no break is found in range.
            search_floor = start + chunk_chars // 2
            split_at = text.rfind("\n\n", search_floor, target_end)
            if split_at > 0:
                end = split_at + 2  # include the blank line in the prior chunk
        chunks.append((text[start:end], start))
        if end >= n:
            break
        start = max(end - overlap_chars, end - chunk_chars + 1)
        # Defensive bound on chunk count.
        if len(chunks) >= _MAX_CHUNKS_PER_DOC:
            log.warning(
                "claude.chunk_cap_reached",
                total_chars=n,
                kept_chars=end,
                max_chunks=_MAX_CHUNKS_PER_DOC,
            )
            break
    return chunks


def _dedupe_within_document(deals: list[ExtractedDeal]) -> list[ExtractedDeal]:
    """Collapse duplicate deals produced by overlap regions of chunks.

    Two extractions of the *same source document* are considered the
    same deal when their (source, target, deal_type, amount, status)
    tuple matches case-insensitively. First-seen wins so the earliest
    char offsets — which point at the document's lede if the model
    consistently picks the strongest evidence — are preserved.

    The V1.6 cross-source dedup layer in ``midas.dedup`` handles the
    case where the SAME deal is reported in two different sources;
    this helper handles within-source duplicates that the chunker
    itself creates via overlap.
    """
    seen: set[tuple[str, str, str, object, str]] = set()
    out: list[ExtractedDeal] = []
    for d in deals:
        key = (
            d.source_party_name.lower().strip(),
            d.target_party_name.lower().strip(),
            d.deal_type.value if hasattr(d.deal_type, "value") else str(d.deal_type),
            d.amount_usd,
            d.status.value if hasattr(d.status, "value") else str(d.status),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(d)
    return out


def _format_known_parties(parties: list[KnownParty]) -> str:
    """Render the known-parties block for the user message.

    Stays compact and deterministic so that a) the message is short and
    b) reordering the list across calls doesn't accidentally invalidate
    any future per-document caching we layer on later.
    """
    if not parties:
        return "(none provided)"
    lines = []
    for party in parties:
        if party.aliases:
            alias_str = ", ".join(party.aliases)
            lines.append(f"- {party.canonical_name} (aliases: {alias_str})")
        else:
            lines.append(f"- {party.canonical_name}")
    return "\n".join(lines)


def _build_user_message(
    *,
    context: ExtractionContext,
    chunk_text: str,
    chunk_index: int,
    chunk_count: int,
) -> str:
    """Render the per-request user message.

    Shared by the real-time and batch extractors so the prompt shape
    stays in lockstep (a divergence here would cause caching misses
    and silent extraction-quality drift between paths).
    """
    chunk_note = (
        f"\n(NOTE: this is chunk {chunk_index + 1} of {chunk_count} from a "
        f"larger document; extract claims that appear in THIS chunk only.)"
        if chunk_count > 1
        else ""
    )
    return (
        f"Source URL: {context.source_url}\n"
        f"Source type: {context.source_type.value}{chunk_note}\n\n"
        f"Known parties (resolve mentions to these canonical names "
        f"when applicable):\n"
        f"{_format_known_parties(context.known_parties)}\n\n"
        f"Document text:\n"
        f"---\n"
        f"{chunk_text}\n"
        f"---\n\n"
        f"Emit one `record_deal` tool call per distinct money-flow "
        f"claim. If the document has no such claims, emit no tool calls."
    )


def _build_request_params(
    *,
    model: str,
    user_message: str,
    max_tokens: int = 4096,
) -> dict[str, Any]:
    """Build the kwargs dict for messages.create() (or its batch equivalent).

    Cache placement: system prompt and tool definition are the stable
    prefix shared across every extraction call. ``cache_control`` lands
    on both so the SDK caches them (stable bytes first, volatile bytes
    after, per the prompt-caching guidance).
    """
    return {
        "model": model,
        "max_tokens": max_tokens,
        "system": [
            {
                "type": "text",
                "text": _SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        "tools": [
            {
                "name": "record_deal",
                "description": (
                    "Record a single directional money-flow claim from "
                    "the document. Call once per distinct deal."
                ),
                "input_schema": _RECORD_DEAL_SCHEMA,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        "messages": [{"role": "user", "content": user_message}],
    }


def _parse_tool_uses(
    content_blocks: list[Any],
    *,
    doc_offset: int,
    extractor_name: str,
) -> list[ExtractedDeal]:
    """Walk a response's content blocks; yield validated ``ExtractedDeal``s.

    ``doc_offset`` is added to model-emitted ``char_start``/``char_end``
    so offsets are document-relative (not chunk-relative). Validation
    failures on a single tool_use are logged and skipped — surviving
    deals in the same response still surface.

    Pulled out into a module-level helper so both the real-time and
    batch extractors apply identical parsing semantics; otherwise it's
    easy for the batch path to drift on (say) the offset translation.
    """
    deals: list[ExtractedDeal] = []
    for block in content_blocks:
        if getattr(block, "type", None) != "tool_use":
            continue
        if getattr(block, "name", None) != "record_deal":
            continue
        raw_input = getattr(block, "input", None)
        payload = dict(raw_input) if isinstance(raw_input, dict) else {}
        if doc_offset:
            start_val = payload.get("char_start")
            if isinstance(start_val, int):
                payload["char_start"] = start_val + doc_offset
            end_val = payload.get("char_end")
            if isinstance(end_val, int):
                payload["char_end"] = end_val + doc_offset
        payload["extractor_name"] = extractor_name
        try:
            deals.append(ExtractedDeal.model_validate(payload))
        except ValidationError as exc:
            log.warning(
                "claude.invalid_tool_use",
                error=str(exc),
                payload_keys=sorted(payload.keys()),
            )
            continue
    return deals


class ClaudeExtractor:
    """LLM extractor backed by ``claude-opus-4-7`` via the Anthropic SDK."""

    name: str = "claude:opus-4-7"

    def __init__(
        self,
        client: anthropic.AsyncAnthropic | None = None,
        model: str = _MODEL_ID,
    ) -> None:
        # The constructor accepts an injected client purely for testing —
        # production code paths should rely on settings + lazy build in
        # ``extract`` so the API key check happens at call time, not at
        # import time.
        self._injected_client = client
        self._model = model

    def _get_client(self) -> anthropic.AsyncAnthropic:
        if self._injected_client is not None:
            return self._injected_client
        if settings.anthropic_api_key is None:
            raise RuntimeError("MIDAS_ANTHROPIC_API_KEY not set")
        return anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key.get_secret_value())

    async def extract(self, context: ExtractionContext) -> list[ExtractedDeal]:
        """Extract money-flow claims from ``context.document_text``.

        Documents up to ``_CHUNK_CHARS`` (~500 KB / ~150K tokens) go in
        a single API call. Larger documents are split into overlapping
        chunks (multi-pass extraction); deals are accumulated and
        deduplicated across chunks before returning. No transactional
        info is dropped — every byte of the document gets scanned at
        least once.
        """
        client = self._get_client()
        document_text = context.document_text
        chunks = _split_into_chunks(document_text)

        if len(chunks) > 1:
            log.info(
                "claude.multipass.start",
                source_url=context.source_url,
                total_chars=len(document_text),
                chunk_count=len(chunks),
            )

        all_deals: list[ExtractedDeal] = []
        for i, (chunk_text, doc_offset) in enumerate(chunks):
            chunk_deals = await self._extract_chunk(
                client,
                context=context,
                chunk_text=chunk_text,
                doc_offset=doc_offset,
                chunk_index=i,
                chunk_count=len(chunks),
            )
            all_deals.extend(chunk_deals)

        if len(chunks) > 1:
            deduped = _dedupe_within_document(all_deals)
            log.info(
                "claude.multipass.done",
                source_url=context.source_url,
                deals_before_dedup=len(all_deals),
                deals_after_dedup=len(deduped),
            )
            return deduped
        return all_deals

    async def _extract_chunk(
        self,
        client: anthropic.AsyncAnthropic,
        *,
        context: ExtractionContext,
        chunk_text: str,
        doc_offset: int,
        chunk_index: int,
        chunk_count: int,
    ) -> list[ExtractedDeal]:
        """Run one extraction pass against one chunk.

        ``doc_offset`` is added back to the model-emitted
        ``char_start``/``char_end`` so the resulting offsets are
        meaningful relative to the original (full) document.
        """
        user_message = _build_user_message(
            context=context,
            chunk_text=chunk_text,
            chunk_index=chunk_index,
            chunk_count=chunk_count,
        )
        params = _build_request_params(model=self._model, user_message=user_message)
        try:
            response = await client.messages.create(**params)
        except anthropic.BadRequestError as exc:
            # Fail-soft: a single chunk hitting an API error should NOT
            # kill the whole document (and certainly not the run). Log,
            # return zero deals for this chunk, and let the other
            # chunks contribute what they can.
            log.warning(
                "claude.bad_request",
                source_url=context.source_url,
                chunk_index=chunk_index,
                chunk_chars=len(chunk_text),
                error=str(exc),
            )
            return []
        return _parse_tool_uses(
            list(response.content),
            doc_offset=doc_offset,
            extractor_name=self.name,
        )
