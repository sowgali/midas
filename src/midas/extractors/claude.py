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

from typing import TYPE_CHECKING, Any

import anthropic
import structlog
from pydantic import ValidationError

from midas.config import settings

from .base import ExtractedDeal, ExtractionContext, KnownParty

log = structlog.get_logger(__name__)

if TYPE_CHECKING:
    from anthropic.types import ToolUseBlock

_MODEL_ID = "claude-opus-4-7"

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
        client = self._get_client()

        user_message = (
            f"Source URL: {context.source_url}\n"
            f"Source type: {context.source_type.value}\n\n"
            f"Known parties (resolve mentions to these canonical names "
            f"when applicable):\n"
            f"{_format_known_parties(context.known_parties)}\n\n"
            f"Document text:\n"
            f"---\n"
            f"{context.document_text}\n"
            f"---\n\n"
            f"Emit one `record_deal` tool call per distinct money-flow "
            f"claim. If the document has no such claims, emit no tool calls."
        )

        # Cache placement: the system prompt and the tool definition are
        # the stable prefix shared across every extraction call. We mark
        # cache_control on the last system block and on the (single)
        # tool definition. The SDK renders tools → system → messages, so
        # both ranges land before the per-document user message and
        # cache cleanly. Per the prompt-caching guidance: stable bytes
        # first, volatile bytes after.
        response = await client.messages.create(
            model=self._model,
            max_tokens=4096,
            system=[
                {
                    "type": "text",
                    "text": _SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            tools=[
                {
                    "name": "record_deal",
                    "description": (
                        "Record a single directional money-flow claim "
                        "from the document. Call once per distinct deal."
                    ),
                    "input_schema": _RECORD_DEAL_SCHEMA,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": user_message}],
        )

        deals: list[ExtractedDeal] = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            tool_block: ToolUseBlock = block
            if tool_block.name != "record_deal":
                continue
            payload = dict(tool_block.input) if isinstance(tool_block.input, dict) else {}
            # Stamp the extractor name server-side rather than asking
            # the model to fill it in — saves tokens and removes a
            # whole class of "model lied about its own name" bugs.
            payload["extractor_name"] = self.name
            try:
                deals.append(ExtractedDeal.model_validate(payload))
            except ValidationError as exc:
                # One bad tool_use shouldn't tank the whole document —
                # extractors are best-effort. Log enough to surface
                # systematic schema drift later without blocking the
                # rest of the deals the model recovered.
                log.warning(
                    "claude.invalid_tool_use",
                    error=str(exc),
                    payload_keys=sorted(payload.keys()),
                )
                continue

        return deals
