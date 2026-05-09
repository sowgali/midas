"""Extractor interface and shared data classes.

The :class:`Extractor` Protocol is the seam between the *parsers* layer
(text → cleaned text) and the *normalizers* layer (extracted candidates →
persisted :class:`midas.models.Deal`). Concrete extractors turn one
``ExtractionContext`` (a chunk of source text plus metadata) into a list
of :class:`ExtractedDeal` candidates.

``ExtractedDeal`` is intentionally **not** a SQLModel — it carries party
*names* rather than UUIDs, and exists only as the wire format between
extractors and the entity-resolution / normalization step that maps
names → ``Entity`` rows and produces a persistable ``Deal``.
"""

from __future__ import annotations

import uuid
from datetime import date
from decimal import Decimal
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

from midas.models.types import DealStatus, DealType, SourceType


class KnownParty(BaseModel):
    """An entity already resolved from the document's metadata.

    Extractors use these to ground party references in the text against
    canonical names + aliases known at extraction time. The ``entity_id``
    is preserved so the normalizer can short-circuit entity resolution
    when an extractor's party name matches one of these directly.
    """

    entity_id: uuid.UUID
    canonical_name: str
    aliases: list[str] = Field(default_factory=list)


class ExtractionContext(BaseModel):
    """Everything a single extractor invocation needs.

    Bundling source metadata with the text avoids threading the same
    arguments through every extractor signature and makes it cheap to
    add new context fields (e.g. published_at) later.
    """

    source_id: uuid.UUID
    source_url: str
    source_type: SourceType
    known_parties: list[KnownParty] = Field(default_factory=list)
    document_text: str


class ExtractedDeal(BaseModel):
    """A single Deal candidate produced by an extractor.

    Mirrors the shape of :class:`midas.models.Deal` but with party
    *names* instead of UUIDs. The normalizer resolves names → entities
    and produces the persistable ``Deal`` + ``EvidenceSpan`` pair.
    ``char_start`` / ``char_end`` index into ``ExtractionContext.document_text``.
    """

    source_party_name: str
    target_party_name: str

    deal_type: DealType
    status: DealStatus

    amount_usd: Decimal | None = None
    amount_native: Decimal | None = None
    currency: str | None = None

    announced_at: date | None = None
    closes_at: date | None = None

    confidence: float = Field(ge=0.0, le=1.0)
    description: str

    evidence_text_snippet: str
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)

    extractor_name: str


@runtime_checkable
class Extractor(Protocol):
    """Strategy interface for any text → ExtractedDeal implementation.

    Implementations live alongside this file: :mod:`midas.extractors.regex`
    for the cheap deterministic pass and :mod:`midas.extractors.claude`
    for the LLM tail. The pipeline composes them, dedup'ing by Deal
    identity at the normalizer.
    """

    name: str

    async def extract(self, context: ExtractionContext) -> list[ExtractedDeal]: ...
