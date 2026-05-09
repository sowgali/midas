"""EvidenceSpan — provenance link from a Deal to the source text.

This is the table that makes the graph trustworthy. Every claim about
money flow points back at *the exact substring* of *the exact source
document* it came from, plus the extractor that produced it.
"""

from __future__ import annotations

import uuid

from sqlalchemy import Column, Index, Text
from sqlmodel import Field, SQLModel


class EvidenceSpan(SQLModel, table=True):
    __tablename__ = "evidence_span"

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)

    deal_id: uuid.UUID = Field(foreign_key="deal.id", index=True)
    source_id: uuid.UUID = Field(foreign_key="source.id", index=True)

    text_snippet: str = Field(sa_column=Column(Text(), nullable=False))

    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)

    extractor: str = Field(
        max_length=128,
        description='Identifier such as "claude:opus-4-7" or "regex:dollar_amount".',
    )

    __table_args__ = (Index("ix_evidence_deal_source", "deal_id", "source_id"),)
