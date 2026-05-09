"""Source-layer base types.

A :class:`RawDocument` is what every source adapter hands off to the parser
layer: the bytes we fetched plus enough metadata to round-trip into the
``Source`` SQLModel later. It is a plain ``BaseModel`` (not a SQLModel
table) — persistence is the storage layer's job.

The :class:`Source` ABC is intentionally minimal. Each concrete adapter
defines its own ``fetch(...)`` parameter shape because the upstream
identifiers differ wildly (SEC needs a filing accession; an RSS feed
takes an item URL). The contract is: produce a fully-populated
``RawDocument``, asynchronously, using the shared rate-limited client.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..models.types import SourceType


class RawDocument(BaseModel):
    """A single fetched document, prior to parsing.

    Identity is ``content_sha256`` — the digest of the raw bytes — so two
    fetches of the same URL collapse to the same document and a re-fetch
    of bit-identical content is a no-op at the storage layer.

    ``content_sha256`` is auto-computed from ``content_bytes`` if omitted
    and re-validated against the bytes if both are provided (mismatch is a
    programming error and surfaces as ``ValidationError``).
    """

    model_config = ConfigDict(frozen=True)

    url: str
    content_bytes: bytes
    content_sha256: str = Field(default="")
    fetched_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    source_type: SourceType
    publisher: str
    title: str | None = None
    published_at: datetime | None = None

    @field_validator("fetched_at", "published_at")
    @classmethod
    def _ensure_tz(cls, v: datetime | None) -> datetime | None:
        if v is None:
            return None
        if v.tzinfo is None:
            return v.replace(tzinfo=UTC)
        return v

    @model_validator(mode="after")
    def _fill_or_check_sha(self) -> RawDocument:
        digest = hashlib.sha256(self.content_bytes).hexdigest()
        if not self.content_sha256:
            # frozen model; bypass __setattr__ via __dict__
            object.__setattr__(self, "content_sha256", digest)
        elif self.content_sha256 != digest:
            raise ValueError(
                f"content_sha256 mismatch: provided {self.content_sha256!r} but "
                f"sha256(content_bytes)={digest!r}"
            )
        return self


class Source(ABC):
    """Abstract source adapter.

    Concrete subclasses implement :meth:`fetch` with whatever parameter
    shape makes sense for that upstream (e.g. SEC takes ``FilingMetadata``,
    an RSS source takes an item URL). The contract is:

    * the call is asynchronous;
    * the call goes through a rate-limited, cached HTTP client;
    * the return value is a fully-populated :class:`RawDocument`.
    """

    @abstractmethod
    async def fetch(self, *args: Any, **kwargs: Any) -> RawDocument:
        """Fetch one document and return it as a :class:`RawDocument`."""
        raise NotImplementedError
