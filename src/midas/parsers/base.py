"""Parser protocol + the trivial pass-through implementation.

A :class:`Parser` takes a :class:`RawDocument` and returns clean prose
text suitable for an :class:`midas.extractors.base.Extractor`. The
:class:`PassthroughParser` is the right answer for sources that already
deliver prose (the IR-press scraper, the RSS reader) — it just decodes
``content_bytes`` as UTF-8 with errors replaced.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from midas.sources.base import RawDocument


class ParserError(Exception):
    """Raised when a document can't be parsed (malformed, empty, etc.)."""


@runtime_checkable
class Parser(Protocol):
    """Strategy interface for content-type-specific cleaning."""

    name: str

    def parse(self, raw: RawDocument) -> str:
        """Return cleaned prose text. Empty string if nothing extractable."""


class PassthroughParser:
    """No-op parser for sources that already deliver UTF-8 prose.

    Used by the IR-press scraper and the RSS reader, both of which run
    their own body-extraction step at fetch time and stash the cleaned
    text directly into ``content_bytes``.
    """

    name = "passthrough"

    def parse(self, raw: RawDocument) -> str:
        return raw.content_bytes.decode("utf-8", errors="replace")
