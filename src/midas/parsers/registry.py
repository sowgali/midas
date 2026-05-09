"""Parser selection by source type.

The pipeline consults :func:`select_parser` to get the right cleaner
for a given :class:`RawDocument`. SEC forms (10-K, 10-Q, 8-K, earnings
calls — all delivered as Inline XBRL HTML by SEC) go through the XBRL
parser; press releases / blog posts / news ship pre-cleaned UTF-8 prose
from their adapters and go through the pass-through.
"""

from __future__ import annotations

from midas.models.types import SourceType
from midas.sources.base import RawDocument

from .base import Parser, PassthroughParser
from .xbrl_html import XbrlHtmlParser

_XBRL_HTML_SOURCES: frozenset[SourceType] = frozenset(
    {
        SourceType.FORM_10K,
        SourceType.FORM_10Q,
        SourceType.FORM_8K,
        SourceType.EARNINGS_CALL,
    }
)


def select_parser(raw: RawDocument) -> Parser:
    """Return the parser appropriate for ``raw``'s source type."""
    if raw.source_type in _XBRL_HTML_SOURCES:
        return XbrlHtmlParser()
    return PassthroughParser()
