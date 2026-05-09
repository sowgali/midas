"""Parsers — raw bytes → clean prose text for the extractor layer.

Sits between the sources layer (fetches bytes, doesn't interpret) and the
extractors layer (operates on prose text, doesn't fetch). Different
source types ship different content — SEC filings come as Inline XBRL,
press releases as plain HTML, news as RSS-summarized HTML — and the
parser registry routes each to the right cleaner so the extractor
always sees the same shape.
"""

from .base import Parser, ParserError, PassthroughParser
from .registry import select_parser
from .xbrl_html import XbrlHtmlParser

__all__ = [
    "Parser",
    "ParserError",
    "PassthroughParser",
    "XbrlHtmlParser",
    "select_parser",
]
