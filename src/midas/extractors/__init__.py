"""Extractors: text → ExtractedDeal candidates.

The pipeline composes :class:`RegexExtractor` (cheap, deterministic
first pass) with :class:`ClaudeExtractor` (LLM long-tail) and dedupes
their outputs at the normalizer layer. New extractors slot in by
implementing the :class:`Extractor` Protocol from :mod:`.base`.
"""

from .base import ExtractedDeal, ExtractionContext, Extractor, KnownParty
from .claude import ClaudeExtractor
from .regex import RegexExtractor

__all__ = [
    "ClaudeExtractor",
    "ExtractedDeal",
    "ExtractionContext",
    "Extractor",
    "KnownParty",
    "RegexExtractor",
]
