"""Inline-XBRL + HTML cleaner for SEC filings.

SEC primary documents (10-K, 10-Q, 8-K) are Inline XBRL: HTML5 with
namespaced tags carrying machine-readable financial facts inline with
the human-readable narrative. Raw decoded bytes can run **>1M tokens**
on a 10-K — well over Claude's context window — and most of that is
``<ix:hidden>``/``<xbrli:context>`` metadata blocks and tag soup that
adds zero signal for deal extraction.

This parser:

1. **Drops** subtrees rooted at metadata-only iXBRL elements
   (``ix:hidden``, ``ix:references``, ``ix:resources``, ``ix:header``,
   ``xbrli:*``, ``link:schemaref``, ``link:linkbaseref``).
2. **Drops** standard noise tags (``script``, ``style``, ``nav``,
   ``header``, ``footer``, ``form``, ``svg``).
3. **Unwraps** text-bearing iXBRL elements (``ix:nonnumeric``,
   ``ix:nonfraction``, ``ix:fraction``, ``ix:numerator``,
   ``ix:denominator``) — keep the rendered value, drop the tag.
4. **Extracts** the body's text with a separator so paragraph
   structure survives roughly.
5. **Normalizes** runaway whitespace.

Empirical: on Google's FY2025 10-K (~5 MB raw, ~1.18M tokens) this
brings the LLM input below the 200K-token mark while preserving
MD&A, Notes, and Item 1 narrative.
"""

from __future__ import annotations

import re

from selectolax.parser import HTMLParser, Node

from midas.sources.base import RawDocument

from .base import ParserError

# Subtrees to drop wholesale (metadata only — no narrative text).
_DROP_SUBTREE: frozenset[str] = frozenset(
    {
        # Inline XBRL metadata containers
        "ix:hidden",
        "ix:references",
        "ix:resources",
        "ix:header",
        # XBRL contexts/units (always machine-only)
        "xbrli:context",
        "xbrli:unit",
        "xbrli:identifier",
        "xbrli:entity",
        "xbrli:period",
        "xbrli:startdate",
        "xbrli:enddate",
        "xbrli:segment",
        "xbrli:scenario",
        "xbrldi:explicitmember",
        # XBRL linkbase references
        "link:schemaref",
        "link:linkbaseref",
        "link:reference",
        # HTML noise — present in some SEC primary docs (forms, scripts)
        "script",
        "style",
        "noscript",
        "nav",
        "header",
        "footer",
        "form",
        "svg",
    }
)

# Inline iXBRL elements that wrap a piece of text. Unwrap → keep text.
_UNWRAP_INLINE: frozenset[str] = frozenset(
    {
        "ix:nonnumeric",
        "ix:nonfraction",
        "ix:fraction",
        "ix:numerator",
        "ix:denominator",
        "ix:exclude",
        "ix:continuation",
    }
)


class XbrlHtmlParser:
    """Strip inline-XBRL noise and return narrative text."""

    name = "xbrl_html"

    def parse(self, raw: RawDocument) -> str:
        html = raw.content_bytes.decode("utf-8", errors="replace")
        if not html.strip():
            return ""

        try:
            tree = HTMLParser(html)
        except Exception as exc:  # pragma: no cover  — selectolax very rarely raises
            raise ParserError(f"failed to parse HTML: {exc}") from exc

        body = tree.body if tree.body is not None else tree.root
        if body is None:
            return ""

        # 1. Drop metadata subtrees. Collect first to avoid mutating during walk.
        to_drop: list[Node] = []
        for node in body.traverse():
            tag = (node.tag or "").lower()
            if tag in _DROP_SUBTREE:
                to_drop.append(node)
        for node in to_drop:
            node.decompose()

        # 2. Unwrap inline iXBRL elements: replace each tag with its text.
        # We do this by replacing the node with a text node carrying its
        # collected text — selectolax doesn't expose a true unwrap, so
        # we use replace_with.
        to_unwrap: list[Node] = []
        for node in body.traverse():
            tag = (node.tag or "").lower()
            if tag in _UNWRAP_INLINE:
                to_unwrap.append(node)
        for node in to_unwrap:
            inner = node.text(separator=" ", strip=False) or ""
            # Replace with the bare text. selectolax replace_with takes a
            # raw HTML string, so we escape minimally to keep the parser
            # happy and re-tokenize cleanly.
            node.replace_with(_escape_for_inline(inner))

        # 3. Get text. Default sep="" smashes paragraphs; use newline so
        # block-level structure survives the strip.
        text = body.text(separator="\n", strip=True)

        # 4. Normalize: collapse 3+ newlines to 2 (paragraph break) and
        # 2+ spaces within a line to 1.
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


def _escape_for_inline(text: str) -> str:
    """Escape just enough to keep selectolax from re-interpreting the text."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
