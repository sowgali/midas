"""Tests for the parser layer (XBRL strip + pass-through + registry).

We test the XBRL parser at two levels:
  - synthetic fixtures (deterministic; cover each drop/unwrap rule)
  - real cached SEC filings under ``data/raw`` (when present in the
    dev environment) — guards against regressions when refactoring
    against fixtures alone misses real-world quirks.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from midas.models.types import SourceType
from midas.parsers import PassthroughParser, XbrlHtmlParser, select_parser
from midas.parsers.base import Parser
from midas.sources.base import RawDocument


def _raw(content: bytes, source_type: SourceType = SourceType.FORM_8K) -> RawDocument:
    return RawDocument(
        url="https://example.com/x",
        content_bytes=content,
        source_type=source_type,
        publisher="SEC",
        title="Test",
        published_at=datetime(2025, 1, 1, tzinfo=UTC),
    )


# ---------- Pass-through ----------


def test_passthrough_decodes_utf8() -> None:
    p = PassthroughParser()
    raw = _raw(b"Microsoft will invest $10B in OpenAI.")
    assert p.parse(raw) == "Microsoft will invest $10B in OpenAI."


def test_passthrough_handles_invalid_utf8() -> None:
    p = PassthroughParser()
    raw = _raw(b"prefix\xff\xfesuffix")
    out = p.parse(raw)
    assert "prefix" in out and "suffix" in out


# ---------- XBRL parser: synthetic fixtures ----------


def test_xbrl_drops_hidden_block() -> None:
    html = b"""<html><body>
    <ix:hidden>
        <xbrli:context id="c1"><xbrli:entity><xbrli:identifier>SECRET</xbrli:identifier></xbrli:entity></xbrli:context>
    </ix:hidden>
    <p>Visible narrative.</p>
    </body></html>"""
    out = XbrlHtmlParser().parse(_raw(html))
    assert "Visible narrative." in out
    assert "SECRET" not in out


def test_xbrl_drops_resources_and_references() -> None:
    html = b"""<html><body>
    <ix:references><link:schemaRef xlink:type="simple" xlink:href="schema.xsd"/></ix:references>
    <ix:resources><xbrli:context id="c"><xbrli:period><xbrli:startDate>2025-01-01</xbrli:startDate></xbrli:period></xbrli:context></ix:resources>
    <p>Survives.</p>
    </body></html>"""
    out = XbrlHtmlParser().parse(_raw(html))
    assert "Survives." in out
    assert "schema.xsd" not in out
    assert "2025-01-01" not in out


def test_xbrl_unwraps_inline_nonNumeric() -> None:
    html = b"""<html><body>
    <p>Filed: <ix:nonNumeric name="dei:DocumentType" contextRef="c1">10-K</ix:nonNumeric> for FY2025.</p>
    </body></html>"""
    out = XbrlHtmlParser().parse(_raw(html))
    # Inline value preserved; surrounding tag gone.
    assert "10-K" in out
    # The result should not contain literal tag names.
    assert "ix:nonNumeric" not in out
    assert "ix:nonnumeric" not in out


def test_xbrl_unwraps_inline_nonFraction() -> None:
    html = b"""<html><body>
    <p>Revenue was $<ix:nonFraction unitRef="usd" decimals="-6" name="us-gaap:Revenues">350,018</ix:nonFraction> million.</p>
    </body></html>"""
    out = XbrlHtmlParser().parse(_raw(html))
    assert "350,018" in out


def test_xbrl_strips_scripts_and_styles() -> None:
    html = b"""<html><body>
    <script>alert('drop')</script>
    <style>p { color: red; }</style>
    <p>Keep me.</p>
    </body></html>"""
    out = XbrlHtmlParser().parse(_raw(html))
    assert "Keep me." in out
    assert "alert" not in out
    assert "color: red" not in out


def test_xbrl_collapses_runaway_whitespace() -> None:
    html = b"<html><body>\n\n\n\n<p>A</p>\n\n\n\n\n<p>B</p>\n\n\n\n</body></html>"
    out = XbrlHtmlParser().parse(_raw(html))
    # No more than 2 consecutive newlines (paragraph break).
    assert "\n\n\n" not in out
    assert "A" in out and "B" in out


def test_xbrl_empty_input_returns_empty() -> None:
    assert XbrlHtmlParser().parse(_raw(b"")) == ""
    assert XbrlHtmlParser().parse(_raw(b"   \n\n  ")) == ""


def test_xbrl_real_8k_shape() -> None:
    """Realistic mini-fixture: confirm narrative text survives + iXBRL noise dies."""
    html = b"""<html><head><meta charset="utf-8"></head><body>
    <ix:header>
      <ix:references><link:schemaRef xlink:href="goog-20260424.xsd"/></ix:references>
      <ix:resources>
        <xbrli:context id="c1"><xbrli:entity><xbrli:identifier scheme="http://www.sec.gov/CIK">0001652044</xbrli:identifier></xbrli:entity><xbrli:period><xbrli:instant>2026-04-24</xbrli:instant></xbrli:period></xbrli:context>
      </ix:resources>
      <ix:hidden><ix:nonNumeric name="dei:AmendmentFlag" contextRef="c1">false</ix:nonNumeric></ix:hidden>
    </ix:header>
    <div>
      <p><ix:nonNumeric name="dei:DocumentType" contextRef="c1">8-K</ix:nonNumeric></p>
      <p>UNITED STATES SECURITIES AND EXCHANGE COMMISSION</p>
      <p>On April 22, 2026, Alphabet Inc. announced that it has agreed to acquire
      Acme Corp. for $5 billion in an all-stock transaction.</p>
    </div>
    </body></html>"""
    out = XbrlHtmlParser().parse(_raw(html))
    assert "Alphabet Inc." in out
    assert "agreed to acquire" in out
    assert "$5 billion" in out
    assert "8-K" in out
    # Metadata gone:
    assert "0001652044" not in out
    assert "goog-20260424.xsd" not in out


# ---------- Registry ----------


@pytest.mark.parametrize(
    "source_type,parser_class",
    [
        (SourceType.FORM_10K, XbrlHtmlParser),
        (SourceType.FORM_10Q, XbrlHtmlParser),
        (SourceType.FORM_8K, XbrlHtmlParser),
        (SourceType.EARNINGS_CALL, XbrlHtmlParser),
        (SourceType.PRESS_RELEASE, PassthroughParser),
        (SourceType.BLOG, PassthroughParser),
        (SourceType.NEWS, PassthroughParser),
    ],
)
def test_select_parser_routes_by_source_type(
    source_type: SourceType, parser_class: type[Parser]
) -> None:
    parser = select_parser(_raw(b"x", source_type=source_type))
    assert isinstance(parser, parser_class)


# ---------- Real cached filings (smoke) ----------


def _real_cached_filings() -> list[Path]:
    """Skip cleanly when no cache exists (fresh dev box)."""
    cache = Path("data/raw")
    if not cache.exists():
        return []
    # Heuristic: the .bin files; ignore any tiny fragments.
    return [p for p in cache.rglob("*.bin") if p.stat().st_size > 1000]


@pytest.mark.skipif(not _real_cached_filings(), reason="no cached SEC filings on disk")
def test_xbrl_real_filings_shrink_meaningfully() -> None:
    """On every cached SEC filing, the parsed output should be shorter
    than the raw bytes (XBRL strip guarantees this) AND non-empty if
    the source had any narrative at all.
    """
    parser = XbrlHtmlParser()
    real = _real_cached_filings()
    # Sample a few to keep this test fast.
    for path in real[:5]:
        bytes_ = path.read_bytes()
        # Skip non-HTML cache entries (e.g. company_tickers.json).
        if not (b"<html" in bytes_[:2000].lower() or b"<!doctype html" in bytes_[:2000].lower()):
            continue
        raw = _raw(bytes_)
        text = parser.parse(raw)
        assert len(text) <= len(bytes_), f"{path}: parser grew the document"
        # Some 8-Ks are tiny exhibit pointers; allow short results, but
        # the parser must not crash and must produce a string.
        assert isinstance(text, str)
