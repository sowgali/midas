"""Tests for the SEC EDGAR adapter.

Network is fully mocked with respx. Cache lives under ``tmp_path``.
"""

from __future__ import annotations

import hashlib
from datetime import date
from pathlib import Path

import httpx
import pytest
import respx

from midas.models.types import SourceType
from midas.sources.http_client import HttpClient
from midas.sources.sec_edgar import FilingMetadata, SecEdgar

_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"


def _tickers_fixture() -> dict[str, dict[str, object]]:
    # SEC's actual schema: dict keyed by string ints, each row {cik_str, ticker, title}
    return {
        "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
        "1": {"cik_str": 789019, "ticker": "MSFT", "title": "Microsoft Corporation"},
    }


def _submissions_fixture() -> dict[str, object]:
    return {
        "cik": "320193",
        "name": "Apple Inc.",
        "filings": {
            "recent": {
                "form": ["10-K", "10-Q", "10-K", "8-K"],
                "filingDate": ["2024-11-01", "2024-08-02", "2023-11-03", "2024-05-03"],
                "reportDate": ["2024-09-28", "2024-06-29", "2023-09-30", ""],
                "accessionNumber": [
                    "0000320193-24-000123",
                    "0000320193-24-000080",
                    "0000320193-23-000106",
                    "0000320193-24-000060",
                ],
                "primaryDocument": [
                    "aapl-20240928.htm",
                    "aapl-20240629.htm",
                    "aapl-20230930.htm",
                    "aapl-8k.htm",
                ],
            }
        },
    }


@pytest.mark.asyncio
async def test_get_cik_returns_zero_padded(tmp_path: Path) -> None:
    with respx.mock(assert_all_called=False) as mock:
        mock.get(_TICKERS_URL).mock(return_value=httpx.Response(200, json=_tickers_fixture()))
        async with HttpClient(cache_dir=tmp_path, rate_per_sec=8.0) as client:
            sec = SecEdgar(http=client)
            cik = await sec.get_cik("AAPL")

    assert cik == "0000320193"


@pytest.mark.asyncio
async def test_get_cik_unknown_ticker(tmp_path: Path) -> None:
    with respx.mock(assert_all_called=False) as mock:
        mock.get(_TICKERS_URL).mock(return_value=httpx.Response(200, json=_tickers_fixture()))
        async with HttpClient(cache_dir=tmp_path, rate_per_sec=8.0) as client:
            sec = SecEdgar(http=client)
            cik = await sec.get_cik("NOPE")
    assert cik is None


@pytest.mark.asyncio
async def test_list_filings_filters_by_form_and_date(tmp_path: Path) -> None:
    cik = "0000320193"
    submissions_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    with respx.mock(assert_all_called=False) as mock:
        mock.get(submissions_url).mock(
            return_value=httpx.Response(200, json=_submissions_fixture())
        )
        async with HttpClient(cache_dir=tmp_path, rate_per_sec=8.0) as client:
            sec = SecEdgar(http=client)
            filings = await sec.list_filings(cik, forms=["10-K"], since=date(2024, 1, 1))

    assert len(filings) == 1
    f = filings[0]
    assert f.form == "10-K"
    assert f.filed_at == date(2024, 11, 1)
    assert f.accession_number == "0000320193-24-000123"
    assert f.primary_document == "aapl-20240928.htm"
    assert f.cik == cik
    assert f.report_date == date(2024, 9, 28)


@pytest.mark.asyncio
async def test_list_filings_no_filters_returns_all(tmp_path: Path) -> None:
    cik = "0000320193"
    submissions_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    with respx.mock(assert_all_called=False) as mock:
        mock.get(submissions_url).mock(
            return_value=httpx.Response(200, json=_submissions_fixture())
        )
        async with HttpClient(cache_dir=tmp_path, rate_per_sec=8.0) as client:
            sec = SecEdgar(http=client)
            filings = await sec.list_filings(cik)

    assert len(filings) == 4
    # the 8-K row has empty reportDate
    eightk = next(f for f in filings if f.form == "8-K")
    assert eightk.report_date is None


@pytest.mark.asyncio
async def test_fetch_filing_builds_correct_url_and_hashes_content(
    tmp_path: Path,
) -> None:
    meta = FilingMetadata(
        cik="0000320193",
        accession_number="0000320193-23-000064",
        form="10-K",
        filed_at=date(2023, 11, 3),
        report_date=date(2023, 9, 30),
        primary_document="aapl-20230930.htm",
    )
    expected_url = (
        "https://www.sec.gov/Archives/edgar/data/320193/000032019323000064/aapl-20230930.htm"
    )
    body = b"<html><body>10-K body</body></html>"

    with respx.mock(assert_all_called=False) as mock:
        route = mock.get(expected_url).mock(
            return_value=httpx.Response(200, content=body, headers={"content-type": "text/html"})
        )
        async with HttpClient(cache_dir=tmp_path, rate_per_sec=8.0) as client:
            sec = SecEdgar(http=client)
            doc = await sec.fetch_filing(meta)

    assert route.called
    assert route.calls.last.request.url == expected_url
    assert doc.url == expected_url
    assert doc.content_bytes == body
    assert doc.content_sha256 == hashlib.sha256(body).hexdigest()
    assert doc.source_type is SourceType.FORM_10K
    assert doc.publisher == "SEC"
    assert doc.title == "10-K 0000320193-23-000064"
    assert doc.fetched_at.tzinfo is not None


@pytest.mark.asyncio
async def test_filing_metadata_zero_pads_cik() -> None:
    meta = FilingMetadata(
        cik="320193",
        accession_number="0000320193-23-000064",
        form="10-K",
        filed_at=date(2023, 11, 3),
        primary_document="aapl-20230930.htm",
    )
    assert meta.cik == "0000320193"
