"""SEC EDGAR adapter.

Three operations make up the EDGAR vertical:

1. :meth:`SecEdgar.get_cik` — ticker -> 10-char zero-padded CIK string,
   via the cached ``company_tickers.json`` directory file.
2. :meth:`SecEdgar.list_filings` — submissions index for one CIK,
   filtered by form code and ``filed_at >=`` date.
3. :meth:`SecEdgar.fetch_filing` — pull the primary document for one
   filing as a :class:`RawDocument`.

Endpoints (per SEC fair-use policy, all under our shared rate limit):

* ``https://www.sec.gov/files/company_tickers.json``
* ``https://data.sec.gov/submissions/CIK{cik:010d}.json``
* ``https://www.sec.gov/Archives/edgar/data/{int_cik}/{accession_no_dashes}/{primary}``
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator

from ..models.types import SourceType
from .base import RawDocument, Source
from .http_client import HttpClient, get_default_client

_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_SUBMISSIONS_URL_TEMPLATE = "https://data.sec.gov/submissions/CIK{cik}.json"
_ARCHIVE_URL_TEMPLATE = (
    "https://www.sec.gov/Archives/edgar/data/{int_cik}/{accession_no_dashes}/{primary}"
)


# Form-code -> SourceType mapping. We keep this explicit (rather than
# parsing the StrEnum value) so adding ``10-K/A`` etc. later is a
# deliberate decision, not an accidental enum miss.
_FORM_TO_SOURCE_TYPE: dict[str, SourceType] = {
    "10-K": SourceType.FORM_10K,
    "10-Q": SourceType.FORM_10Q,
    "8-K": SourceType.FORM_8K,
}


class FilingMetadata(BaseModel):
    """One row from the submissions index for a given CIK."""

    cik: str = Field(description="10-char zero-padded CIK string.")
    accession_number: str = Field(
        description="EDGAR accession with dashes, e.g. '0000320193-23-000064'.",
    )
    form: str = Field(description="SEC form code, e.g. '10-K'.")
    filed_at: date
    report_date: date | None = None
    primary_document: str = Field(
        description="Primary document filename within the filing archive.",
    )

    @field_validator("cik")
    @classmethod
    def _zero_pad_cik(cls, v: str) -> str:
        v = v.strip()
        if not v.isdigit():
            raise ValueError(f"cik must be all digits, got {v!r}")
        return v.zfill(10)


def _strip_accession_dashes(accession: str) -> str:
    return accession.replace("-", "")


class SecEdgar(Source):
    """SEC EDGAR source adapter.

    Lightweight: holds a reference to an :class:`HttpClient` and translates
    EDGAR's parallel-array JSON into typed records. The client owns the
    cache + rate limit, so a second call for the same URL is free.
    """

    def __init__(self, http: HttpClient | None = None) -> None:
        self._http = http
        self.publisher = "SEC"

    async def _client(self) -> HttpClient:
        if self._http is not None:
            return self._http
        return await get_default_client()

    # ---------- ticker -> CIK ----------

    async def get_cik(self, ticker: str) -> str | None:
        """Return the 10-char zero-padded CIK for ``ticker`` or ``None``.

        The tickers directory is small and updates rarely; the on-disk cache
        in :class:`HttpClient` keeps this to one network call per process.
        """
        ticker = ticker.upper().strip()
        client = await self._client()
        data: Any = await client.get_json(_TICKERS_URL)

        # The file is a dict-of-rows keyed by stringified ints. We don't
        # rely on the keys; we just iterate values.
        if isinstance(data, dict):
            rows: list[Any] = list(data.values())
        elif isinstance(data, list):
            rows = data
        else:
            return None

        for row in rows:
            if not isinstance(row, dict):
                continue
            row_ticker = str(row.get("ticker", "")).upper()
            if row_ticker == ticker:
                cik_int = int(row["cik_str"])
                return f"{cik_int:010d}"
        return None

    # ---------- list filings ----------

    async def list_filings(
        self,
        cik: str,
        *,
        forms: list[str] | None = None,
        since: date | None = None,
    ) -> list[FilingMetadata]:
        """Return the recent filings for ``cik``, filtered by form / date.

        ``cik`` must already be 10-char zero-padded (use :meth:`get_cik`).
        """
        if not cik.isdigit() or len(cik) != 10:
            raise ValueError(f"cik must be 10-digit zero-padded, got {cik!r}")

        client = await self._client()
        url = _SUBMISSIONS_URL_TEMPLATE.format(cik=cik)
        data: Any = await client.get_json(url)

        recent = data.get("filings", {}).get("recent", {}) if isinstance(data, dict) else {}
        forms_arr: list[Any] = list(recent.get("form", []))
        filed_arr: list[Any] = list(recent.get("filingDate", []))
        report_arr: list[Any] = list(recent.get("reportDate", []))
        accession_arr: list[Any] = list(recent.get("accessionNumber", []))
        primary_arr: list[Any] = list(recent.get("primaryDocument", []))

        n = min(
            len(forms_arr),
            len(filed_arr),
            len(accession_arr),
            len(primary_arr),
        )

        forms_filter = {f for f in forms} if forms else None
        out: list[FilingMetadata] = []
        for i in range(n):
            form = str(forms_arr[i])
            if forms_filter is not None and form not in forms_filter:
                continue
            try:
                filed_at = date.fromisoformat(str(filed_arr[i]))
            except ValueError:
                continue
            if since is not None and filed_at < since:
                continue

            report_date: date | None = None
            if i < len(report_arr) and report_arr[i]:
                try:
                    report_date = date.fromisoformat(str(report_arr[i]))
                except ValueError:
                    report_date = None

            out.append(
                FilingMetadata(
                    cik=cik,
                    accession_number=str(accession_arr[i]),
                    form=form,
                    filed_at=filed_at,
                    report_date=report_date,
                    primary_document=str(primary_arr[i]),
                )
            )
        return out

    # ---------- fetch one filing ----------

    async def fetch_filing(self, meta: FilingMetadata) -> RawDocument:
        """Fetch the primary document of one filing and wrap it in a RawDocument."""
        int_cik = int(meta.cik)
        url = _ARCHIVE_URL_TEMPLATE.format(
            int_cik=int_cik,
            accession_no_dashes=_strip_accession_dashes(meta.accession_number),
            primary=meta.primary_document,
        )
        client = await self._client()
        body = await client.get_bytes(url)

        source_type = _FORM_TO_SOURCE_TYPE.get(meta.form)
        if source_type is None:
            # Unknown form: persist as 8-K-equivalent "miscellaneous EDGAR
            # filing" would be wrong; surface it as an error so callers add
            # a mapping deliberately.
            raise ValueError(
                f"No SourceType mapping for SEC form {meta.form!r}; add it to _FORM_TO_SOURCE_TYPE."
            )

        published_at: datetime | None = (
            datetime.combine(meta.filed_at, datetime.min.time()) if meta.filed_at else None
        )

        return RawDocument(
            url=url,
            content_bytes=body,
            source_type=source_type,
            publisher=self.publisher,
            title=f"{meta.form} {meta.accession_number}",
            published_at=published_at,
        )

    # ---------- Source ABC ----------

    async def fetch(self, *args: Any, **kwargs: Any) -> RawDocument:
        """Adapter for the :class:`Source` ABC; delegates to :meth:`fetch_filing`.

        Accepts a single positional ``FilingMetadata`` (or ``meta=`` kwarg)
        for compatibility with the abstract signature in :class:`Source`.
        """
        meta = args[0] if args else kwargs["meta"]
        if not isinstance(meta, FilingMetadata):
            raise TypeError(f"expected FilingMetadata, got {type(meta).__name__}")
        return await self.fetch_filing(meta)
