"""Enumerations shared across the data model.

All of these are :class:`enum.StrEnum` so they round-trip cleanly through
JSON, log lines, and SQL ``VARCHAR`` columns without manual conversion.
"""

from __future__ import annotations

from enum import StrEnum


class EntityType(StrEnum):
    """What kind of money-handling actor an :class:`Entity` represents."""

    PUBLIC_COMPANY = "public_company"
    PRIVATE_COMPANY = "private_company"
    FUND = "fund"
    GOVERNMENT = "government"
    NONPROFIT = "nonprofit"


class SourceType(StrEnum):
    """The kind of document a :class:`Source` is.

    SEC form codes use their canonical hyphenated names (``10-K``, etc.) so
    they can be compared directly with EDGAR responses.
    """

    FORM_10K = "10-K"
    FORM_10Q = "10-Q"
    FORM_8K = "8-K"
    PRESS_RELEASE = "press_release"
    BLOG = "blog"
    NEWS = "news"
    EARNINGS_CALL = "earnings_call"


class DealType(StrEnum):
    """The economic shape of a :class:`Deal`."""

    INVESTMENT = "investment"
    ACQUISITION = "acquisition"
    COMMERCIAL_CONTRACT = "commercial_contract"
    PARTNERSHIP = "partnership"
    LICENSING = "licensing"
    DEBT = "debt"
    GRANT = "grant"


class DealStatus(StrEnum):
    """Lifecycle state of a :class:`Deal`."""

    ANNOUNCED = "announced"
    CLOSED = "closed"
    RUMORED = "rumored"
    TERMINATED = "terminated"
