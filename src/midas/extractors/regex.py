"""Regex-based deal extractor.

Cheap, deterministic first pass over the document text. Catches the
canonical deal shapes that show up verbatim in press-release language:

* ``"<X> will invest $N (billion|million) in <Y>"`` → INVESTMENT
* ``"<X> invested $N in <Y>"`` → INVESTMENT
* ``"<X> acquired <Y> for $N"`` → ACQUISITION
* ``"<X> announced a K-year, $N contract with <Y>"`` → COMMERCIAL_CONTRACT

Anything more nuanced (partnerships without a clear payer, multi-clause
sentences, indirect references) is intentionally left for the LLM
extractor — better to under-fire here than to over-fire and pollute
downstream dedup.
"""

from __future__ import annotations

import re
from decimal import Decimal

from midas.models.types import DealStatus, DealType

from .base import ExtractedDeal, ExtractionContext, KnownParty

# ---------- Money parsing ----------

# Captures the numeric body and an optional scale suffix (billion/million/B/M).
# Designed to match inside a larger pattern, not anchored.
_AMOUNT_RE = re.compile(
    r"\$\s*(?P<num>[\d,]+(?:\.\d+)?)\s*(?P<scale>billion|million|B|M)?",
    re.IGNORECASE,
)

_SCALE_MULTIPLIERS: dict[str, Decimal] = {
    "billion": Decimal("1000000000"),
    "b": Decimal("1000000000"),
    "million": Decimal("1000000"),
    "m": Decimal("1000000"),
}


def _parse_amount(text: str) -> Decimal:
    """Parse a money string like ``"$1.5 billion"`` or ``"$1,000,000"``.

    Always returns a Decimal — float math on currency is a footgun we
    refuse to ship. Raises :class:`ValueError` if no recognizable amount
    is present so callers fail loudly rather than silently dropping
    matches.
    """
    match = _AMOUNT_RE.search(text)
    if match is None:
        raise ValueError(f"no amount found in {text!r}")
    raw = match.group("num").replace(",", "")
    base = Decimal(raw)
    scale = match.group("scale")
    if scale is None:
        return base
    return base * _SCALE_MULTIPLIERS[scale.lower()]


# ---------- Deal patterns ----------

# Each pattern produces named groups: ``source``, ``target``, ``amount``.
# The ``amount`` group is fed straight to ``_parse_amount`` (it captures
# the whole "$N (billion|million)?" run). Patterns are anchored loosely —
# we want them to fire mid-sentence, not just at line starts.
#
# The ``\b`` boundaries matter: without them ``acquired`` happily matches
# inside ``reacquired`` and we get spurious ACQUISITION deals.

_PATTERNS: list[tuple[re.Pattern[str], DealType]] = [
    # "Microsoft will invest $10 billion in OpenAI."
    # "Microsoft plans to invest $10 billion in OpenAI."
    (
        re.compile(
            r"(?P<source>[\w&.\- ]+?)\s+(?:will|plans to|is planning to|intends to)?\s*"
            r"\binvest\b\s+(?P<amount>\$\s*[\d,]+(?:\.\d+)?\s*(?:billion|million|B|M)?)\s+"
            r"in\s+(?P<target>[\w&.\- ]+?)(?=[.,;]|\s+(?:and|to|on|in|for)\b|$)",
            re.IGNORECASE,
        ),
        DealType.INVESTMENT,
    ),
    # "Microsoft invested $10 billion in OpenAI."
    (
        re.compile(
            r"(?P<source>[\w&.\- ]+?)\s+\binvested\b\s+"
            r"(?P<amount>\$\s*[\d,]+(?:\.\d+)?\s*(?:billion|million|B|M)?)\s+"
            r"in\s+(?P<target>[\w&.\- ]+?)(?=[.,;]|\s+(?:and|to|on|in|for)\b|$)",
            re.IGNORECASE,
        ),
        DealType.INVESTMENT,
    ),
    # "Google acquired DeepMind for $500 million."
    (
        re.compile(
            r"(?P<source>[\w&.\- ]+?)\s+\bacquired\b\s+(?P<target>[\w&.\- ]+?)\s+"
            r"for\s+(?P<amount>\$\s*[\d,]+(?:\.\d+)?\s*(?:billion|million|B|M)?)",
            re.IGNORECASE,
        ),
        DealType.ACQUISITION,
    ),
    # "Anthropic announced a 5-year, $4 billion compute contract with Amazon."
    # source spends → target is the *counterparty*. Following our directional
    # convention (money flows source→target), we treat the announcer as the
    # payer here only when the next clause names a counterparty after "with".
    # That's the press-release shape we see most often.
    (
        re.compile(
            r"(?P<source>[\w&.\- ]+?)\s+(?:announced|signed)\s+(?:a\s+)?"
            r"\d+-year,?\s*(?P<amount>\$\s*[\d,]+(?:\.\d+)?\s*(?:billion|million|B|M)?)\s+"
            r"(?:[\w\- ]+?\s+)?contract\s+with\s+(?P<target>[\w&.\- ]+?)(?=[.,;]|$)",
            re.IGNORECASE,
        ),
        DealType.COMMERCIAL_CONTRACT,
    ),
]


# ---------- Party resolution ----------


def _resolve_party(raw_name: str, parties: list[KnownParty]) -> tuple[str, bool] | None:
    """Map a free-text party reference to a known canonical name.

    Returns ``(canonical_name, exact_match)`` or ``None`` if no match.
    ``exact_match`` is True when the raw text equals the canonical name
    or an alias case-insensitively (modulo surrounding whitespace);
    False when matched only as a substring. Confidence scoring uses
    that distinction — substring matches are weaker signal.
    """
    needle = raw_name.strip().lower()
    if not needle:
        return None

    # First pass: exact (case-insensitive) match against canonical or alias.
    for party in parties:
        names = [party.canonical_name, *party.aliases]
        for name in names:
            if needle == name.lower():
                return party.canonical_name, True

    # Second pass: substring match in either direction. The needle
    # appears inside a known name ("Google" inside "Google LLC") or vice
    # versa ("Microsoft Corp." contains the alias "Microsoft").
    for party in parties:
        names = [party.canonical_name, *party.aliases]
        for name in names:
            n = name.lower()
            if needle in n or n in needle:
                return party.canonical_name, False

    return None


# ---------- Extractor ----------


class RegexExtractor:
    """Pattern-matching extractor for the easy 80%.

    Conforms to :class:`midas.extractors.base.Extractor`. ``async`` is
    free here (no I/O, no awaits) — the interface is async because the
    LLM extractor needs it, and uniformity beats branching at the
    pipeline layer.
    """

    name: str = "regex"

    async def extract(self, context: ExtractionContext) -> list[ExtractedDeal]:
        results: list[ExtractedDeal] = []
        text = context.document_text

        for pattern, deal_type in _PATTERNS:
            for match in pattern.finditer(text):
                source_raw = match.group("source")
                target_raw = match.group("target")
                amount_raw = match.group("amount")

                source_resolved = _resolve_party(source_raw, context.known_parties)
                target_resolved = _resolve_party(target_raw, context.known_parties)
                if source_resolved is None or target_resolved is None:
                    # Drop matches we can't ground in a known entity. The
                    # LLM extractor handles long-tail name resolution.
                    continue

                source_name, source_exact = source_resolved
                target_name, target_exact = target_resolved

                try:
                    amount = _parse_amount(amount_raw)
                except ValueError:
                    continue

                # Both sides exact → 0.7. Either side substring → 0.6.
                # The boundary is deliberately stark: substring matches
                # are noisier and we'd rather have the LLM corroborate
                # them than promote them on regex alone.
                confidence = 0.7 if (source_exact and target_exact) else 0.6

                snippet = match.group(0)
                results.append(
                    ExtractedDeal(
                        source_party_name=source_name,
                        target_party_name=target_name,
                        deal_type=deal_type,
                        status=DealStatus.ANNOUNCED,
                        amount_usd=amount,
                        amount_native=amount,
                        currency="USD",
                        announced_at=None,
                        closes_at=None,
                        confidence=confidence,
                        description=snippet.strip(),
                        evidence_text_snippet=snippet,
                        char_start=match.start(),
                        char_end=match.end(),
                        extractor_name=self.name,
                    )
                )

        return results
