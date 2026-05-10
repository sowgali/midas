"""V1.8 open-world entity resolution.

Pre-V1.8 the pipeline was **closed-world**: every extracted party name
had to match the seed registry or the deal got dropped on the floor.
That made the seed YAML the ceiling on graph coverage — every new
counterparty type (power utilities, REITs, networking vendors, foreign
labs, foundation models we'd never heard of) silently disappeared.

This module makes the resolver **open-world** with quality control:

1. **Exact match** (canonical name / alias, lowercase). Same as V1.
2. **Normalized match**: strip ", Inc.", " Corp", " LLC" and similar
   suffixes plus punctuation, lowercase. Catches "Microsoft" /
   "Microsoft Corp" / "Microsoft Corporation" without registry edits.
3. **Auto-create** as ``Entity(discovered=True)`` if the name is plausibly
   an entity (passes :func:`is_extractable_entity_name`). Creates a
   :class:`Entity` row, updates the resolver's in-memory cache, returns
   the new id. Subsequent resolves in the same run see it.

The :func:`is_extractable_entity_name` filter rejects pronouns, generic
referents ("the Company", "the Issuer"), and broad collective nouns
("bondholders", "shareholders"). Personal-name detection is deliberately
left to the human-review step (``midas review``) — false-positive risk
on heuristic detection is high.
"""

from __future__ import annotations

import re
import uuid
from collections.abc import Iterable

import structlog
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col, select

from midas.extractors.base import KnownParty
from midas.models import Entity, EntityType
from midas.storage.repository import EntityRepository

log = structlog.get_logger(__name__)


# ---------- Name normalization ----------

# Listed longest-first so "ltd." matches before "ltd". Each is stripped
# only when it appears as a *trailing* token after lowercase + punct
# strip. Conservative — corporate suffixes plus "Holdings" since that
# turns up frequently in sub-brand names.
_CORP_SUFFIXES: tuple[str, ...] = (
    "incorporated",
    "corporation",
    "holdings",
    "holding",
    "company",
    "limited",
    "group",
    "plc",
    "llc",
    "ltd",
    "lp",
    "inc",
    "corp",
    "co",
    "sa",
    "ag",
    "nv",
    "gmbh",
    "pbc",
)

# Strip both ASCII and smart quotes — Claude often returns the latter
# in extracted prose. Smart quotes spelled via \u escapes so ruff
# doesn't flag the regex as ambiguous (RUF001).
_PUNCT_RE = re.compile("[\u2018\u2019\u201c\u201d\"',.()]")
_WS_RE = re.compile(r"\s+")


def normalize_entity_name(name: str) -> str:
    """Return a canonical form for fuzzy matching.

    Lowercases, strips punctuation, collapses whitespace, and removes a
    single trailing corporate suffix (Inc / Corp / LLC / etc.).

    >>> normalize_entity_name("Microsoft Corporation")
    'microsoft'
    >>> normalize_entity_name("Wiz, Inc.")
    'wiz'
    >>> normalize_entity_name("Anthropic, PBC")
    'anthropic'
    >>> normalize_entity_name("Hugging Face")
    'hugging face'
    """
    s = _PUNCT_RE.sub(" ", name)
    s = _WS_RE.sub(" ", s).strip().lower()
    if not s:
        return s
    tokens = s.split(" ")
    if len(tokens) > 1 and tokens[-1] in _CORP_SUFFIXES:
        tokens = tokens[:-1]
    return " ".join(tokens)


# ---------- Quality filter ----------

# Strings that look like an entity name when read alone but aren't.
# Two layers: exact-string blacklist + structural patterns. Both
# learned from observed false positives in the V1.8 demo run on
# Hugging Face's blog corpus (e.g. "Undisclosed investors", "1st
# Place Winner", "EduRénov program (public school renovation projects)").

# A *first word* in this set means the whole string is a placeholder.
_PLACEHOLDER_PREFIX_TOKENS: frozenset[str] = frozenset(
    {
        "undisclosed",
        "unspecified",
        "unnamed",
        "anonymous",
        "various",  # "various investors"
    },
)

# A *last word* in this set means the string names a *role* or
# *aggregate* of people / orgs, not a counterparty.
_AGGREGATE_SUFFIX_TOKENS: frozenset[str] = frozenset(
    {
        # Competition placements
        "winner",
        "winners",
        "finalist",
        "finalists",
        "entrant",
        "entrants",
        "participant",
        "participants",
        "candidates",
        "candidate",
        "honoree",
        "honorees",
        "awardee",
        "awardees",
        "nominee",
        "nominees",
        "recipient",
        "recipients",
        # Authorship roles
        "authors",
        "author",
        "co-authors",
        "contributors",
        "contributor",
        "researchers",
        "researcher",
        # Geo / generic groupings
        "region",
        "respondents",
        "respondent",
        "attendees",
        "attendee",
        "sponsors",
        "sponsor",
    },
)


# Generic referents that the LLM occasionally emits as a "party" but
# which don't represent a real counterparty.
_GENERIC_NON_ENTITIES: frozenset[str] = frozenset(
    {
        # First/second/third-person referents
        "we",
        "us",
        "our",
        "ourselves",
        "they",
        "them",
        "themselves",
        "i",
        "you",
        # Self-references in filings
        "the company",
        "the corporation",
        "the issuer",
        "the registrant",
        "the parent",
        "the parent company",
        "the group",
        "the partnership",
        "the trust",
        "the fund",
        "the firm",
        "the business",
        # Generic markets / abstractions
        "the public",
        "the market",
        "the markets",
        "third parties",
        "third-party",
        "third party",
        "the industry",
        # NOTE: "the European Commission" / specific governments / agencies are
        # NOT in this stoplist — they're real counterparties (regulatory fines,
        # grants, procurement). Auto-create as discovered; human review can
        # reclassify entity_type=GOVERNMENT.
        # Aggregate investor classes
        "investors",
        "public investors",
        "bond investors",
        "bondholders",
        "bond holders",
        "noteholders",
        "shareholders",
        "stockholders",
        "creditors",
        "lenders",
        # Litigation generics
        "claimants",
        "plaintiffs",
        "litigants",
        "litigation plaintiffs",
        "settlement claimants",
        "privacy litigation plaintiffs",
        "privacy settlement claimants",
        # Generic supply/customer aggregates
        "customers",
        "the customer",
        "consumers",
        "vendors",
        "suppliers",
        "subsidiaries",
        "affiliates",
        # People / role aggregates
        "employees",
        "employee",
        "officers",
        "directors",
        "the board",
        "management",
        "competitors",
        "regulators",
    },
)


def is_extractable_entity_name(name: str) -> bool:
    """Heuristic: should this string auto-create a discovered entity?

    Rejects generic referents, pronouns, collective nouns, and obviously
    non-entity strings. Liberal: anything plausibly a real org passes.
    Borderline cases (executives' personal names, regulatory bodies) are
    deliberately allowed through and triaged at the human-review step.
    """
    s = name.strip()
    if len(s) < 3:
        return False
    if s.lower() in _GENERIC_NON_ENTITIES:
        return False
    # Must contain at least one alphabetic character.
    if not any(c.isalpha() for c in s):
        return False

    tokens = s.split()
    # Reject "Undisclosed *", "Unspecified *", "Various *".
    if tokens and tokens[0].lower() in _PLACEHOLDER_PREFIX_TOKENS:
        return False
    # Reject "* Winner(s)", "* Participants", "* Authors", "* Region".
    if tokens and tokens[-1].lower().rstrip(",.") in _AGGREGATE_SUFFIX_TOKENS:
        return False

    # All-lowercase single words are usually generic ("vendors",
    # "creditors"). Real company names have at least one capital, or
    # are multi-word, or include an acronym chunk.
    return not (s.islower() and " " not in s)


# ---------- Entity resolver ----------


class EntityResolver:
    """Case-insensitive name → :class:`Entity.id` lookup with open-world creation.

    Built once per ingest run from the registry; cheap to query.
    Subsequent ``resolve_or_create`` calls inside the same run mutate
    the in-memory caches so newly-discovered entities are visible to
    later extractions in the same loop.
    """

    def __init__(self, entities: Iterable[Entity]) -> None:
        self._by_name: dict[str, uuid.UUID] = {}
        self._by_norm: dict[str, uuid.UUID] = {}
        self._known_parties: list[KnownParty] = []
        for entity in entities:
            self._index(entity)

    def _index(self, entity: Entity) -> None:
        for raw_key in [entity.canonical_name, *entity.aliases]:
            self._index_key(entity, raw_key)
        self._known_parties.append(
            KnownParty(
                entity_id=entity.id,
                canonical_name=entity.canonical_name,
                aliases=list(entity.aliases),
            ),
        )

    def _index_key(self, entity: Entity, key: str) -> None:
        exact = key.strip().lower()
        if not exact:
            return
        if exact in self._by_name and self._by_name[exact] != entity.id:
            log.warning(
                "resolver.alias_collision",
                key=key,
                first_id=str(self._by_name[exact]),
                second_id=str(entity.id),
            )
        else:
            self._by_name[exact] = entity.id

        norm = normalize_entity_name(key)
        # Don't override an existing normalized-key mapping with a different
        # entity — first-wins. That preserves the seeded entity's primacy
        # if a discovered one shares a normalized form.
        if norm and norm not in self._by_norm:
            self._by_norm[norm] = entity.id

    def resolve(self, name: str) -> uuid.UUID | None:
        """Match ``name`` to a known entity by exact-lower or normalized form."""
        if not name:
            return None
        exact = name.strip().lower()
        if exact in self._by_name:
            return self._by_name[exact]
        norm = normalize_entity_name(name)
        if norm and norm in self._by_norm:
            return self._by_norm[norm]
        return None

    async def resolve_or_create(
        self,
        session: AsyncSession,
        name: str,
        *,
        entity_type: EntityType = EntityType.PRIVATE_COMPANY,
    ) -> uuid.UUID | None:
        """Resolve ``name``; if no match and the name passes the quality
        filter, insert ``Entity(discovered=True)`` and return its id.

        Returns ``None`` only when the name fails the quality filter
        (so the caller can record a discarded extraction). The caller
        is responsible for committing the surrounding transaction.
        """
        existing = self.resolve(name)
        if existing is not None:
            return existing

        if not is_extractable_entity_name(name):
            log.debug("resolver.filter.rejected", name=name)
            return None

        canonical_name = name.strip()
        new_entity = Entity(
            canonical_name=canonical_name,
            aliases=[],
            entity_type=entity_type,
            sector_tags=[],
            discovered=True,
        )
        await EntityRepository(session).add(new_entity)
        # Update caches so later extractions in the same run see the row.
        self._index(new_entity)
        log.info(
            "resolver.discovered.created",
            name=canonical_name,
            entity_id=str(new_entity.id),
        )
        return new_entity.id

    @property
    def known_parties(self) -> list[KnownParty]:
        return list(self._known_parties)

    @classmethod
    async def from_session(cls, session: AsyncSession) -> EntityResolver:
        # Skip soft-rejected rows here once a `rejected` flag exists; for
        # V1.8 every row is in scope (seeded + discovered alike).
        result = await session.execute(select(Entity).where(col(Entity.id).is_not(None)))
        return cls(list(result.scalars().all()))
