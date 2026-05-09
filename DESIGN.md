# midas — V1 Design

> Mapping cash flow between companies in a sector (starting with AI), from
> top-of-pyramid hyperscalers and labs down to the last mile of the economy.

---

## Mental model

What we're really building is an **ETL + knowledge graph** pipeline:

```
Sources → Parsers → Extractors (LLM) → Normalizers → Store → Graph
  ↑                       ↓
  └── provenance tracked through every layer ──┘
```

The non-obvious thing: **provenance is the feature**, not an afterthought.
Every dollar amount on every edge needs to point back to "this sentence in
this filing" — otherwise the graph is just vibes, and we'll never trust our
own analysis. Bake this in from day one.

---

## 1. Python 3.12 setup

Use **`uv`** for env + dependency management. Modern standard — handles
Python versions, lockfiles, and venvs in one tool, and is ~10–100× faster
than poetry/pip-tools.

### Project layout

```
midas/
├── pyproject.toml          # deps + tool config (ruff, mypy, pytest)
├── uv.lock
├── .python-version         # "3.12"
├── .env.example            # ANTHROPIC_API_KEY, SEC_USER_AGENT, etc.
├── src/midas/              # src layout (avoids import-shadowing bugs)
│   ├── __init__.py
│   ├── config.py           # settings, secrets (pydantic-settings)
│   ├── models/             # data models (pydantic + SQLModel)
│   ├── sources/            # acquisition (scrapers, API clients)
│   ├── parsers/            # raw → text
│   ├── extractors/         # text → structured Deal
│   ├── storage/            # DB + repositories
│   ├── graph/              # NetworkX builder + viz
│   ├── pipeline.py         # orchestration
│   └── cli.py              # typer-based CLI
├── tests/
└── data/
    ├── raw/                # cached HTML/PDF/XBRL — never re-scraped
    └── processed/          # normalized parquet/jsonl
```

### Toolchain

| Tool | Purpose |
|---|---|
| `uv` | env + dependency management |
| `ruff` | lint + format (replaces black/flake8/isort) |
| `mypy` or `pyright` | type checking; pydantic gives us a lot for free |
| `pytest` + `pytest-asyncio` + `respx` | tests; `respx` mocks `httpx` for offline HTTP tests |
| `typer` | CLI (`midas ingest sec --ticker MSFT`, `midas graph render`) |

---

## 2. Source acquisition / scraping module

Key insight: **don't scrape what we can fetch structurally.** Tier the sources:

| Tier | Source | Why | Tool |
|---|---|---|---|
| 1 | **SEC EDGAR** (10-K, 10-Q, 8-K) | Free API, structured XBRL financials, mandatory disclosure | `httpx` + EDGAR JSON endpoints |
| 2 | **Company IR press releases** | Authoritative for deals, partnerships | `httpx` + `selectolax`/`bs4` |
| 3 | **Official blogs** (OpenAI, Anthropic, etc.) | Only path for private cos | RSS where possible, scrape otherwise |
| 4 | **News aggregators** | Coverage, not source-of-truth | NewsAPI / GDELT / RSS |
| ⚠️ | Bloomberg / WSJ / Reuters | They actively block; don't bother | — |

### Module layout

```
sources/
├── base.py          # Source ABC + RawDocument(url, content_bytes, content_sha256,
│                   #                          fetched_at, source_type, publisher,
│                   #                          title, published_at)
├── http_client.py   # shared httpx.AsyncClient w/ rate limiting, retries, on-disk cache
├── sec_edgar.py     # ticker → CIK → filings; XBRL + HTML
├── ir_press.py      # configurable per-company IR feed scrapers
├── blog_rss.py      # generic RSS fetcher
└── registry.py      # which sources to try for which entity
```

### Rules to encode from day 1

- **Async-first** with `httpx.AsyncClient` and `asyncio`. Fan-out matters once
  we're scraping many companies × many filings. The shared client owns the
  rate limiter (semaphore + token bucket) so concurrency stays inside SEC's
  ≤10 req/s envelope across all coroutines.
- **Cache raw responses to disk** keyed by `sha256(url)`, with a sibling
  `.meta.json` recording `fetched_at`, content type, and status. Re-running
  the pipeline should never re-hit the network unless explicitly asked.
- **Rate-limit at the client layer**, not per-scraper. SEC requires ≤10 req/s
  with a contact User-Agent; we cap at **8 req/s by default**
  (`MIDAS_HTTP_RATE_LIMIT_PER_SEC`) to leave headroom.
- **`tenacity`** for retries with exponential backoff (its async API).
- Use **`playwright`** only when forced to (most IR sites are static HTML).

---

## 3. Data models

Use **`pydantic v2`** for in-memory validation, **`SQLModel`** (pydantic +
SQLAlchemy) for persistence — same class, two jobs.

```python
# Entity — a company or other money-handling actor
class Entity:
    id: UUID
    canonical_name: str            # "Alphabet Inc."
    aliases: list[str]             # ["Google", "GOOG", "GOOGL"]
    ticker: str | None
    cik: str | None                # SEC identifier
    entity_type: Literal["public_company", "private_company",
                         "fund", "government", "nonprofit"]
    sector_tags: list[str]         # ["ai", "cloud"]
    country: str | None

# Source — where we got the claim from
class Source:
    id: UUID
    url: str
    source_type: Literal["10-K", "10-Q", "8-K", "press_release",
                         "blog", "news", "earnings_call"]
    publisher: str                 # "SEC" or company name
    published_at: datetime
    fetched_at: datetime
    content_sha256: str            # raw cache key

# Deal — a single money-flow claim
class Deal:
    id: UUID
    from_entity_id: UUID
    to_entity_id: UUID
    deal_type: Literal["investment", "acquisition", "commercial_contract",
                       "partnership", "licensing", "debt", "grant"]
    amount_usd: Decimal | None     # normalized
    amount_native: Decimal | None
    currency: str | None
    announced_at: date | None
    closes_at: date | None
    status: Literal["announced", "closed", "rumored", "terminated"]
    confidence: float              # 0..1
    description: str               # short human summary

# EvidenceSpan — provenance: which source supports this deal
class EvidenceSpan:
    deal_id: UUID
    source_id: UUID
    text_snippet: str              # exact quote
    char_start: int
    char_end: int
    extractor: str                 # "llm:claude-opus-4-7" / "regex:dollar_amount"
```

### Three design choices worth flagging

1. **`Deal` is directional** (from→to). For partnerships without a clear payer,
   model as two deals or use `deal_type="partnership"` with `amount_usd=None`.
2. **Many EvidenceSpans per Deal.** Same deal gets reported in the press
   release, the 10-Q, and the earnings call — that's *good*, it raises
   confidence. Dedup at the Deal level, not the Source level.
3. **`amount_usd` is `Decimal`, not `float`.** Float math will quietly give us
   $999,999,999.99996 and we'll spend an afternoon debugging.

---

## 4. Extraction (text → Deals)

Two complementary extractors:

- **Regex / heuristic extractor** for known patterns:
  `$\d+(\.\d+)?\s*(billion|million|B|M)`, "invested in", "acquired", etc.
  Fast, deterministic, catches the easy stuff.
- **LLM extractor (Claude)** for the rest. Use **tool-use / structured
  outputs** with the `Deal` schema as the tool definition — the model returns
  validated structured data, not free-form JSON to repair. With prompt caching
  on the schema + instructions, this gets cheap fast.

Concrete shape:

```python
class Extractor(Protocol):
    name: ClassVar[str]                       # e.g. "claude:opus-4-7", "regex"
    async def extract(
        self, context: ExtractionContext
    ) -> list[ExtractedDeal]: ...
```

`ExtractionContext` carries `document_text`, `source_id` / `source_url` /
`source_type`, and `known_parties` (canonical name + aliases per known
entity). The extractor returns `ExtractedDeal`s with party names, an
exact `evidence_text_snippet` + char offsets, a confidence in `[0, 1]`,
and the `extractor_name` that produced it. Entity resolution (name →
`Entity.id`) happens later in the pipeline, after extraction.

**Entity resolution** ("Google" → Alphabet's `Entity` row) is its own concern.
Start with a hand-curated alias table for the top ~50 companies and graduate
to fuzzy matching later. Don't over-engineer in V1.

---

## 5. Storage + graph

**Postgres** (via SQLModel + SQLAlchemy 2.0 async + `asyncpg`) for entities,
deals, sources, evidence. Reasons it's the right call here:

- **Async-native driver (`asyncpg`)** — pairs cleanly with the async scraping
  layer. SQLite's async support is a thread-pool fiction, not real concurrency.
- **`JSONB` columns** — useful for `aliases`, raw extractor output, source
  metadata that doesn't deserve a column yet.
- **Full-text search (`tsvector`)** — we'll want this for searching evidence
  spans ("find every mention of 'compute commitment'") without pulling in
  Elasticsearch.
- **Concurrent writes** without `database is locked` errors when many
  scrapers are running.
- **`pgvector`** is there if we later want embedding-based entity resolution
  or semantic search over evidence.

Local dev: a `docker-compose.yml` with one Postgres 16 service. Connection
URL via env var, `.env.example` checked in.

For the graph itself, resist Neo4j for V1:

- **NetworkX** builds the graph on-demand from Postgres. `MultiDiGraph` so
  multiple deals between the same pair of entities each get their own edge.
- **PyVis** or **Plotly Sankey** for visualization — both export to
  interactive HTML.
- Migration path: when the graph itself becomes the workload (graph queries
  are the hot path, not SQL), swap NetworkX for **Kùzu** (embedded graph DB)
  or Neo4j. Postgres stays as the source of truth either way.

```
src/midas/storage/
├── db.py            # async engine, session factory, session() ctxmgr
└── repository.py    # EntityRepo, SourceRepo, DealRepo, EvidenceRepo (async)

alembic/             # migrations live at repo root, not under storage/
├── env.py           # async migration env; imports models for autogenerate
├── script.py.mako
└── versions/
    └── 0001_initial_schema.py

src/midas/graph/
├── builder.py       # SQL → networkx.MultiDiGraph (with as_of date filter)
├── queries.py       # "downstream of OpenAI", "all flows into NVDA"
└── viz.py           # render to HTML
```

---

## Development workflow

### Parallelism via subagents

Independent slices get parallel subagents; sequential dependencies stay on
the main thread. Concretely for V1:

| Can run in parallel | Must be sequential |
|---|---|
| Models, sources, extractors, graph (after schema is locked) | Project skeleton → schema → everything else |
| Each `sources/*.py` (SEC, IR press, RSS) | Pipeline orchestration (depends on all of the above) |
| Each `extractors/*.py` impl | Tests that exercise the full vertical slice |

Rule of thumb: if two tasks don't write to the same files and don't depend
on each other's output shape, dispatch them as separate subagents in one
turn. If they share a schema, lock the schema first (sequential), then
parallelize the consumers.

### Git discipline

- **Commit after each incremental, working step.** Skeleton, models, each
  source impl, each extractor impl, etc. Small commits, present-tense
  imperative messages ("Add SEC EDGAR async client", "Wire deal extractor
  interface").
- Subagents work in isolated worktrees when their changes are non-trivial,
  to avoid stomping on each other; merged back to `main` once green.
- Tests + lint must pass before each commit. `ruff check`, `mypy`, `pytest`
  in CI later, locally for now.
- `main` stays runnable at every commit.

## V1 build order

Build in this order to get end-to-end value fastest:

1. **Project skeleton** — uv, pyproject.toml, src layout, ruff/mypy/pytest wired up.
2. **Data models** — pydantic + SQLModel, with an in-memory test suite.
   Get the schema right before building anything that produces data for it.
3. **Seed entity registry** — hand-curated YAML of ~20 AI-sector companies
   with aliases/CIKs. Loaded into the DB on `midas init`.
4. **One full vertical slice on one company**: SEC EDGAR fetcher → 10-K
   parser → LLM extractor → store → render NVDA's edges. Resist the urge to
   go wide before this works end-to-end.
5. **Add press release source** — same vertical, second source type.
6. **Scale horizontally** — apply to all seeded companies.
7. **Graph viz + simple queries** — Sankey of "money out of OpenAI",
   subgraph rendering.

---

## Decisions

| # | Decision | Implication |
|---|---|---|
| 1 | **`uv`** for env + deps | `pyproject.toml` + `uv.lock`; `.python-version = 3.12` |
| 2 | **Async-first** (`httpx.AsyncClient`, `asyncio`) | Shared rate-limited client; SQLAlchemy async; `pytest-asyncio` |
| 3 | **Model-agnostic extractor interface** | `extractors/base.py` `Extractor` ABC; `claude.py` first impl, swappable for OpenAI / local |
| 4 | **Temporal graph** | `Deal.announced_at` / `closes_at`; queries take an "as-of" date; graph builder filters by date range |
| 5 | **Private companies in scope** | Press releases + news for OpenAI/Anthropic-class entities; `confidence` reflects source tier |
| 6 | **Postgres** as system of record | `asyncpg` driver, `JSONB`, `tsvector`, `pgvector`-ready; Docker Compose for local |

### Extractor interface sketch

```python
# extractors/base.py
class Extractor(Protocol):
    name: str                       # "claude:opus-4-7", "regex", "openai:gpt-...
    async def extract(
        self,
        text: str,
        context: ExtractionContext, # known parties, source metadata
    ) -> list[ExtractedDeal]:       # Deal candidates + EvidenceSpans
        ...
```

Concrete implementations live in `extractors/claude.py`, `extractors/regex.py`,
etc. The pipeline composes them (regex first as a cheap pass, LLM for the
long tail) and merges results, dedup'd by Deal identity.
