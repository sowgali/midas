# midas

Map cash flow between companies in a sector — starting with AI — by parsing
earnings reports and press releases into a knowledge graph with end-to-end
provenance.

See [`DESIGN.md`](DESIGN.md) for the V1 architecture.

## Quickstart

Prerequisites: [`uv`](https://docs.astral.sh/uv/) and Docker.

```sh
# 1. Install Python 3.12 + project deps into a local .venv
uv sync

# 2. Set up local config
cp .env.example .env
$EDITOR .env                 # at minimum, set MIDAS_SEC_USER_AGENT and MIDAS_ANTHROPIC_API_KEY

# 3. Start Postgres
docker compose up -d

# 4. Verify install
uv run midas version
uv run pytest
```

## Layout

```
src/midas/
├── config.py        runtime settings
├── cli.py           typer entry point
├── models/          pydantic + SQLModel schemas (entities, deals, sources, evidence)
├── sources/         async fetchers for SEC EDGAR, IR pages, RSS, news
├── parsers/         raw -> text (HTML, PDF, XBRL)
├── extractors/      text -> structured deals (model-agnostic interface)
├── storage/         async SQLAlchemy engine, repositories, migrations
└── graph/           NetworkX builder + interactive HTML viz
```

## Development

```sh
uv run ruff check
uv run ruff format
uv run mypy
uv run pytest
```
