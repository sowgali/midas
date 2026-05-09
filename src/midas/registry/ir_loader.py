"""Load + validate the IR (Investor Relations) source registry.

The registry is a hand-curated YAML at ``src/midas/registry/ir_sources.yaml``
mapping entity canonical names to either an RSS feed URL or an
``IrPressConfig`` (per-site CSS selectors). The CLI's ``midas ingest ir``
walks this list to populate non-SEC sources.

Each source carries its ``entity_canonical_name`` so the loader can
short-circuit unknown entities loudly (catching typos in the YAML
rather than silently extracting deals against the wrong row).
"""

from __future__ import annotations

from importlib.resources import as_file, files
from pathlib import Path
from typing import Annotated, Any, Literal

import structlog
import yaml
from pydantic import BaseModel, Discriminator, Field, TypeAdapter

from midas.models.types import SourceType

log = structlog.get_logger(__name__)

DEFAULT_IR_SOURCES_FILENAME = "ir_sources.yaml"


class _BaseIrSource(BaseModel):
    entity_canonical_name: str
    publisher: str


class RssSourceConfig(_BaseIrSource):
    """A simple RSS / Atom feed."""

    type: Literal["rss"]
    feed_url: str
    source_type: SourceType = SourceType.BLOG


class IrPressSourceConfig(_BaseIrSource):
    """A per-site HTML-list scraper using CSS selectors.

    Mirrors the runtime ``midas.sources.ir_press.IrPressConfig`` shape,
    minus ``entity_id`` (resolved at load time from canonical name).
    """

    type: Literal["ir_press"]
    index_url: str
    item_selector: str
    link_selector: str
    title_selector: str
    date_selector: str
    date_format: str
    article_body_selector: str
    link_base_url: str | None = None
    source_type: SourceType = SourceType.PRESS_RELEASE


class PlaywrightSourceConfig(_BaseIrSource):
    """A headless-browser scraper for Cloudflare / JS-rendered sites.

    Mirrors :class:`midas.sources.playwright_source.PlaywrightSourceConfig`
    minus ``entity_id``. Used for OpenAI / Anthropic news where ``rss``
    and ``ir_press`` both fail (Cloudflare TLS check + JS rendering).
    """

    type: Literal["playwright"]
    index_url: str
    item_selector: str
    title_selector: str | None = None
    date_selector: str | None = None
    date_format: str | None = None
    article_body_selector: str
    link_base_url: str | None = None
    wait_after_load_ms: int = 2500
    navigation_timeout_ms: int = 30_000
    source_type: SourceType = SourceType.BLOG


IrSourceConfig = Annotated[
    RssSourceConfig | IrPressSourceConfig | PlaywrightSourceConfig,
    Discriminator("type"),
]
_IrSourceListAdapter: TypeAdapter[list[IrSourceConfig]] = TypeAdapter(list[IrSourceConfig])


class IrSourcesFile(BaseModel):
    """Top-level structure of ``ir_sources.yaml``."""

    sources: list[IrSourceConfig] = Field(default_factory=list)


def default_ir_sources_path() -> Path:
    resource = files("midas.registry") / DEFAULT_IR_SOURCES_FILENAME
    with as_file(resource) as path:
        return Path(path)


def parse_ir_sources(
    path: Path | None = None,
) -> list[RssSourceConfig | IrPressSourceConfig | PlaywrightSourceConfig]:
    """Read and validate the IR YAML.

    Pydantic discriminator-based validation rejects unknown ``type``
    values + missing required fields up front — typos in the YAML
    fail at load, not at scrape time.
    """
    path = path or default_ir_sources_path()
    with path.open("rb") as f:
        data: dict[str, Any] = yaml.safe_load(f) or {}

    parsed = IrSourcesFile.model_validate(data)
    return list(parsed.sources)
