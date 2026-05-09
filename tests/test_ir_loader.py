"""Tests for the V1.7 IR-source registry loader.

The shipped ``ir_sources.yaml`` is small + curated, so the live config
itself is part of what's covered: each row must validate, point at a
known entity from the seed, and be either RSS or IR-press shape.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from midas.registry import (
    IrPressSourceConfig,
    PlaywrightSourceConfig,
    RssSourceConfig,
    parse_ir_sources,
    parse_seed,
)


def test_default_ir_sources_yaml_validates() -> None:
    sources = parse_ir_sources()
    assert len(sources) >= 5  # ~7 RSS + 2 Playwright at V1.9.
    for cfg in sources:
        assert isinstance(cfg, RssSourceConfig | IrPressSourceConfig | PlaywrightSourceConfig)


def test_every_ir_source_resolves_to_a_seeded_entity() -> None:
    """Catch typos in canonical names at test time, not at run time."""
    seeded = {e.canonical_name for e in parse_seed()}
    for cfg in parse_ir_sources():
        assert cfg.entity_canonical_name in seeded, (
            f"{cfg.entity_canonical_name!r} in ir_sources.yaml has no Entity row in seed.yaml"
        )


def test_ir_sources_include_at_least_one_private_company() -> None:
    """Private cos (OpenAI / Anthropic / etc.) need IR — that's the whole point."""
    seeded = {e.canonical_name: e for e in parse_seed()}
    has_private = False
    for cfg in parse_ir_sources():
        ent = seeded.get(cfg.entity_canonical_name)
        if ent is not None and ent.entity_type.value == "private_company":
            has_private = True
            break
    assert has_private, "IR registry has no private-company sources — that's the killer use case."


def test_rss_source_round_trip(tmp_path: Path) -> None:
    """A minimal hand-written YAML parses cleanly into the right discriminator branch."""
    p = tmp_path / "ir.yaml"
    p.write_text(
        yaml.safe_dump(
            {
                "sources": [
                    {
                        "entity_canonical_name": "OpenAI",
                        "type": "rss",
                        "feed_url": "https://openai.com/news/rss",
                        "publisher": "OpenAI",
                        "source_type": "blog",
                    },
                ],
            },
        ),
    )
    parsed = parse_ir_sources(p)
    assert len(parsed) == 1
    assert isinstance(parsed[0], RssSourceConfig)
    assert parsed[0].feed_url == "https://openai.com/news/rss"
    assert parsed[0].source_type.value == "blog"


def test_ir_press_source_round_trip(tmp_path: Path) -> None:
    p = tmp_path / "ir.yaml"
    p.write_text(
        yaml.safe_dump(
            {
                "sources": [
                    {
                        "entity_canonical_name": "Acme Co.",
                        "type": "ir_press",
                        "publisher": "Acme",
                        "index_url": "https://acme.example.com/press",
                        "item_selector": "article",
                        "link_selector": "a.headline",
                        "title_selector": "h2",
                        "date_selector": "time",
                        "date_format": "%B %d, %Y",
                        "article_body_selector": "div.article-body",
                        "link_base_url": "https://acme.example.com",
                    },
                ],
            },
        ),
    )
    parsed = parse_ir_sources(p)
    assert isinstance(parsed[0], IrPressSourceConfig)
    assert parsed[0].article_body_selector == "div.article-body"


def test_unknown_type_rejected(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text(
        yaml.safe_dump(
            {
                "sources": [
                    {
                        "entity_canonical_name": "X",
                        "type": "smtp",  # not a real source type
                        "publisher": "X",
                    },
                ],
            },
        ),
    )
    with pytest.raises(Exception):  # noqa: B017 — pydantic raises ValidationError
        parse_ir_sources(p)


def test_empty_yaml_parses_to_empty_list(tmp_path: Path) -> None:
    p = tmp_path / "empty.yaml"
    p.write_text("")
    assert parse_ir_sources(p) == []
