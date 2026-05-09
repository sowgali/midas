"""Entity registry — the hand-curated seed of known companies."""

from .loader import default_seed_path, load_seed_registry, parse_seed

__all__ = ["default_seed_path", "load_seed_registry", "parse_seed"]
