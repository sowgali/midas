"""Cross-dialect column type aliases.

Production runs on Postgres (``JSONB``, native ``UUID``); tests can run on
SQLite for speed by transparently falling back to ``JSON`` / generic
``Uuid``. Centralizing this keeps the model files free of dialect noise.
"""

from __future__ import annotations

from sqlalchemy import JSON
from sqlalchemy.dialects.postgresql import JSONB

# JSON list/dict column: JSONB on Postgres, JSON elsewhere.
JSON_VARIANT = JSON().with_variant(JSONB(), "postgresql")
