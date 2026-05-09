"""Runtime configuration loaded from environment / .env file.

All settings are prefixed with ``MIDAS_`` so they don't collide with other
env vars. Import the module-level :data:`settings` singleton from elsewhere
in the codebase rather than constructing :class:`Settings` directly.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Process-wide settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="MIDAS_",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_url: str = Field(
        default="postgresql+asyncpg://midas:midas@localhost:5432/midas",
        description="SQLAlchemy async URL for Postgres.",
    )
    sec_user_agent: str = Field(
        default="midas-research contact@example.com",
        description="Identifier sent to SEC EDGAR (required by their fair-use policy).",
    )
    anthropic_api_key: SecretStr | None = Field(
        default=None,
        description="API key for the Claude extractor; required only when running extraction.",
    )
    cache_dir: Path = Field(
        default=Path("data/raw"),
        description="On-disk cache for raw fetched documents.",
    )
    http_rate_limit_per_sec: float = Field(
        default=8.0,
        description="Global cap on HTTP request rate (shared client).",
    )


settings = Settings()
