"""Shared HTTP client.

This is the *one* place in the codebase that talks to the network. It
gives us, for free across every source adapter:

* a global rate cap (``settings.http_rate_limit_per_sec``) — the SEC's
  fair-use policy is 10 req/s with a contact User-Agent; default 8;
* an on-disk cache keyed by ``sha256(url)`` so re-running the pipeline
  doesn't re-hit the network;
* tenacity retries with exponential backoff on transient failures (5xx,
  ``httpx.TransportError``);
* a configured User-Agent identifying a real human (SEC requirement).

The rate limiter is hand-rolled (asyncio.Lock + monotonic clock) — a
simple sliding-window throttle is plenty here and avoids pulling in
``aiolimiter`` for ~30 lines of code.

Use the module-level :func:`get_default_client` singleton in production
code; tests instantiate :class:`HttpClient` directly with their own
``cache_dir`` (typically ``tmp_path``).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ..config import settings


def _is_retryable_status(exc: BaseException) -> bool:
    return isinstance(exc, httpx.HTTPStatusError) and 500 <= exc.response.status_code < 600


_RETRY_EXC = (httpx.TransportError,)


class _RateLimiter:
    """Sliding-window rate limiter: at most ``rate`` calls per second.

    Implemented with a deque of the last ``rate`` call timestamps. On each
    ``acquire`` we sleep until the oldest one falls outside the 1-second
    window. Cheap, lock-protected, and good enough for our 8 req/s cap.
    """

    def __init__(self, rate_per_sec: float) -> None:
        if rate_per_sec <= 0:
            raise ValueError("rate_per_sec must be > 0")
        self._rate = rate_per_sec
        self._window = 1.0
        self._max_in_window = max(1, int(rate_per_sec))
        self._timestamps: deque[float] = deque()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            while True:
                now = time.monotonic()
                # drop timestamps outside the window
                while self._timestamps and (now - self._timestamps[0]) >= self._window:
                    self._timestamps.popleft()
                if len(self._timestamps) < self._max_in_window:
                    self._timestamps.append(now)
                    return
                # sleep just long enough for the oldest call to age out
                sleep_for = self._window - (now - self._timestamps[0])
                # tiny epsilon so we don't spin
                await asyncio.sleep(max(sleep_for, 0.001))


def _cache_paths(cache_dir: Path, url: str) -> tuple[Path, Path]:
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
    sub = cache_dir / digest[:2]
    return sub / f"{digest}.bin", sub / f"{digest}.meta.json"


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _atomic_write_text(path: Path, data: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(data, encoding="utf-8")
    os.replace(tmp, path)


class HttpClient:
    """Shared async HTTP client.

    Owns the rate limiter, the on-disk cache, and the retry policy. Use as
    an async context manager so the underlying ``httpx.AsyncClient`` is
    closed cleanly::

        async with HttpClient() as client:
            data = await client.get_bytes(url)
    """

    def __init__(
        self,
        *,
        cache_dir: Path | None = None,
        rate_per_sec: float | None = None,
        user_agent: str | None = None,
        timeout: float = 30.0,
        max_attempts: int = 4,
    ) -> None:
        self._cache_dir = Path(cache_dir) if cache_dir is not None else settings.cache_dir
        self._rate_limiter = _RateLimiter(
            rate_per_sec if rate_per_sec is not None else settings.http_rate_limit_per_sec
        )
        self._user_agent = user_agent or settings.sec_user_agent
        self._timeout = timeout
        self._max_attempts = max_attempts
        self._client: httpx.AsyncClient | None = None

    # ---------- lifecycle ----------

    async def __aenter__(self) -> HttpClient:
        self._ensure_client()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                headers={
                    "User-Agent": self._user_agent,
                    "Accept-Encoding": "gzip, deflate",
                },
                timeout=self._timeout,
                follow_redirects=True,
            )
        return self._client

    # ---------- cache ----------

    def _read_cache(self, url: str) -> tuple[bytes, dict[str, Any]] | None:
        bin_path, meta_path = _cache_paths(self._cache_dir, url)
        if not bin_path.exists() or not meta_path.exists():
            return None
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return bin_path.read_bytes(), meta

    def _write_cache(
        self,
        url: str,
        body: bytes,
        *,
        status_code: int,
        content_type: str | None,
    ) -> None:
        bin_path, meta_path = _cache_paths(self._cache_dir, url)
        meta = {
            "url": url,
            "status_code": status_code,
            "content_type": content_type,
            "fetched_at": datetime.now(UTC).isoformat(),
        }
        _atomic_write_bytes(bin_path, body)
        _atomic_write_text(meta_path, json.dumps(meta, separators=(",", ":")))

    # ---------- fetch ----------

    async def _do_request(self, url: str) -> httpx.Response:
        client = self._ensure_client()
        await self._rate_limiter.acquire()
        resp = await client.get(url)
        # raise on 5xx so tenacity can retry; 4xx surfaces immediately
        if 500 <= resp.status_code < 600:
            resp.raise_for_status()
        return resp

    async def get_bytes(self, url: str) -> bytes:
        """Fetch ``url`` and return raw bytes, hitting the cache when present."""
        cached = self._read_cache(url)
        if cached is not None:
            return cached[0]

        retry = AsyncRetrying(
            stop=stop_after_attempt(self._max_attempts),
            wait=wait_exponential(multiplier=0.2, min=0.2, max=4.0),
            retry=(
                retry_if_exception_type(_RETRY_EXC) | retry_if_exception_type(httpx.HTTPStatusError)
            ),
            reraise=True,
        )

        resp: httpx.Response | None = None
        async for attempt in retry:
            with attempt:
                candidate = await self._do_request(url)
                # only retry on 5xx; non-retryable HTTPStatusError must not loop
                if 500 <= candidate.status_code < 600:
                    candidate.raise_for_status()
                resp = candidate

        assert resp is not None
        # raise_for_status one more time for any non-2xx that slipped through
        if resp.status_code >= 400:
            resp.raise_for_status()

        body = resp.content
        self._write_cache(
            url,
            body,
            status_code=resp.status_code,
            content_type=resp.headers.get("content-type"),
        )
        return body

    async def get_text(self, url: str) -> str:
        body = await self.get_bytes(url)
        # respect a charset on the content-type if cached, but bytes->utf-8
        # is fine for everything we fetch (SEC JSON, HTML, XBRL).
        return body.decode("utf-8", errors="replace")

    async def get_json(self, url: str) -> Any:
        body = await self.get_bytes(url)
        return json.loads(body.decode("utf-8"))


# ---------- module-level singleton ----------

_default_client: HttpClient | None = None
_default_lock = asyncio.Lock()


async def get_default_client() -> HttpClient:
    """Return a process-wide :class:`HttpClient` configured from settings."""
    global _default_client
    if _default_client is None:
        async with _default_lock:
            if _default_client is None:
                _default_client = HttpClient()
                # eagerly construct the underlying httpx client
                _default_client._ensure_client()
    return _default_client
