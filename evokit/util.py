from __future__ import annotations

import asyncio
import re
import threading
import uuid
from functools import lru_cache
from typing import Any, Awaitable, Iterable, TypeVar

T = TypeVar("T")

_DURATION = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([smhd]?)\s*$")
_UNITS = {"": 1, "s": 1, "m": 60, "h": 3600, "d": 86400}


def parse_duration(value: str | float | int | None) -> float | None:
    """'10m' -> 600.0, '7d' -> 604800.0, 30 -> 30.0."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = _DURATION.match(value)
    if not match:
        raise ValueError(f"invalid duration: {value!r} (use e.g. '30s', '10m', '8h', '7d')")
    return float(match.group(1)) * _UNITS[match.group(2)]


def new_id(prefix: str = "") -> str:
    return prefix + uuid.uuid4().hex[:10]


@lru_cache(maxsize=512)
def _glob_regex(pattern: str) -> re.Pattern:
    out = []
    i = 0
    while i < len(pattern):
        if pattern.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
        elif pattern.startswith("**", i):
            out.append(".*")
            i += 2
        elif pattern[i] == "*":
            out.append("[^/]*")
            i += 1
        elif pattern[i] == "?":
            out.append("[^/]")
            i += 1
        else:
            out.append(re.escape(pattern[i]))
            i += 1
    return re.compile("^" + "".join(out) + "$")


def glob_match(path: str, patterns: Iterable[str]) -> bool:
    """Match a posix relative path against glob patterns where `**` crosses directories."""
    return any(_glob_regex(p).match(path) for p in patterns)


class _BackgroundLoop:
    """One event loop in a daemon thread, used to drive async components from sync code."""

    def __init__(self):
        self._loop: asyncio.AbstractEventLoop | None = None
        self._lock = threading.Lock()

    def _ensure(self) -> asyncio.AbstractEventLoop:
        with self._lock:
            if self._loop is None:
                loop = asyncio.new_event_loop()
                thread = threading.Thread(target=loop.run_forever, name="evokit-loop", daemon=True)
                thread.start()
                self._loop = loop
            return self._loop

    def run(self, coro: Awaitable[T]) -> T:
        loop = self._ensure()
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is loop:
            raise RuntimeError("cannot run a blocking evokit call from inside the evokit event loop; await it instead")
        return asyncio.run_coroutine_threadsafe(coro, loop).result()


background = _BackgroundLoop()


def truncate(text: str | None, limit: int) -> str | None:
    if text is None or len(text) <= limit:
        return text
    half = limit // 2
    return text[:half] + f"\n... [{len(text) - limit} chars truncated] ...\n" + text[-half:]


def as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None
