from __future__ import annotations

import asyncio
import atexit
import contextlib
import functools
import os
import re
import signal
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
        handle_termination()
        loop = self._ensure()
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is loop:
            raise RuntimeError("cannot run a blocking evokit call from inside the evokit event loop; await it instead")
        return asyncio.run_coroutine_threadsafe(coro, loop).result()


background = _BackgroundLoop()


# Children (agents, evaluators) run in their own process groups, so a timeout can kill everything
# they started. The same isolation means they outlive evokit unless evokit kills them on its way out.

_children: set[int] = set()
_children_lock = threading.Lock()
_exit_hook = False
_signal_handlers = False


def spawned(proc) -> None:
    """Kill this child's process group if evokit exits or is terminated (SIGTERM, SIGHUP)."""
    global _exit_hook
    with _children_lock:
        _children.add(proc.pid)
        if not _exit_hook:
            atexit.register(kill_children)
            _exit_hook = True
    handle_termination()


def reaped(proc) -> None:
    with _children_lock:
        _children.discard(proc.pid)


def kill_children() -> None:
    with _children_lock:
        pids = list(_children)
        _children.clear()
    for pid in pids:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(pid, signal.SIGKILL)


def handle_termination() -> None:
    """On SIGTERM or SIGHUP, kill the children, then do what the signal did before. Signal handlers
    can only be installed from the main thread; calls from other threads, and later calls, do nothing."""
    global _signal_handlers
    if _signal_handlers or threading.current_thread() is not threading.main_thread():
        return
    _signal_handlers = True
    for sig in (signal.SIGTERM, signal.SIGHUP):
        previous = signal.getsignal(sig)
        if previous is not signal.SIG_IGN:  # e.g. SIGHUP under nohup: the run is meant to survive it
            signal.signal(sig, functools.partial(_terminate, previous=previous))


def _terminate(signum, frame, previous) -> None:
    kill_children()
    if callable(previous):
        previous(signum, frame)
        return
    signal.signal(signum, signal.SIG_DFL)
    os.kill(os.getpid(), signum)


def truncate(text: str | None, limit: int) -> str | None:
    if text is None or len(text) <= limit:
        return text
    half = limit // 2
    return text[:half] + f"\n... [{len(text) - limit} chars truncated] ...\n" + text[-half:]


NAME_LIMIT = 100


def title_and_body(text: str | None) -> tuple[str, str]:
    """A name (the first non-empty line, without markdown marks, at most NAME_LIMIT characters) and
    a description (the rest), from a reply or a note."""
    lines = (text or "").strip().splitlines()
    if not lines:
        return "", ""
    name = lines[0].strip().lstrip("#*-> ").rstrip("*").strip()
    if name.lower().startswith(("title:", "summary:")):
        name = name.split(":", 1)[1].strip()
    if len(name) > NAME_LIMIT:
        name = name[: NAME_LIMIT - 3].rstrip() + "..."
    return name, "\n".join(lines[1:]).strip()


def as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None
