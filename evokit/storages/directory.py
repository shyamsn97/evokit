"""A run in one directory on disk, shared safely by several processes (CLI calls, MCP, agents)."""

from __future__ import annotations

import fcntl
import os
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from evokit.errors import RunBusy
from evokit.storage import Storage, check_path


class _FileLock:
    """An exclusive fcntl lock, reentrant within one thread, serialized across threads."""

    def __init__(self, path: Path):
        self.path = path
        self._thread_lock = threading.RLock()
        self._depth = 0
        self._fd: int | None = None

    def __enter__(self) -> "_FileLock":
        self._thread_lock.acquire()
        if self._depth == 0:
            self._fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
            fcntl.flock(self._fd, fcntl.LOCK_EX)
        self._depth += 1
        return self

    def __exit__(self, *exc) -> None:
        self._depth -= 1
        if self._depth == 0 and self._fd is not None:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
            os.close(self._fd)
            self._fd = None
        self._thread_lock.release()


class DirectoryStorage(Storage):
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = _FileLock(self.root / ".lock")

    def _path(self, path: str) -> Path:
        return self.root / check_path(path)

    def write(self, path: str, data: bytes) -> None:
        target = self._path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(f".{target.name}.tmp{os.getpid()}.{threading.get_ident()}")
        tmp.write_bytes(data)
        os.replace(tmp, target)

    def read(self, path: str) -> bytes:
        return self._path(path).read_bytes()

    def read_from(self, path: str, offset: int) -> bytes:
        with open(self._path(path), "rb") as f:
            f.seek(offset)
            return f.read()

    def exists(self, path: str) -> bool:
        return self._path(path).is_file()

    def list(self, prefix: str) -> list[str]:
        base = self._path(prefix)
        if not base.is_dir():
            return []
        return sorted(p.relative_to(self.root).as_posix() for p in base.rglob("*") if p.is_file())

    def append(self, path: str, data: bytes) -> None:
        target = self._path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, open(target, "ab") as f:
            f.write(data)

    @contextmanager
    def local(self, path: str) -> Iterator[Path]:
        folder = self._path(path)
        folder.mkdir(parents=True, exist_ok=True)
        yield folder

    @contextmanager
    def lock(self, name: str) -> Iterator[None]:
        path = self._path(f"run/{name}.lock")
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            holder = os.read(fd, 100).decode(errors="replace").strip() or "another process"
            os.close(fd)
            raise RunBusy(f"this run is being driven by {holder}", hint=f"wait for it, or stop it; the lock is {path}") from None
        os.ftruncate(fd, 0)
        os.write(fd, f"pid {os.getpid()}".encode())
        try:
            yield
        finally:
            os.ftruncate(fd, 0)
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    @property
    def key(self) -> str:
        return f"directory:{self.root}"

    def __repr__(self) -> str:
        return f"DirectoryStorage({str(self.root)!r})"
