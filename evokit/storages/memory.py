"""A run in this process's memory: for tests, notebooks and throwaway runs."""

from __future__ import annotations

import shutil
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from evokit.storage import Storage, check_path


class MemoryStorage(Storage):
    def __init__(self):
        self._files: dict[str, bytes] = {}
        self._lock = threading.Lock()

    def write(self, path: str, data: bytes) -> None:
        self._files[check_path(path)] = bytes(data)

    def read(self, path: str) -> bytes:
        try:
            return self._files[check_path(path)]
        except KeyError:
            raise FileNotFoundError(path) from None

    def exists(self, path: str) -> bool:
        return check_path(path) in self._files

    def list(self, prefix: str) -> list[str]:
        start = check_path(prefix).rstrip("/") + "/"
        return sorted(p for p in self._files if p.startswith(start))

    def append(self, path: str, data: bytes) -> None:
        with self._lock:
            self._files[check_path(path)] = self._files.get(path, b"") + data

    @contextmanager
    def local(self, path: str) -> Iterator[Path]:
        prefix = check_path(path).rstrip("/") + "/"
        folder = Path(tempfile.mkdtemp(prefix="evokit-"))
        try:
            for key, data in list(self._files.items()):
                if key.startswith(prefix):
                    target = folder / key[len(prefix):]
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(data)
            yield folder
            for key in [k for k in self._files if k.startswith(prefix)]:
                del self._files[key]
            for file in folder.rglob("*"):
                if file.is_file():
                    self._files[prefix + file.relative_to(folder).as_posix()] = file.read_bytes()
        finally:
            shutil.rmtree(folder, ignore_errors=True)

    def __repr__(self) -> str:
        return "MemoryStorage()"
