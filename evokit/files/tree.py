"""Files: an immutable {path: bytes} value, the genome type for evolving programs."""

from __future__ import annotations

import difflib
import hashlib
import json
import os
from pathlib import Path
from typing import Callable, Iterable, Iterator, Mapping

from evokit.util import glob_match, truncate

DEFAULT_IGNORE = (
    ".git/**",
    ".evokit/**",
    ".claude/**",
    ".cursor/**",
    "**/__pycache__/**",
    "**/*.pyc",
    ".venv/**",
    "venv/**",
    "node_modules/**",
    "**/.DS_Store",
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _norm(path: str) -> str:
    return Path(path).as_posix()


def walk_files(directory: str | Path, ignore: Iterable[str] = DEFAULT_IGNORE) -> Iterator[tuple[str, Path]]:
    """(relative posix path, full path) for every regular file not matching `ignore`."""
    directory = Path(directory)
    ignore = tuple(ignore)
    for root, dirs, names in os.walk(directory):
        rel_root = Path(root).relative_to(directory).as_posix()
        rel_root = "" if rel_root == "." else rel_root + "/"
        dirs[:] = sorted(d for d in dirs if not glob_match(rel_root + d + "/x", ignore))
        for name in sorted(names):
            rel = rel_root + name
            full = Path(root) / name
            if full.is_symlink() or glob_match(rel, ignore):
                continue
            yield rel, full


class Files:
    """Immutable file contents keyed by relative path. Equality depends only on contents."""

    __slots__ = ("_files",)

    def __init__(self, files: Mapping[str, bytes] | None = None):
        self._files = {_norm(p): bytes(d) for p, d in sorted((files or {}).items())}

    @classmethod
    def from_dict(cls, files: Mapping[str, bytes | str]) -> "Files":
        return cls({p: d.encode() if isinstance(d, str) else d for p, d in files.items()})

    @classmethod
    def from_directory(
        cls,
        directory: str | Path,
        include: Iterable[str] = ("**",),
        exclude: Iterable[str] = (),
        ignore: Iterable[str] = DEFAULT_IGNORE,
    ) -> "Files":
        include, exclude = tuple(include), tuple(exclude)
        return cls(
            {
                rel: full.read_bytes()
                for rel, full in walk_files(directory, ignore)
                if glob_match(rel, include) and not glob_match(rel, exclude)
            }
        )

    def paths(self) -> list[str]:
        return list(self._files)

    def read(self, path: str) -> bytes:
        return self._files[_norm(path)]

    def text(self, path: str) -> str:
        return self.read(path).decode("utf-8", errors="replace")

    def with_files(self, files: Mapping[str, bytes | str | None]) -> "Files":
        """A new value with files replaced, added, or (value None) removed."""
        updated = dict(self._files)
        for path, data in files.items():
            if data is None:
                updated.pop(_norm(path), None)
            else:
                updated[_norm(path)] = data.encode() if isinstance(data, str) else data
        return Files(updated)

    def write_to(self, directory: str | Path) -> None:
        directory = Path(directory)
        for path, data in self._files.items():
            target = directory / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)

    def render(self, max_chars: int = 40_000) -> str:
        """Every file as a markdown section with a fenced block, for prompts."""
        out = [f"### {path}\n```\n{truncate(self.text(path), max_chars)}\n```" for path in self._files]
        return "\n\n".join(out) or "(empty: no files yet)"

    def diff(self, other: "Files", a: str = "a", b: str = "b") -> str:
        """A unified diff from this value to `other`, one section per changed file."""
        out = []
        for path in sorted(set(self.paths()) | set(other.paths())):
            old = self.text(path).splitlines(keepends=True) if path in self else []
            new = other.text(path).splitlines(keepends=True) if path in other else []
            out.extend(difflib.unified_diff(old, new, f"{a}/{path}", f"{b}/{path}"))
        return "".join(out)

    def describe(self) -> dict:
        """A short JSON summary (for command output); the contents are in the store."""
        return {"files": self.paths(), "digest": self.digest}

    @property
    def digest(self) -> str:
        return sha256(json.dumps({p: sha256(d) for p, d in self._files.items()}).encode())

    def to_json(self, put_blob: Callable[[bytes], str]) -> dict[str, str]:
        return {path: put_blob(data) for path, data in self._files.items()}

    @classmethod
    def from_json(cls, data: Mapping[str, str], get_blob: Callable[[str], bytes]) -> "Files":
        return cls({path: get_blob(key) for path, key in data.items()})

    def __contains__(self, path: str) -> bool:
        return _norm(path) in self._files

    def __len__(self) -> int:
        return len(self._files)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Files) and self._files == other._files

    def __hash__(self) -> int:
        return hash(self.digest)

    def __repr__(self) -> str:
        return f"Files({len(self)} files, {self.digest[:8]})"
