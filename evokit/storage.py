"""Storage: bytes at paths. It knows nothing about individuals, archives, logs or costs.

Everything above it picks its own path and format:

    history/records.jsonl      evokit.history
    archive/changes.jsonl      the Archive base class
    blobs/ab/cdef...           genome contents, content-addressed
    artifacts/<id>/            each individual's evaluation folder, through `local`
"""

from __future__ import annotations

import hashlib
import importlib
import json
from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

from evokit.errors import UsageError
from evokit.records import Evaluation, Individual
from evokit.util import title_and_body


class Storage(ABC):
    @abstractmethod
    def write(self, path: str, data: bytes) -> None:
        """Replace the file at `path`."""

    @abstractmethod
    def read(self, path: str) -> bytes:
        """Raises FileNotFoundError if there is no file at `path`."""

    @abstractmethod
    def exists(self, path: str) -> bool: ...

    @abstractmethod
    def list(self, prefix: str) -> list[str]:
        """The paths of files under `prefix`, sorted."""

    @abstractmethod
    def append(self, path: str, data: bytes) -> None:
        """Add `data` to the end of the file at `path`, creating it if needed. Safe for concurrent writers."""

    def read_from(self, path: str, offset: int) -> bytes:
        """The bytes of the file at `path` from `offset` on: how a log is followed without re-reading it."""
        return self.read(path)[offset:]

    @abstractmethod
    def local(self, path: str) -> AbstractContextManager[Path]:
        """A real folder holding what is stored under `path`, for code that needs one. What is in the
        folder when the block ends is what is stored under `path`."""

    @abstractmethod
    def lock(self, name: str) -> AbstractContextManager[None]:
        """Held by one holder at a time, across processes. Raises RunBusy at once if another holder
        has it; loops take lock("driver") so only one process drives a run."""


def check_path(path: str) -> str:
    parts = path.split("/")
    if not path or path.startswith("/") or any(p in ("", ".", "..") for p in parts):
        raise ValueError(f"invalid storage path: {path!r}")
    return path


# genomes and individuals

TYPE_KEY = "$type"


def put_blob(storage: Storage, data: bytes) -> str:
    """Store bytes under their hash; equal content is stored once."""
    key = hashlib.sha256(data).hexdigest()
    path = f"blobs/{key[:2]}/{key[2:]}"
    if not storage.exists(path):
        storage.write(path, data)
    return key


def get_blob(storage: Storage, key: str) -> bytes:
    return storage.read(f"blobs/{key[:2]}/{key[2:]}")


def encode_genome(genome: Any, storage: Storage) -> Any:
    """JSON values pass through; objects with to_json(put_blob) are saved with their type."""
    if hasattr(genome, "to_json") and hasattr(type(genome), "from_json"):
        cls = type(genome)
        return {TYPE_KEY: f"{cls.__module__}:{cls.__qualname__}", "data": genome.to_json(lambda b: put_blob(storage, b))}
    try:
        json.dumps(genome)
    except (TypeError, ValueError):
        raise UsageError(
            f"cannot save a genome of type {type(genome).__name__}",
            hint="use JSON values (str, numbers, lists, dicts), or give the type to_json(put_blob) and from_json(data, get_blob)",
        ) from None
    return genome


def decode_genome(data: Any, storage: Storage) -> Any:
    if isinstance(data, dict) and set(data) == {TYPE_KEY, "data"}:
        module, _, name = data[TYPE_KEY].partition(":")
        cls = getattr(importlib.import_module(module), name)
        return cls.from_json(data["data"], lambda key: get_blob(storage, key))
    return data


def encode(individual: Individual, storage: Storage) -> dict:
    return {
        "id": individual.id,
        "name": individual.name,
        "description": individual.description,
        "genome": encode_genome(individual.genome, storage),
        "parents": list(individual.parents),
        "info": dict(individual.info),
        "evaluation": individual.evaluation.to_dict() if individual.evaluation is not None else None,
    }


def decode(data: dict, storage: Storage) -> Individual:
    info = data.get("info") or {}
    if "name" in data:
        name, description = data["name"], data.get("description") or ""
    else:  # written before individuals had names
        name, description = title_and_body(str(info.get("summary") or info.get("note") or ""))
    return Individual(
        id=data["id"],
        genome=decode_genome(data["genome"], storage),
        parents=tuple(data.get("parents") or ()),
        info=info,
        evaluation=Evaluation.from_dict(data["evaluation"]) if data.get("evaluation") else None,
        name=name,
        description=description,
    )


def append_json(storage: Storage, path: str, record: dict) -> None:
    storage.append(path, (json.dumps(record, default=str) + "\n").encode())


def read_json_lines(storage: Storage, path: str) -> list[dict]:
    if not storage.exists(path):
        return []
    return [json.loads(line) for line in storage.read(path).decode().splitlines() if line.strip()]
