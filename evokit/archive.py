"""Archive: which evaluated individuals are kept (survivor selection).

Every archive has members by id, metadata per member (any JSON: an island and a cell, a
behaviour descriptor, a novelty score) and state of its own (any JSON: a threshold, a counter).
A subclass writes `add`, its rule for who gets in, and changes the archive only through
`insert`, `remove` and `set_state`. With a storage, the base class appends each change to
`<name>/changes.jsonl`, and an archive built on an existing run replays that log, so it comes
back exactly as it was.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any

from evokit.records import Individual
from evokit.storage import Storage, append_json, decode, encode, read_json_lines


class Archive(ABC):
    def __init__(self, storage: Storage | None = None, name: str = "archive"):
        self.storage = storage
        self.log = f"{name}/changes.jsonl"
        self._members: dict[str, Individual] = {}
        self.metadata: dict[str, dict] = {}
        self.state: dict[str, Any] = {}
        if storage is not None:
            for change in read_json_lines(storage, self.log):
                if "state" in change:
                    self.state = change["state"]
                elif "removed" in change:
                    self._members.pop(change["removed"], None)
                    self.metadata.pop(change["removed"], None)
                else:
                    individual = decode(change["inserted"], storage)
                    self._members[individual.id] = individual
                    self.metadata[individual.id] = change["metadata"]

    @abstractmethod
    def add(self, individual: Individual) -> bool:
        """Admit an evaluated individual or not, removing others if needed. The one method a subclass writes."""

    def insert(self, individual: Individual, **metadata: Any) -> None:
        """Make `individual` a member. Inserting a member again replaces its metadata."""
        self._members[individual.id] = individual
        self.metadata[individual.id] = metadata
        self._append({"inserted": encode(individual, self.storage) if self.storage else None, "metadata": metadata})

    def remove(self, id: str) -> None:
        del self._members[id]
        del self.metadata[id]
        self._append({"removed": id})

    def set_state(self, **state: Any) -> None:
        self.state = {**self.state, **state}
        self._append({"state": self.state})

    def _append(self, change: dict) -> None:
        if self.storage is not None:
            append_json(self.storage, self.log, {**change, "time": time.time()})

    def members(self) -> list[Individual]:
        return list(self._members.values())

    def get(self, id: str) -> Individual | None:
        return self._members.get(id)

    def __len__(self) -> int:
        return len(self._members)

    def __contains__(self, id: str) -> bool:
        return id in self._members
