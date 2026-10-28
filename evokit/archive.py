"""Archive: which evaluated individuals are kept (survivor selection).

Every archive has members by id, metadata per member (any JSON: an island and a cell, a
behaviour descriptor, a novelty score) and state of its own (any JSON: a threshold, a counter).
A subclass writes `add`, its rule for who gets in, and changes the archive only through
`insert`, `remove` and `set_state`. The base class appends each change to `<name>/changes.jsonl`
in its storage: in memory until it is attached to an experiment's storage, which keeps its members
from then on. An archive attached to an existing experiment replays that log, so it comes back
exactly as it was.

`sample(n, strategy)` picks members for a recipe (parents, inspirations): the archive offers its
candidates and the strategy (see `evokit.strategies`) picks among them; `top(n)` gives the best.
"""

from __future__ import annotations

import random
import time
from abc import ABC, abstractmethod
from typing import Any

from evokit.records import Individual
from evokit.storage import Storage, append_json, decode, encode, read_json_lines
from evokit.storages.memory import MemoryStorage
from evokit.strategies import Strategy, uniform


def _replay(storage: Storage, log: str) -> tuple[dict[str, Individual], dict[str, dict], dict[str, Any]]:
    members: dict[str, Individual] = {}
    metadata: dict[str, dict] = {}
    state: dict[str, Any] = {}
    for change in read_json_lines(storage, log):
        if "state" in change:
            state = change["state"]
        elif "removed" in change:
            members.pop(change["removed"], None)
            metadata.pop(change["removed"], None)
        else:
            individual = decode(change["inserted"], storage)
            members[individual.id] = individual
            metadata[individual.id] = change["metadata"]
    return members, metadata, state


def read_members(storage: Storage, name: str = "archive") -> dict[str, Individual]:
    """An archive's current members by id, from its log alone (no archive class needed)."""
    return _replay(storage, f"{name}/changes.jsonl")[0]


class Archive(ABC):
    strategy: Strategy = staticmethod(uniform)
    """What `sample` uses when it is given no strategy."""

    def __init__(self, name: str = "archive", seed: int | None = None):
        self.name = name
        self.storage: Storage = MemoryStorage()
        self.attached = False
        self.log = f"{name}/changes.jsonl"
        self._members: dict[str, Individual] = {}
        self.metadata: dict[str, dict] = {}
        self.state: dict[str, Any] = {}
        self.rng = random.Random(seed)

    def attach(self, storage: Storage, objective: str = "score", minimize: bool = False) -> None:
        """Keep the archive in `storage` from now on, restoring what its log there holds. An
        archive that ranks gets the task's `objective` and `minimize` unless it was given its own."""
        if self.attached and self.storage is not storage:
            raise ValueError("this archive already belongs to another experiment")
        self.storage, self.attached = storage, True
        if storage.exists(self.log):
            self._members, self.metadata, self.state = _replay(storage, self.log)
        else:
            for individual in list(self._members.values()):
                self._append({"inserted": encode(individual, storage), "metadata": self.metadata[individual.id]})

    def __getstate__(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k not in ("storage", "attached")}

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self.storage, self.attached = MemoryStorage(), False

    @abstractmethod
    def add(self, individual: Individual) -> bool:
        """Admit an evaluated individual or not, removing others if needed. The one method a subclass writes."""

    def insert(self, individual: Individual, **metadata: Any) -> None:
        """Make `individual` a member. Inserting a member again replaces its metadata."""
        self._members[individual.id] = individual
        self.metadata[individual.id] = metadata
        self._append({"inserted": encode(individual, self.storage), "metadata": metadata})

    def remove(self, id: str) -> None:
        del self._members[id]
        del self.metadata[id]
        self._append({"removed": id})

    def set_state(self, **state: Any) -> None:
        self.state = {**self.state, **state}
        self._append({"state": self.state})

    def _append(self, change: dict) -> None:
        append_json(self.storage, self.log, {**change, "time": time.time()})

    def members(self) -> list[Individual]:
        return list(self._members.values())

    def get(self, id: str) -> Individual | None:
        return self._members.get(id)

    def candidates(self) -> list[Individual]:
        """What `sample` picks from: every member (best first, in an archive that ranks)."""
        return self.members()

    def sample(self, n: int = 1, strategy: Strategy | None = None) -> list[Individual]:
        """Up to `n` members, picked by `strategy` (default: the archive's own) from its candidates."""
        candidates = self.candidates()
        if not candidates:
            return []
        return list((strategy or self.strategy)(candidates, n, self))

    def top(self, n: int = 1) -> list[Individual]:
        """The `n` best members; for an archive that does not rank, the newest."""
        return list(reversed(self.members()))[:n]

    def __len__(self) -> int:
        return len(self._members)

    def __contains__(self, id: str) -> bool:
        return id in self._members


def improves(archive: Archive, individual: Individual, best: Individual | None) -> bool:
    """Whether `individual` (evaluated) strictly beats `best`, by the archive's own ranking."""
    score, better = getattr(archive, "score", None), getattr(archive, "better", None)
    if score is not None and better is not None:
        new = score(individual)
        old = score(best) if best is not None else None
        return new is not None and (old is None or better(new, old))
    after = archive.best() if hasattr(archive, "best") else None
    return after is not None and after.id == individual.id and (best is None or best.id != individual.id)
