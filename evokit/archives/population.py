from __future__ import annotations

from evokit.archives.ranked import RankedArchive
from evokit.records import Individual
from evokit.storage import Storage


class PopulationArchive(RankedArchive):
    """Keep the top `size` individuals (truncation selection)."""

    def __init__(self, size: int, objective: str = "score", minimize: bool = False, *, storage: Storage | None = None, name: str = "archive"):
        super().__init__(objective, minimize, storage=storage, name=name)
        self.size = size

    def add(self, individual: Individual) -> bool:
        if self.score(individual) is None:
            return False
        members = self.members()
        if len(members) >= self.size and not self.better(self.score(individual), self.score(members[self.size - 1])):
            return False
        self.insert(individual)
        for extra in self.members()[self.size:]:
            self.remove(extra.id)
        return True
