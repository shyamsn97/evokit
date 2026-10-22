from __future__ import annotations

import random

from evokit.archive import Archive
from evokit.records import Selection
from evokit.strategy import Strategy


class Tournament(Strategy):
    """Tournament selection: each parent is the best of `size` members drawn at random.
    `parents=2` picks two parents per selection, for crossover."""

    def __init__(self, size: int = 3, parents: int = 1, seed: int | None = None):
        self.size, self.parents = size, parents
        self.rng = random.Random(seed)

    async def select(self, archive: Archive, n: int) -> list[Selection]:
        members = archive.members()
        if not members:
            return []
        rank = {m.id: i for i, m in enumerate(members)}

        def winner():
            return min(self.rng.sample(members, min(self.size, len(members))), key=lambda m: rank[m.id])

        return [Selection(tuple(winner() for _ in range(self.parents))) for _ in range(n)]
