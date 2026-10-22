from __future__ import annotations

import math
import random
from collections import Counter

from evokit.archive import Archive
from evokit.records import Individual, Selection
from evokit.strategy import Strategy


class ProportionalSelection(Strategy):
    """Fitness-proportionate selection with a penalty for children (the DGM's parent selection).

    Weight: sigmoid(sharpness * (score - midpoint)) / (1 + children), where children counts the
    archive members that list the individual as a parent, so well-explored individuals are
    picked less. Individuals at or above `solved` are not candidates. Sampled with replacement.
    """

    def __init__(
        self,
        sharpness: float = 10.0,
        midpoint: float = 0.5,
        solved: float | None = 1.0,
        objective: str = "score",
        seed: int | None = None,
    ):
        self.sharpness, self.midpoint, self.solved, self.objective = sharpness, midpoint, solved, objective
        self.rng = random.Random(seed)

    def _score(self, individual: Individual) -> float:
        return float(individual.objective(self.objective) or 0.0)

    def weights(self, members: list[Individual]) -> list[float]:
        children = Counter(parent for m in members for parent in m.parents)
        return [
            1 / (1 + math.exp(-self.sharpness * (self._score(m) - self.midpoint))) / (1 + children[m.id])
            for m in members
        ]

    async def select(self, archive: Archive, n: int) -> list[Selection]:
        members = archive.members()
        if not members:
            return []
        weights = self.weights(members)
        if self.solved is not None:
            unsolved = [(m, w) for m, w in zip(members, weights) if self._score(m) < self.solved]
            if unsolved:
                members, weights = [m for m, _ in unsolved], [w for _, w in unsolved]
        return [Selection((parent,)) for parent in self.rng.choices(members, weights, k=n)]
