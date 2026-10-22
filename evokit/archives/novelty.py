from __future__ import annotations

import math
from typing import Callable, Sequence

from evokit.archive import Archive
from evokit.records import Individual
from evokit.storage import Storage

Behaviour = Callable[[Individual], Sequence[float]]


class NoveltyArchive(Archive):
    """Novelty search's archive: admit an individual whose behaviour is far from its `k` nearest
    members, with an adaptive threshold (lowered after each rejection, raised after each admission).

    `behaviour` is a function from an individual to a vector, or the names of the objectives that
    make up the vector. Each member's metadata is {"behaviour": [...], "novelty": float}.
    """

    def __init__(
        self,
        behaviour: Behaviour | Sequence[str],
        k: int = 15,
        threshold: float = 1.0,
        lower: float = 0.95,
        raise_by: float = 1.05,
        *,
        storage: Storage | None = None,
        name: str = "archive",
    ):
        super().__init__(storage, name)
        if callable(behaviour):
            self.behaviour = behaviour
        else:
            names = tuple(behaviour)
            self.behaviour = lambda individual: [individual.objective(n) for n in names]
        self.k, self.initial_threshold, self.lower, self.raise_by = k, threshold, lower, raise_by

    def novelty(self, behaviour: Sequence[float]) -> float:
        others = [m["behaviour"] for m in self.metadata.values()]
        if not others:
            return math.inf
        distances = sorted(math.dist(behaviour, other) for other in others)[: self.k]
        return sum(distances) / len(distances)

    def add(self, individual: Individual) -> bool:
        if individual.evaluation is None or not individual.evaluation.objectives:
            return False
        behaviour = [float(x) for x in self.behaviour(individual)]
        novelty = self.novelty(behaviour)
        threshold = self.state.get("threshold", self.initial_threshold)
        if novelty < threshold:
            self.set_state(threshold=threshold * self.lower)
            return False
        self.insert(individual, behaviour=behaviour, novelty=None if math.isinf(novelty) else novelty)
        self.set_state(threshold=threshold * self.raise_by)
        return True
