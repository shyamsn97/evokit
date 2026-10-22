from __future__ import annotations

from evokit.archives.best import BestArchive
from evokit.records import Individual
from evokit.storage import Storage
from evokit.util import as_float


class SimplerBestArchive(BestArchive):
    """Keep the single best individual; a tie (within `tolerance`) is admitted only if it is
    smaller by the `size` objective (autoresearch's simplicity rule)."""

    def __init__(
        self,
        objective: str = "score",
        size: str = "lines",
        tolerance: float = 1e-6,
        minimize: bool = False,
        *,
        storage: Storage | None = None,
        name: str = "archive",
    ):
        super().__init__(objective, minimize, storage=storage, name=name)
        self.size = size
        self.tolerance = tolerance

    def _size(self, individual: Individual) -> float:
        value = as_float(individual.objective(self.size))
        return float("inf") if value is None else value

    def admits(self, individual: Individual, best: Individual) -> bool:
        score, current = self.score(individual), self.score(best)
        if abs(score - current) <= self.tolerance:
            return self._size(individual) < self._size(best)
        return self.better(score, current)
