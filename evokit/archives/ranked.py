from __future__ import annotations

from evokit.archive import Archive
from evokit.records import Individual
from evokit.storage import Storage
from evokit.util import as_float


class RankedArchive(Archive):
    """Base for archives that rank by one objective (default: the task's). An individual without
    that objective is never admitted."""

    def __init__(self, objective: str | None = None, minimize: bool | None = None, *, name: str = "archive", seed: int | None = None):
        super().__init__(name, seed)
        self.objective = objective or "score"
        self.minimize = bool(minimize)
        self._given = objective is not None, minimize is not None

    def attach(self, storage: Storage, objective: str = "score", minimize: bool = False) -> None:
        if not self._given[0]:
            self.objective = objective
        if not self._given[1]:
            self.minimize = minimize
        super().attach(storage)

    def score(self, individual: Individual) -> float | None:
        """The ranking value, or None if the individual cannot be admitted."""
        return as_float(individual.objective(self.objective))

    def better(self, a: float, b: float) -> bool:
        return a < b if self.minimize else a > b

    def rank(self, individuals: list[Individual]) -> list[Individual]:
        """Best first; individuals without a score are dropped."""
        scored = [i for i in individuals if self.score(i) is not None]
        return sorted(scored, key=self.score, reverse=not self.minimize)

    def members(self) -> list[Individual]:
        """Best first."""
        return self.rank(super().members())

    def best(self) -> Individual | None:
        members = self.members()
        return members[0] if members else None

    def top(self, n: int = 1) -> list[Individual]:
        return self.members()[:n]
