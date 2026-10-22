from __future__ import annotations

from typing import Literal

from evokit.archives.ranked import RankedArchive
from evokit.records import Individual
from evokit.storage import Storage


class BestArchive(RankedArchive):
    """Keep a single best individual.

    ties="accept" also admits equal scores, so the lineage moves forward on non-regressing
    changes (AVO's commit rule when the evaluator fails incorrect programs).
    """

    def __init__(
        self,
        objective: str = "score",
        minimize: bool = False,
        ties: Literal["reject", "accept"] = "reject",
        *,
        storage: Storage | None = None,
        name: str = "archive",
    ):
        super().__init__(objective, minimize, storage=storage, name=name)
        if ties not in ("reject", "accept"):
            raise ValueError("ties must be 'reject' or 'accept'")
        self.ties = ties

    def admits(self, individual: Individual, best: Individual) -> bool:
        """Whether `individual` replaces the current best."""
        score, current = self.score(individual), self.score(best)
        if score == current:
            return self.ties == "accept"
        return self.better(score, current)

    def add(self, individual: Individual) -> bool:
        if self.score(individual) is None:
            return False
        best = self.best()
        if best is not None and not self.admits(individual, best):
            return False
        if best is not None:
            self.remove(best.id)
        self.insert(individual)
        return True
