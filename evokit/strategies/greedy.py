from __future__ import annotations

from evokit.archive import Archive
from evokit.records import Selection
from evokit.strategy import Strategy


class Greedy(Strategy):
    """Always build on the archive's best individual (hill climbing; AVO's single lineage)."""

    async def select(self, archive: Archive, n: int) -> list[Selection]:
        best = archive.best()
        return [] if best is None else [Selection((best,)) for _ in range(n)]
