from __future__ import annotations

from evokit.archives.ranked import RankedArchive
from evokit.records import Individual


class UnboundedArchive(RankedArchive):
    """Admit every individual that has a score, and never drop one (the DGM's open-ended archive)."""

    def add(self, individual: Individual) -> bool:
        if self.score(individual) is None:
            return False
        self.insert(individual)
        return True


class LatestArchive(RankedArchive):
    """Keep only the newest individual that has a score (the DGM ablation without open-ended exploration)."""

    def add(self, individual: Individual) -> bool:
        if self.score(individual) is None:
            return False
        for member in self.members():
            self.remove(member.id)
        self.insert(individual)
        return True
