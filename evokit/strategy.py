"""Strategy: which individuals to build on? (parent selection)"""

from __future__ import annotations

from abc import ABC, abstractmethod

from evokit.archive import Archive
from evokit.records import Selection


class Strategy(ABC):
    """Picks parents (and inspirations) from the archive. Reads everything it needs from the
    archive, so it keeps no state that has to be saved. Never changes the archive."""

    @abstractmethod
    async def select(self, archive: Archive, n: int) -> list[Selection]:
        """n selections, or [] if there is nothing to select from."""
