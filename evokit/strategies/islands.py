from __future__ import annotations

import random

from evokit.archive import Archive
from evokit.records import Selection
from evokit.strategy import Strategy


class IslandSampler(Strategy):
    """Islands take turns. The parent is a random member of the island (`exploration`), one of
    its best (`exploitation`), or one of the best overall (`elite`). Inspirations: the island's
    top members, then the best of other random cells, `inspirations` in all.

    Islands are read from `archive.niches()` keys of the form (island, cell), as IslandArchive
    gives them; an archive without niches is one island.
    """

    def __init__(
        self,
        exploration: float = 0.2,
        exploitation: float = 0.7,
        elite: float = 0.1,
        inspirations: int = 4,
        top: int = 3,
        seed: int | None = None,
    ):
        self.exploration, self.exploitation, self.elite = exploration, exploitation, elite
        self.inspirations, self.top = inspirations, top
        self.turn = 0
        self._rng = random.Random(seed)

    async def select(self, archive: Archive, n: int) -> list[Selection]:
        everyone = archive.members()
        if not everyone:
            return []
        rank = {m.id: i for i, m in enumerate(everyone)}
        niches = archive.niches() if hasattr(archive, "niches") else {None: everyone}
        islands: dict = {}
        for key, members in niches.items():
            island, cell = key if isinstance(key, tuple) and len(key) == 2 else (None, key)
            if members:
                islands.setdefault(island, {})[cell] = sorted(members, key=lambda m: rank.get(m.id, len(rank)))
        keys = sorted(islands, key=str)
        selections = []
        for _ in range(n):
            cells = islands[keys[self.turn % len(keys)]]
            self.turn += 1
            pool = sorted((m for ms in cells.values() for m in ms), key=lambda m: rank.get(m.id, len(rank)))
            r = self._rng.random() * (self.exploration + self.exploitation + self.elite)
            if r < self.exploration:
                parent = self._rng.choice(pool)
            elif r < self.exploration + self.exploitation:
                parent = self._rng.choice(pool[: self.top])
            else:
                parent = self._rng.choice(everyone[: self.top])
            chosen = [parent]
            for m in pool:
                if len(chosen) > self.inspirations // 2:
                    break
                if m.id != parent.id:
                    chosen.append(m)
            others = [ms[0] for ms in cells.values() if all(c.id != ms[0].id for c in chosen)]
            self._rng.shuffle(others)
            chosen += others[: self.inspirations + 1 - len(chosen)]
            selections.append(Selection((parent,), tuple(chosen[1 : self.inspirations + 1])))
        return selections
