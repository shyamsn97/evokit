from __future__ import annotations

import difflib
import random
from typing import Mapping, Sequence

from evokit.archives.ranked import RankedArchive
from evokit.records import Individual
from evokit.storage import Storage
from evokit.util import as_float


def genome_text(individual: Individual) -> str:
    """The genome as text: every file of a Files genome, else str(genome)."""
    genome = individual.genome
    if hasattr(genome, "paths") and hasattr(genome, "text"):
        return "\n".join(genome.text(p) for p in genome.paths())
    return str(genome)


class IslandArchive(RankedArchive):
    """Islands, each a MAP-Elites grid (AlphaEvolve's and OpenEvolve's program database).

    A child joins its parent's island; a seed (no parents) joins island 0 and is copied to every
    other island. Its cell is given by `features`: an objective (binned over `ranges`), or
    "diversity", the mean edit distance to a sample of the island. A cell keeps its `cell_size`
    best individuals; with features=() each island is one population of that size.
    Each member's metadata is {"island": i, "cell": [...]}. `migrate(top)` copies each island's
    top individuals to the next island in a ring; the recipe decides when.
    """

    def __init__(
        self,
        islands: int = 4,
        features: Sequence[str] = ("lines", "diversity"),
        bins: int = 10,
        ranges: Mapping[str, tuple[float, float]] | None = None,
        cell_size: int = 1,
        diversity_sample: int = 8,
        objective: str = "score",
        minimize: bool = False,
        seed: int = 0,
        *,
        storage: Storage | None = None,
        name: str = "archive",
    ):
        super().__init__(objective, minimize, storage=storage, name=name)
        self.islands, self.features, self.bins, self.cell_size = islands, tuple(features), bins, cell_size
        self.ranges = {"lines": (0.0, 150.0), "diversity": (0.0, 1.0), **(ranges or {})}
        self.diversity_sample = diversity_sample
        self._rng = random.Random(seed)

    def island_of(self, id: str) -> int | None:
        meta = self.metadata.get(id)
        return None if meta is None else meta["island"]

    def island_members(self, island: int) -> list[Individual]:
        return self.rank([self._members[id] for id, m in self.metadata.items() if m["island"] == island])

    def niches(self) -> dict[tuple, list[Individual]]:
        """Members grouped by (island, cell), each best first."""
        grouped: dict[tuple, list[Individual]] = {}
        for id, m in self.metadata.items():
            grouped.setdefault((m["island"], tuple(m["cell"])), []).append(self._members[id])
        return {key: self.rank(group) for key, group in grouped.items()}

    def diversity(self, individual: Individual, island: int) -> float:
        others = [m for m in self.island_members(island) if m.id != individual.id]
        if not others:
            return 0.0
        sample = self._rng.sample(others, min(self.diversity_sample, len(others)))
        text = genome_text(individual)
        return sum(1 - difflib.SequenceMatcher(None, text, genome_text(m)).ratio() for m in sample) / len(sample)

    def cell(self, individual: Individual, island: int) -> list[int]:
        cell = []
        for name in self.features:
            value = self.diversity(individual, island) if name == "diversity" else as_float(individual.objective(name))
            low, high = self.ranges.get(name, (0.0, 1.0))
            cell.append(min(self.bins - 1, max(0, int(((value or 0.0) - low) / (high - low) * self.bins))))
        return cell

    def _place(self, individual: Individual, island: int) -> bool:
        cell = self.cell(individual, island)
        occupants = [self._members[id] for id, m in self.metadata.items() if m["island"] == island and m["cell"] == cell]
        kept = self.rank([*occupants, individual])[: self.cell_size]
        if all(k.id != individual.id for k in kept):
            return False
        for occupant in occupants:
            if all(k.id != occupant.id for k in kept):
                self.remove(occupant.id)
        self.insert(individual, island=island, cell=cell)
        return True

    def _copy_to(self, individual: Individual, island: int) -> None:
        if any(m.genome == individual.genome for m in self.island_members(island)):
            return
        copy = Individual.new(individual.genome, [individual.id], {"operator": "migration", "island": island})
        self._place(copy.with_evaluation(individual.evaluation), island)

    def add(self, individual: Individual) -> bool:
        if self.score(individual) is None:
            return False
        if not individual.parents:
            admitted = self._place(individual, 0)
            for island in range(1, self.islands):
                self._copy_to(individual, island)
            return admitted
        parent_island = self.island_of(individual.parents[0])
        return self._place(individual, 0 if parent_island is None else parent_island)

    def migrate(self, top: int = 2) -> None:
        """Copy each island's `top` best individuals to the next island in a ring."""
        if self.islands < 2:
            return
        tops = [self.island_members(i)[:top] for i in range(self.islands)]
        for island, migrants in enumerate(tops):
            for migrant in migrants:
                self._copy_to(migrant, (island + 1) % self.islands)

    def table(self, island: int) -> str:
        """One island's grid as text: rows are the first feature's bins, columns the second's,
        each cell its best score."""
        cells = {tuple(c): group for (i, c), group in self.niches().items() if i == island}
        lines = [f"island {island}: {sum(len(g) for g in cells.values())} individuals"]
        if len(self.features) == 2:
            low, high = self.ranges.get(self.features[0], (0.0, 1.0))
            lines.append(f"  {self.features[0]} \\ {self.features[1]}".ljust(20) + " ".join(f"{d:>6}" for d in range(self.bins)))
            for row in sorted({cell[0] for cell in cells}):
                label = f"{low + row * (high - low) / self.bins:.4g}+"
                row_cells = [cells.get((row, d)) for d in range(self.bins)]
                lines.append(f"  {label:>17} " + " ".join(f"{self.score(c[0]):6.3f}" if c else "     ." for c in row_cells))
        return "\n".join(lines)
