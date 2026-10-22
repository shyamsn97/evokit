"""A generational genetic algorithm: tournament selection into a fixed-size population."""

from __future__ import annotations

import evokit
from evokit.archives import PopulationArchive
from evokit.loop import EvalFn
from evokit.operator import Operator
from evokit.records import Individual
from evokit.storage import Storage
from evokit.strategies import Tournament


async def ga(
    eval_fn: EvalFn,
    mutate: Operator,
    *,
    storage: Storage,
    archive: PopulationArchive,
    generations: int = 100,
    tournament: int = 3,
    seed: int | None = None,
) -> Individual | None:
    """Each generation makes `archive.size` children from tournament winners and evaluates them
    together. The caller passes a seeded PopulationArchive; `mutate` is any operator, e.g. Mutate(fn).
    """
    strategy = Tournament(tournament, seed=seed)
    for _ in range(generations):
        asked = await evokit.ask(storage, archive, strategy, mutate, n=archive.size)
        if asked.exhausted:
            break
        await evokit.tell(storage, archive, await evokit.evaluate(storage, asked.individuals, eval_fn=eval_fn))
    return archive.best()
