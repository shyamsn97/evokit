"""A generational genetic algorithm: tournament selection into a fixed-size population."""

from __future__ import annotations

import asyncio

import evokit
from evokit.archives import PopulationArchive
from evokit.loop import EvalFn
from evokit.operator import Operator
from evokit.records import Individual
from evokit.history import History
from evokit.strategies import Tournament


async def ga(
    eval_fn: EvalFn,
    mutate: Operator,
    *,
    history: History,
    archive: PopulationArchive,
    generations: int = 100,
    tournament: int = 3,
    seed: int | None = None,
) -> Individual | None:
    """Each generation makes `archive.size` children from tournament winners and evaluates them
    together. The caller passes a seeded PopulationArchive; `mutate` is any operator, e.g. Mutate(fn).
    """
    strategy = Tournament(tournament, seed=seed)
    score = evokit.as_eval_fn(eval_fn)
    for _ in range(generations):
        asked = await evokit.ask(history, archive, strategy, mutate, n=archive.size)
        if asked.exhausted:
            break
        evaluations = await asyncio.gather(*map(score, asked.individuals))
        await evokit.tell(history, archive, asked.individuals, evaluations)
    return archive.best()
