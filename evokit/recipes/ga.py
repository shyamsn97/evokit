"""A generational genetic algorithm: tournament selection into a fixed-size population."""

from __future__ import annotations

import asyncio
import inspect
import random
from pathlib import Path
from typing import Any, Callable

from evokit.archives import PopulationArchive
from evokit.records import Individual
from evokit.strategies import tournament
from evokit.task import Task


async def ga(
    task: Task,
    mutate: Callable[[Any], Any],
    *,
    dir: str | Path | None = None,
    archive: PopulationArchive | None = None,
    generations: int = 100,
    tournament_size: int = 3,
    max_evaluations: int | None = None,
    max_cost: float | None = None,
    seed: int | None = None,
) -> Individual | None:
    """Each generation makes `archive.size` children from tournament winners and evaluates them
    together. `mutate` is a function from a genome to a new one, sync or async; `archive` defaults
    to PopulationArchive(10)."""
    archive = archive if archive is not None else PopulationArchive(10)
    if seed is not None:
        archive.rng = random.Random(seed)
    experiment = task.create_experiment(dir, archive=archive, max_evaluations=max_evaluations, max_cost=max_cost)
    name = getattr(mutate, "__name__", "mutate")

    async def child(parent: Individual) -> tuple[Individual, dict]:
        genome = mutate(parent.genome)
        if inspect.isawaitable(genome):
            genome = await genome
        return Individual.new(genome, [parent.id], {"operator": name}), {}

    for _ in range(generations):
        if experiment.limits.reached():
            break
        parents = archive.sample(archive.size, tournament(tournament_size))
        children = await asyncio.gather(*(experiment.create(child(p)) for p in parents))
        await asyncio.gather(*(experiment.submit(c) for c in children if c is not None))
    experiment.note("stop", experiment.limits.reached() or f"{generations} generations")
    return experiment.best()
