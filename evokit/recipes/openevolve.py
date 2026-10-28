"""OpenEvolve (an open AlphaEvolve): islands of MAP-Elites grids, inspirations, an LLM ensemble writing diffs."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path
from typing import Mapping, Sequence

from evokit.archive import Archive
from evokit.archives import IslandArchive
from evokit.files import with_lines
from evokit.operators import Model, diff
from evokit.records import Individual
from evokit.strategies import Strategy
from evokit.task import Task


async def openevolve(
    task: Task,
    models: Mapping[str, Model],
    weights: Mapping[str, float] | None = None,
    *,
    dir: str | Path | None = None,
    archive: IslandArchive | None = None,
    inspirations: int = 4,
    migrate_every: int = 20,
    migrate_top: int = 2,
    workers: int = 4,
    max_evaluations: int | None = 1000,
    max_cost: float | None = None,
    seed: int | None = None,
) -> Individual | None:
    """`models` maps a name to a model, picked by `weights` (e.g. {"fast": 0.8, "strong": 0.2}).
    Islands take turns; each child is a diff of a parent from the island, shown `inspirations`
    others (see `islands`). Every `migrate_every` evaluations, each island sends its `migrate_top`
    best to the next.

    `archive` defaults to IslandArchive(islands=4, features=("lines", "diversity")); for a Files
    genome, a "lines" objective is added to every evaluation. Cascade evaluation (a quick test
    first) is part of the task's evaluate.
    """
    if task.files is not None:
        task = replace(task, evaluate=with_lines(task.evaluate))
    archive = archive if archive is not None else IslandArchive(seed=seed or 0)
    experiment = task.create_experiment(dir, archive=archive, max_evaluations=max_evaluations, max_cost=max_cost)
    strategy = islands()
    names = list(models)
    odds = [(weights or {}).get(name, 1.0) for name in names]

    async def worker() -> None:
        while not experiment.limits.reached():
            parent, *others = archive.sample(1 + inspirations, strategy) or [None]
            if parent is None:
                return
            name = archive.rng.choices(names, odds)[0]
            if child := await experiment.create(diff(models[name], parent, others, task=task)):
                await experiment.submit(child.with_info(operator=name))
            if experiment.since_note("migrate") >= migrate_every:
                experiment.note("migrate")
                archive.migrate(top=migrate_top)

    await asyncio.gather(*(model.check() for model in models.values()))
    await asyncio.gather(*(worker() for _ in range(workers)))
    experiment.note("stop", experiment.limits.reached() or "nothing left to select")
    return experiment.best()


def islands(exploration: float = 0.2, exploitation: float = 0.7, elite: float = 0.1, top: int = 3) -> Strategy:
    """OpenEvolve's sampling, from one island's members (IslandArchive offers one island per call):
    the first pick, the parent, is a random member (`exploration`), one of the island's `top` best
    (`exploitation`), or one of the `top` best overall (`elite`). The rest are inspirations: the
    island's best members, then the best of its other cells (`archive.niches()`), in random order."""

    def pick(candidates: Sequence[Individual], n: int, archive: Archive) -> list[Individual]:
        rng, pool, everyone = archive.rng, list(candidates), archive.top(len(archive))
        r = rng.random() * (exploration + exploitation + elite)
        if r < exploration:
            parent = rng.choice(pool)
        elif r < exploration + exploitation:
            parent = rng.choice(pool[:top])
        else:
            parent = rng.choice(everyone[:top] or pool)
        chosen = [parent]
        for m in pool:
            if len(chosen) > (n - 1) // 2:
                break
            if m.id != parent.id:
                chosen.append(m)
        ids, rank = {m.id for m in pool}, {m.id: i for i, m in enumerate(everyone)}
        niches = archive.niches() if hasattr(archive, "niches") else {}
        cells = [min(ms, key=lambda m: rank.get(m.id, len(rank))) for ms in niches.values() if ms and ms[0].id in ids]
        others = [m for m in cells if all(c.id != m.id for c in chosen)]
        rng.shuffle(others)
        return (chosen + others)[:n]

    return pick
