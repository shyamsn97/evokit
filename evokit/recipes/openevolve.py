"""OpenEvolve (an open AlphaEvolve): islands of MAP-Elites grids, inspirations, an LLM ensemble writing diffs."""

from __future__ import annotations

import asyncio
from typing import Mapping

import evokit
from evokit import history
from evokit.archives import IslandArchive
from evokit.files import DiffOperator, Project
from evokit.loop import EvalFn, Limits
from evokit.operators import Choice, Model
from evokit.records import Individual
from evokit.storage import Storage
from evokit.strategies import IslandSampler


async def openevolve(
    project: Project,
    eval_fn: EvalFn,
    models: Mapping[str, Model],
    weights: Mapping[str, float] | None = None,
    *,
    storage: Storage,
    archive: IslandArchive,
    instructions: str | None = None,
    inspirations: int = 4,
    migrate_every: int = 20,
    migrate_top: int = 2,
    workers: int = 4,
    max_evaluations: int = 1000,
    max_cost: float | None = None,
    max_failures: int | None = 10,
    seed: int | None = None,
) -> Individual | None:
    """`models` maps a name to a model, picked by `weights` (e.g. {"fast": 0.8, "strong": 0.2}).
    Every `migrate_every` evaluations, each island sends its `migrate_top` best to the next.

    The caller passes a seeded IslandArchive, e.g. IslandArchive(islands=4, features=("lines",
    "diversity")), with an eval_fn wrapped in files.with_lines for the "lines" feature.
    Cascade evaluation (a quick test first) is part of the eval_fn.
    """
    strategy = IslandSampler(inspirations=inspirations, seed=seed)
    operator = Choice(
        {name: DiffOperator(model, project, instructions=instructions) for name, model in models.items()},
        weights=weights,
        seed=seed,
    )

    limits = Limits(max_evaluations, max_cost, max_failures)

    async def worker() -> None:
        while not limits.reached(storage):
            asked = await evokit.ask(storage, archive, strategy, operator)
            if asked.exhausted:
                return
            evaluated = await evokit.evaluate(storage, asked.individuals, eval_fn=eval_fn)
            before = history.counts(storage)["evaluated"]
            await evokit.tell(storage, archive, evaluated)
            if before // migrate_every < history.counts(storage)["evaluated"] // migrate_every:
                archive.migrate(top=migrate_top)

    await operator.check()
    with storage.lock("driver"):
        await asyncio.gather(*(worker() for _ in range(workers)))
    history.note(storage, "stop", limits.reached(storage) or "nothing left to select")
    return archive.best()
