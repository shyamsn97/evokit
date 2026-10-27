"""OpenEvolve (an open AlphaEvolve): islands of MAP-Elites grids, inspirations, an LLM ensemble writing diffs."""

from __future__ import annotations

import asyncio
from typing import Mapping

import evokit
from evokit.archives import IslandArchive
from evokit.files import DiffOperator, Project
from evokit.loop import EvalFn, Limits
from evokit.operators import Choice, Model
from evokit.records import Individual
from evokit.history import History
from evokit.strategies import IslandSampler


async def openevolve(
    project: Project,
    eval_fn: EvalFn,
    models: Mapping[str, Model],
    weights: Mapping[str, float] | None = None,
    *,
    history: History,
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
    score = evokit.as_eval_fn(eval_fn)

    async def worker() -> None:
        while not limits.reached(history):
            asked = await evokit.ask(history, archive, strategy, operator)
            if asked.exhausted:
                return
            evaluations = await asyncio.gather(*map(score, asked.individuals))
            before = history.counts()["evaluated"]
            await evokit.tell(history, archive, asked.individuals, evaluations)
            if before // migrate_every < history.counts()["evaluated"] // migrate_every:
                archive.migrate(top=migrate_top)

    await operator.check()
    with history.storage.lock("driver"):
        await asyncio.gather(*(worker() for _ in range(workers)))
    history.note("stop", limits.reached(history) or "nothing left to select")
    return archive.best()
