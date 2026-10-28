"""Darwin Gödel Machine: the genome is a coding agent's repository, and the parent agent writes the child."""

from __future__ import annotations

import asyncio
import functools
import math
from collections import Counter
from pathlib import Path
from typing import Awaitable, Callable, Sequence

from evokit.archive import Archive
from evokit.archives import LatestArchive, UnboundedArchive
from evokit.evaluate import Evaluate, as_evaluate
from evokit.experiment import Experiment
from evokit.files import Files
from evokit.operators.self_modify import self_modify
from evokit.records import Evaluation, Individual
from evokit.strategies import Strategy
from evokit.task import Task

Modify = Callable[[Individual, Experiment], Awaitable[tuple[Individual, dict]]]
"""`modify(parent, experiment)`: the child and what it cost, or OperatorError."""


async def dgm(
    task: Task,
    *,
    dir: str | Path | None = None,
    archive: Archive | None = None,
    modify: Modify | None = None,
    fixed_agent: Files | None = None,
    workers: int = 4,
    max_children: int = 80,
    max_cost: float | None = None,
    max_evaluations: int | None = None,
    sharpness: float = 10.0,
    midpoint: float = 0.5,
    open_ended: bool = True,
) -> Individual | None:
    """Parents are sampled by score, discounted by their children; each parent reads its own
    evaluation logs and rewrites its own code (`self_modify`, or `modify`). Stops after
    `max_children` children, or at a limit.

    The task's seed is the agent's repository. Its evaluate runs the agent (operators.script_agent)
    on the benchmark, fails agents that can no longer do the task, and returns logs under `logs/`
    (Evaluation.logs) for the next diagnosis; `staged_eval` runs the full benchmark only for
    agents that do well on a few tasks. The task's instructions describe the benchmark to the
    diagnosing agent.

    The paper's ablations: `fixed_agent=seed_agent` (no self-improvement), and
    `open_ended=False` (build greedily on the newest working agent, a LatestArchive).
    """

    def children(experiment: Experiment) -> str | None:
        return f"{max_children} children" if experiment.counts()["created"] >= max_children else None

    if archive is None:
        archive = UnboundedArchive() if open_ended else LatestArchive()
    experiment = task.create_experiment(dir, archive=archive, max_cost=max_cost, max_evaluations=max_evaluations, stop=(children,))
    strategy = proportional(sharpness, midpoint, objective=task.objective) if open_ended else None
    modify = modify or functools.partial(self_modify, fixed_agent=fixed_agent, objective=task.objective)

    async def worker() -> None:
        while not experiment.limits.reached():
            parents = archive.sample(1, strategy)
            if not parents:
                return
            if child := await experiment.create(modify(parents[0], experiment)):
                await experiment.submit(child)

    await asyncio.gather(*(worker() for _ in range(workers)))
    experiment.note("stop", experiment.limits.reached() or "nothing left to select")
    return experiment.best()


def proportional(sharpness: float = 10.0, midpoint: float = 0.5, solved: float | None = 1.0, objective: str = "score") -> Strategy:
    """Fitness-proportionate sampling with a penalty for children (the DGM's parent selection).

    Weight: sigmoid(sharpness * (score - midpoint)) / (1 + children), where children counts the
    candidates that list the individual as a parent, so well-explored individuals are picked
    less. Individuals at or above `solved` are not picked while others remain. With replacement."""

    def score(individual: Individual) -> float:
        return float(individual.objective(objective) or 0.0)

    def pick(candidates: Sequence[Individual], n: int, archive: Archive) -> list[Individual]:
        kids = Counter(parent for m in candidates for parent in m.parents)
        members = list(candidates)
        weights = [1 / (1 + math.exp(-sharpness * (score(m) - midpoint))) / (1 + kids[m.id]) for m in members]
        if solved is not None and (unsolved := [(m, w) for m, w in zip(members, weights) if score(m) < solved]):
            members, weights = [m for m, _ in unsolved], [w for _, w in unsolved]
        return archive.rng.choices(members, weights, k=n)

    return pick


def staged_eval(quick: Evaluate, full: Evaluate, promote: Callable[[Evaluation], bool]) -> Evaluate:
    """DGM's staged evaluation: run `quick`; only if promote(its evaluation), run `full` and return that."""

    quick, full = as_evaluate(quick), as_evaluate(full)

    async def staged(individual: Individual) -> Evaluation:
        evaluation = await quick(individual)
        return await full(individual) if promote(evaluation) else evaluation

    return staged
