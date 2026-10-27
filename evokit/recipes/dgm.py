"""Darwin Gödel Machine: the genome is a coding agent's repository, and the parent agent writes the child."""

from __future__ import annotations

import asyncio
from typing import Callable

import evokit
from evokit.archive import Archive
from evokit.agents import SelfModify
from evokit.files import Files
from evokit.loop import EvalFn, Limits, as_eval_fn
from evokit.operator import Operator
from evokit.records import Evaluation, Individual
from evokit.history import History
from evokit.strategies import Greedy, ProportionalSelection


async def dgm(
    eval_fn: EvalFn,
    *,
    history: History,
    archive: Archive,
    operator: Operator | None = None,
    fixed_agent: Files | None = None,
    instructions: str | None = None,
    objective: str = "score",
    workers: int = 4,
    max_children: int = 80,
    max_cost: float | None = None,
    max_failures: int | None = 10,
    sharpness: float = 10.0,
    midpoint: float = 0.5,
    open_ended: bool = True,
) -> Individual | None:
    """Parents are sampled by score, discounted by their children; each parent reads its own
    evaluation logs and rewrites its own code (SelfModify).

    The caller passes a seeded archive: UnboundedArchive keeps every working agent. The eval_fn
    runs the agent (files.script_agent) on the benchmark, fails agents that can no longer do the
    task, and returns logs under `logs/` (Evaluation.logs) for the next diagnosis; `staged_eval` runs the full
    benchmark only for agents that do well on a few tasks. `instructions` describe the benchmark
    to the diagnosing agent.

    The paper's ablations: `fixed_agent=seed_agent` (no self-improvement), and
    `open_ended=False` with a LatestArchive (build greedily on the newest working agent).
    """
    strategy = ProportionalSelection(sharpness, midpoint, objective=objective) if open_ended else Greedy()
    operator = operator or SelfModify(history.storage, fixed_agent=fixed_agent, instructions=instructions, objective=objective)

    limits = Limits(max_cost=max_cost, max_failures=max_failures)
    score = as_eval_fn(eval_fn)

    def stop_reason() -> str | None:
        if history.counts()["created"] >= max_children:
            return f"{max_children} children"
        return limits.reached(history)

    async def worker() -> None:
        while not stop_reason():
            asked = await evokit.ask(history, archive, strategy, operator)
            if asked.exhausted:
                return
            evaluations = await asyncio.gather(*map(score, asked.individuals))
            await evokit.tell(history, archive, asked.individuals, evaluations)

    await operator.check()
    with history.storage.lock("driver"):
        await asyncio.gather(*(worker() for _ in range(workers)))
    history.note("stop", stop_reason() or "nothing left to select")
    return archive.best()


def staged_eval(quick: EvalFn, full: EvalFn, promote: Callable[[Evaluation], bool]) -> EvalFn:
    """DGM's staged evaluation: run `quick`; only if promote(its evaluation), run `full` and return that."""

    quick, full = as_eval_fn(quick), as_eval_fn(full)

    async def staged(individual: Individual) -> Evaluation:
        evaluation = await quick(individual)
        return await full(individual) if promote(evaluation) else evaluation

    return staged
