"""Darwin Gödel Machine: the genome is a coding agent's repository, and the parent agent writes the child."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Callable

import evokit
from evokit import history
from evokit.archive import Archive
from evokit.files import Files, SelfModify
from evokit.loop import EvalFn, call_eval_fn
from evokit.operator import Operator
from evokit.records import Evaluation, Individual
from evokit.storage import Storage
from evokit.strategies import Greedy, ProportionalSelection


async def dgm(
    eval_fn: EvalFn,
    *,
    storage: Storage,
    archive: Archive,
    operator: Operator | None = None,
    fixed_agent: Files | None = None,
    instructions: str | None = None,
    objective: str = "score",
    workers: int = 4,
    max_children: int = 80,
    max_cost: float | None = None,
    sharpness: float = 10.0,
    midpoint: float = 0.5,
    open_ended: bool = True,
) -> Individual | None:
    """Parents are sampled by score, discounted by their children; each parent reads its own
    evaluation logs and rewrites its own code (SelfModify).

    The caller passes a seeded archive: UnboundedArchive keeps every working agent. The eval_fn
    runs the agent (files.script_agent) on the benchmark, fails agents that can no longer do the
    task, and writes logs to `workdir/logs` for the next diagnosis; `staged_eval` runs the full
    benchmark only for agents that do well on a few tasks. `instructions` describe the benchmark
    to the diagnosing agent.

    The paper's ablations: `fixed_agent=seed_agent` (no self-improvement), and
    `open_ended=False` with a LatestArchive (build greedily on the newest working agent).
    """
    strategy = ProportionalSelection(sharpness, midpoint, objective=objective) if open_ended else Greedy()
    operator = operator or SelfModify(storage, fixed_agent=fixed_agent, instructions=instructions, objective=objective)

    async def worker() -> None:
        while history.counts(storage)["created"] < max_children and (
            max_cost is None or history.usage(storage)["cost_usd"] < max_cost
        ):
            asked = await evokit.ask(storage, archive, strategy, operator)
            if asked.exhausted:
                return
            await evokit.tell(storage, archive, await evokit.evaluate(storage, asked.individuals, eval_fn=eval_fn))

    await asyncio.gather(*(worker() for _ in range(workers)))
    return archive.best()


def staged_eval(quick: EvalFn, full: EvalFn, promote: Callable[[Evaluation], bool]) -> EvalFn:
    """DGM's staged evaluation: run `quick`; only if promote(its evaluation), run `full` and return that."""

    async def eval_fn(individual: Individual, workdir: Path) -> Evaluation:
        evaluation = await call_eval_fn(quick, individual, workdir)
        return await call_eval_fn(full, individual, workdir) if promote(evaluation) else evaluation

    return eval_fn
