"""autoresearch (Karpathy, 2026): one lineage; keep a change only if it is better, or ties and is smaller."""

from __future__ import annotations

import asyncio
from pathlib import Path

import evokit
from evokit.archive import Archive
from evokit.agents import CodingAgent, CodingAgentOperator
from evokit.files import Project
from evokit.loop import EvalFn, Limits
from evokit.records import Individual
from evokit.history import History
from evokit.strategies import Greedy


async def autoresearch(
    project: Project,
    eval_fn: EvalFn,
    agent: CodingAgent,
    *,
    history: History,
    archive: Archive,
    instructions: str | None = None,
    objective: str = "val_bpb",
    results: str | Path | None = "results.tsv",
    max_evaluations: int | None = None,
    max_cost: float | None = None,
    max_failures: int | None = 10,
) -> Individual | None:
    """Edit, evaluate, keep or revert, and log every attempt to `results` (a TSV); never stop,
    unless `max_evaluations` (counted in this call) is given, the run has spent `max_cost` US dollars
    (checked between changes, so the last session can go over), or the agent fails `max_failures`
    times in a row.

    The caller passes a seeded archive, usually SimplerBestArchive, and an eval_fn wrapped in
    files.with_lines so the archive's size objective exists. A fresh agent session makes each change.
    """
    strategy = Greedy()
    operator = CodingAgentOperator(agent, project, history=history, instructions=instructions)
    limits = Limits(max_cost=max_cost, max_failures=max_failures)
    score = evokit.as_eval_fn(eval_fn)
    done = 0

    def stop_reason() -> str | None:
        if max_evaluations is not None and done >= max_evaluations:
            return f"{max_evaluations} evaluations"
        return limits.reached(history)

    await operator.check()
    with history.storage.lock("driver"):
        while not stop_reason():
            asked = await evokit.ask(history, archive, strategy, operator)
            if asked.exhausted:
                break
            evaluations = await asyncio.gather(*map(score, asked.individuals))
            admitted = {i.id for i in await evokit.tell(history, archive, asked.individuals, evaluations)}
            done += len(evaluations)
            if results is not None:
                with open(results, "a") as tsv:
                    for child, evaluation in zip(asked.individuals, evaluations):
                        child = child.with_evaluation(evaluation)
                        status = "keep" if child.id in admitted else "discard" if evaluation.objectives else "crash"
                        tsv.write(row(child, status, objective))
    history.note("stop", stop_reason() or "nothing left to select")
    return archive.best()


def row(individual: Individual, status: str, objective: str) -> str:
    """One line of results.tsv: id, objective, status, and the version's name."""
    value = individual.objective(objective)
    name = individual.name.replace("\t", " ")
    return f"{individual.id}\t{'' if value is None else f'{value:.6f}'}\t{status}\t{name}\n"
