"""autoresearch (Karpathy, 2026): one lineage; keep a change only if it is better, or ties and is smaller."""

from __future__ import annotations

from pathlib import Path

import evokit
from evokit import history
from evokit.archive import Archive
from evokit.files import CodingAgent, CodingAgentOperator, Project
from evokit.loop import EvalFn, Limits
from evokit.records import Individual
from evokit.storage import Storage
from evokit.strategies import Greedy


async def autoresearch(
    project: Project,
    eval_fn: EvalFn,
    agent: CodingAgent,
    *,
    storage: Storage,
    archive: Archive,
    instructions: str | None = None,
    objective: str = "val_bpb",
    results: str | Path | None = "results.tsv",
    max_evaluations: int | None = None,
    max_failures: int | None = 10,
) -> Individual | None:
    """Edit, evaluate, keep or revert, and log every attempt to `results` (a TSV); never stop,
    unless `max_evaluations` (counted in this call) is given or the agent fails `max_failures` times in a row.

    The caller passes a seeded archive, usually SimplerBestArchive, and an eval_fn wrapped in
    files.with_lines so the archive's size objective exists. A fresh agent session makes each change.
    """
    strategy = Greedy()
    operator = CodingAgentOperator(agent, project, instructions=instructions)
    limits = Limits(max_failures=max_failures)
    done = 0

    def stop_reason() -> str | None:
        if max_evaluations is not None and done >= max_evaluations:
            return f"{max_evaluations} evaluations"
        return limits.reached(storage)

    await operator.check()
    with storage.lock("driver"):
        while not stop_reason():
            asked = await evokit.ask(storage, archive, strategy, operator)
            if asked.exhausted:
                break
            evaluated = await evokit.evaluate(storage, asked.individuals, eval_fn=eval_fn)
            admitted = await evokit.tell(storage, archive, evaluated)
            done += len(evaluated)
            if results is not None:
                with open(results, "a") as tsv:
                    for child in evaluated:
                        status = "keep" if child in admitted else "discard" if child.evaluation.objectives else "crash"
                        tsv.write(row(child, status, objective))
    history.note(storage, "stop", stop_reason() or "nothing left to select")
    return archive.best()


def row(individual: Individual, status: str, objective: str) -> str:
    """One line of results.tsv: id, objective, status, and the version's name."""
    value = individual.objective(objective)
    name = individual.name.replace("\t", " ")
    return f"{individual.id}\t{'' if value is None else f'{value:.6f}'}\t{status}\t{name}\n"
