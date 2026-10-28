"""autoresearch (Karpathy, 2026): one lineage; keep a change only if it is better, or ties and is smaller."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, replace
from pathlib import Path

from evokit.agents import Agent
from evokit.archive import Archive
from evokit.archives import SimplerBestArchive
from evokit.errors import EnvironmentModified
from evokit.files import with_lines
from evokit.records import Individual
from evokit.task import Task
from evokit.util import truncate

RESULTS = "results.tsv"

CHANGE = """Make one change to the program that you expect to improve its score. Score it before you
finish; if your change made it worse, put back the best version you measured."""


@dataclass
class Change:
    """What this version is."""

    title: str
    description: str = ""


async def autoresearch(
    task: Task,
    agent: Agent,
    *,
    dir: str | Path | None = None,
    archive: Archive | None = None,
    max_cost: float | None = None,
    max_evaluations: int | None = None,
) -> Individual | None:
    """Edit, evaluate, keep or revert, and log every attempt to results.tsv in the experiment;
    never stop, unless a limit is reached (checked between changes, so the last one can go over).

    A fresh agent session makes each change, starting from the best. `archive` defaults to
    SimplerBestArchive, which breaks ties by the code's size: for a Files genome, a "lines"
    objective is added to every evaluation.
    """
    if task.files is not None:
        task = replace(task, evaluate=with_lines(task.evaluate))
    archive = archive if archive is not None else SimplerBestArchive(size="lines")
    experiment = task.create_experiment(dir, archive=archive, max_cost=max_cost, max_evaluations=max_evaluations)
    await agent.check()
    while not experiment.limits.reached():
        parent = archive.top(1)[0]
        started = time.monotonic()
        async with experiment.start(agent, start=parent, submit=False) as directory:
            turn = await agent.prompt(CHANGE, output=Change)
            try:
                genome = await asyncio.to_thread(task.read, directory)
            except EnvironmentModified as exc:
                genome, turn = None, replace(turn, error=f"{exc.message}. {exc.hint}")
        if genome is None or genome == parent.genome:
            experiment.note("failed", turn.error or f"the agent made no changes: {truncate(turn.text, 2000)}")
            continue
        info = {"operator": "agent", "session": turn.session}
        if turn.output is None:
            info["agent_error"] = truncate(turn.error or "", 4000)
        title, description = (turn.output.title, turn.output.description) if turn.output else ("", "")
        child = Individual.new(genome, [parent.id], info, name=title, description=truncate(description, 4000))
        child = await experiment.submit(child, {"seconds": time.monotonic() - started})
        status = "keep" if child.info["admitted"] else "discard" if child.evaluation.objectives else "crash"
        experiment.storage.append(RESULTS, row(child, status, task.objective).encode())
    experiment.note("stop", experiment.limits.reached())
    return experiment.best()


def row(individual: Individual, status: str, objective: str) -> str:
    """One line of results.tsv: id, objective, status, and the version's name."""
    value = individual.objective(objective)
    name = individual.name.replace("\t", " ")
    return f"{individual.id}\t{'' if value is None else f'{value:.6f}'}\t{status}\t{name}\n"
