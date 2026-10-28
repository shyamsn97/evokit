"""The `evokit` commands: plain functions over an opened Experiment, served as the CLI, the MCP
server and the skill's command reference.

Each takes the Experiment first; `experiment_tools(experiment)` binds them and names them as
commands. The read commands (stats, log, show, diff, grep, notes) are the same functions agents
get as tools, and work on any experiment directory; the others need its task. `evokit run`
(in cli.py) runs a recipe.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Callable

from evokit.agents.tools import tool_name
from evokit.errors import UsageError
from evokit.experiment import READ_TOOLS, Experiment
from evokit.records import Individual
from evokit.util import background


def _archive(experiment: Experiment):
    if experiment.archive is None:
        raise UsageError("this experiment's archive cannot be reopened", hint="continue it from Python, with the archive it was made with")
    return experiment.archive


def status(experiment: Experiment) -> dict:
    """The archive's size and best individual, and what the experiment has created, evaluated and spent."""
    archive = _archive(experiment)
    return {
        "task": str(experiment.task.directory) if experiment.task and experiment.task.directory else None,
        "archive": type(archive).__name__,
        "members": len(archive),
        "best": experiment.best(),
        "counts": dict(experiment.counts()),
        "usage": dict(experiment.usage()),
        "pending": [i.id for i in experiment.pending()],
    }


def list_individuals(experiment: Experiment) -> list[Individual]:
    """The archive's members (best first, for ranked archives)."""
    return _archive(experiment).members()


def checkout_individual(experiment: Experiment, id: str, directory: str) -> dict:
    """Write this individual (with the task's fixed files around it) into a new directory to edit.

    Edit the genome files there, then evaluate-directory.
    """
    individual, task = experiment.lookup(id), experiment._task()
    path = Path(directory)
    if path.exists() and any(path.iterdir()):
        raise UsageError(f"directory is not empty: {path}", hint="check out into a new directory")
    task.checkout(individual.genome, path)
    return {
        "directory": str(path.resolve()),
        "individual": individual.id,
        "genome": task.editable,
        "hint": f"Edit only files matching {task.editable}, then `evokit evaluate-directory {path} --parent {individual.id}`.",
    }


def evaluate_directory(experiment: Experiment, directory: str, parents: list[str] | None = None, *, name: str = "", description: str = "") -> Individual:
    """Make an individual from a directory's genome files, evaluate it, and offer it to the archive.

    `name` is one line saying what this version is, `description` what changed and why. The
    result has its evaluation and info.admitted. Edits to environment files (anything outside
    the genome) are rejected.
    """
    path = Path(directory)
    if not path.is_dir():
        raise UsageError(f"not a directory: {path}")
    _archive(experiment)
    parents = [experiment.lookup(p).id for p in parents or []]
    child = Individual.new(experiment._task().read(path), parents, name=name, description=description)
    return background.run(experiment.submit(child))


COMMANDS: dict[str, Callable] = {
    "status": status,
    "list-individuals": list_individuals,
    "checkout-individual": checkout_individual,
    "evaluate-directory": evaluate_directory,
}
"""The commands on an experiment opened with its task; each takes the Experiment first."""

READ = {tool_name(t): t for t in READ_TOOLS}
"""The commands that work on any experiment directory, task or not; each takes the Experiment first."""


def experiment_tools(experiment: Experiment) -> dict[str, Callable]:
    """Every command for `experiment`, by name, bound to it: plain synchronous functions (also
    for a script, a notebook or an RLM). One opened without its task has the read commands."""
    tools = {name: functools.partial(fn, experiment) for name, fn in READ.items()}
    if experiment.task is not None:
        tools |= {name: functools.partial(fn, experiment) for name, fn in COMMANDS.items()}
    return tools
