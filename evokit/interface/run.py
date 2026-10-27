"""The `evokit` commands: plain functions over an opened Run, served as the CLI, the MCP server
and the skill's command reference.

Each takes the Run first; `run_tools(run)` binds them, adds the read tools agents get (log, show,
diff, grep, over the run's history) and the task factory's own, and names them as commands.
"""

from __future__ import annotations

import asyncio
import dataclasses
import datetime as _dt
import functools
from pathlib import Path
from typing import Any, Callable

import evokit
from evokit.agents.tools import READ_TOOLS, tool_name
from evokit.errors import UsageError
from evokit.files.tree import Files
from evokit.loop import Limits
from evokit.records import Evaluation, Individual
from evokit.task import Run
from evokit.util import background, truncate


def _component(run: Run, name: str) -> Any:
    value = getattr(run, name)
    if value is None:
        hint = "set task.evaluate in config.yaml" if name == "eval_fn" else "open the run with its task (--task TASK_DIR)"
        raise UsageError(f"this run has no {name.replace('_', ' ')}", hint=hint)
    return value


def status(run: Run) -> dict:
    """The archive's size and best individual, and what the run has created, evaluated and spent."""
    archive = _component(run, "archive")
    return {
        "archive": type(archive).__name__,
        "members": len(archive),
        "best": archive.best(),
        "strategy": type(run.strategy).__name__,
        "operator": None if run.operator is None else type(run.operator).__name__,
        "counts": dict(run.history.counts()),
        "usage": dict(run.history.usage()),
        "pending": [i.id for i in run.history.pending()],
    }


def select_individuals(run: Run, n: int = 1) -> list:
    """Ask the strategy what to build on next: each selection has parents and inspirations.

    An empty list means the archive is empty: seed it with a genome that has no parent.
    """
    return background.run(_component(run, "strategy").select(_component(run, "archive"), n))


def list_individuals(run: Run) -> list[Individual]:
    """The archive's members (best first, for ranked archives)."""
    return _component(run, "archive").members()


def best_individual(run: Run) -> Individual | None:
    """The archive's best member."""
    return _component(run, "archive").best()


def list_events(run: Run, limit: int | None = 100) -> list[dict]:
    """The run's history, oldest first: created, evaluated (and whether admitted), discarded, notes."""
    records = [_summary(r) for r in run.history.records()]
    return records[-limit:] if limit else records


def checkout_individual(run: Run, id: str, directory: str) -> dict:
    """Write the full project (environment plus this individual's genome) into a new directory to edit.

    Edit the genome files there, then evaluate-directory.
    """
    individual, project = run.history.lookup(id), _component(run, "task").project
    if not isinstance(individual.genome, Files):
        raise UsageError(f"individual {individual.id} does not have a Files genome")
    path = Path(directory)
    if path.exists() and any(path.iterdir()):
        raise UsageError(f"directory is not empty: {path}", hint="check out into a new directory")
    project.write(individual.genome, path)
    return {
        "directory": str(path.resolve()),
        "individual": individual.id,
        "genome": project.editable,
        "hint": f"Edit only files matching {project.editable}, then `evokit evaluate-directory {path} --parent {individual.id}`.",
    }


def evaluate_directory(run: Run, directory: str, parents: list[str] | None = None, name: str = "", description: str = "") -> Individual:
    """Make an individual from a directory's genome files, evaluate it, and offer it to the archive.

    `name` is one line saying what this version is, `description` what changed and why. The
    result has its evaluation and info.admitted. Edits to environment files (anything outside
    the genome) are rejected.
    """
    path = Path(directory)
    if not path.is_dir():
        raise UsageError(f"not a directory: {path}")
    parents = [run.history.lookup(p).id for p in parents or []]
    child = Individual.new(_component(run, "task").project.read(path), parents, name=name, description=description)
    return background.run(evokit.submit(run.history, _component(run, "archive"), _component(run, "eval_fn"), child))


def run_loop(
    run: Run, workers: int = 1, max_evaluations: int | None = None, max_cost: float | None = None, max_failures: int | None = 10
) -> Individual | None:
    """Run ask, evaluate and tell with the run's operator until a limit is reached or nothing can be selected.

    Limits count the whole run, seeds included; `max_failures` counts operator failures in a
    row. The operator is checked first, and only one process at a time can run a run.
    Returns the best individual.
    """
    if run.operator is None:
        raise UsageError("this run has no operator", hint="you are the operator: use select-individuals and evaluate-directory")
    archive = _component(run, "archive")
    if not len(archive):
        raise UsageError("the archive is empty", hint="a run is seeded with its task's files when it is created")
    return background.run(
        evolve(run, workers, Limits(max_evaluations, max_cost, max_failures))
    )


async def evolve(run: Run, workers: int, limits: Limits) -> Individual | None:
    """`workers` loops of ask, evaluate and tell over the run's components, until `limits` are reached."""
    history, archive, strategy, operator = run.history, run.archive, run.strategy, run.operator
    score = evokit.as_eval_fn(_component(run, "eval_fn"))

    async def worker() -> None:
        while not limits.reached(history):
            asked = await evokit.ask(history, archive, strategy, operator)
            if asked.exhausted:
                return
            evaluations = await asyncio.gather(*map(score, asked.individuals))
            await evokit.tell(history, archive, asked.individuals, evaluations)

    await operator.check()
    with history.storage.lock("driver"):
        await asyncio.gather(*(worker() for _ in range(workers)))
    history.note("stop", limits.reached(history) or "nothing left to select")
    return archive.best()


COMMANDS: dict[str, Callable] = {
    "status": status,
    "select-individuals": select_individuals,
    "list-individuals": list_individuals,
    "best-individual": best_individual,
    "list-events": list_events,
    "checkout-individual": checkout_individual,
    "evaluate-directory": evaluate_directory,
    "run": run_loop,
}
"""The commands on a run opened with its task; each takes the Run first."""

READ = {tool_name(t): t for t in READ_TOOLS}
"""The commands that work on any run directory, task or not; each takes the run's History first."""

CLI_ONLY = ("run",)
"""Commands the MCP server leaves out: `run` drives the whole search, which is not an agent's tool."""


def run_tools(run: Run) -> dict[str, Callable]:
    """Every command for `run`, by name, bound to it: plain synchronous functions (also for a
    script, a notebook or an RLM). A run opened only to be read has the read commands."""
    tools = {name: functools.partial(fn, run.history) for name, fn in READ.items()}
    if run.task is not None:
        tools |= {name: functools.partial(fn, run) for name, fn in COMMANDS.items()}
    return tools | {tool_name(t).replace("_", "-"): t for t in run.tools}


def _summary(record: dict) -> dict:
    """A history record without encoded genomes, for reading."""
    kind = record["kind"]
    out: dict[str, Any] = {"kind": kind, "time": record.get("time")}
    if kind == "created":
        out["ids"] = [c["id"] for c in record["children"]]
        out["parents"] = record.get("parents", [])
    elif kind == "evaluated":
        individual = record["individual"]
        evaluation = individual.get("evaluation") or {}
        out.update(id=individual["id"], added=record["added"], objectives=evaluation.get("objectives", {}))
        if evaluation.get("feedback"):
            out["feedback"] = truncate(evaluation["feedback"], 500)
    elif kind == "discarded":
        out.update(id=record["id"], reason=record["reason"])
    elif kind == "note":
        out.update(note=record["note"], text=truncate(record.get("text", ""), 500))
    if record.get("usage"):
        out["usage"] = record["usage"]
    return out


def encode(value: Any) -> Any:
    """Convert records and results to JSON-compatible data."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Individual):
        return {
            "id": value.id,
            "name": value.name,
            "description": value.description,
            "genome": encode(value.genome),
            "parents": list(value.parents),
            "evaluation": encode(value.evaluation),
            "info": encode(dict(value.info)),
        }
    if isinstance(value, Evaluation):
        return value.to_dict()
    if hasattr(value, "describe"):
        return encode(value.describe())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (_dt.datetime, _dt.date)):
        return value.isoformat()
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: encode(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {str(k): encode(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [encode(v) for v in value]
    if hasattr(value, "to_dict"):
        return encode(value.to_dict())
    return repr(value)
