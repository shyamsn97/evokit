"""RunCommands: the run's components behind agent-facing commands (CLI, MCP, Python tools)."""

from __future__ import annotations

import asyncio
from typing import Any

import evokit
from evokit import history
from evokit.archive import Archive
from evokit.errors import UsageError
from evokit.interface.commands import command
from evokit.interface.history import HistoryCommands
from evokit.loop import EvalFn
from evokit.operator import Operator
from evokit.records import Individual
from evokit.storage import Storage
from evokit.strategy import Strategy
from evokit.util import background, truncate


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


class RunCommands:
    """status, select-individuals, evaluate-genome, list/best-individual(s), list-events, run; `history` has log, show, diff, grep."""

    def __init__(
        self,
        storage: Storage,
        archive: Archive,
        strategy: Strategy,
        operator: Operator | None = None,
        eval_fn: EvalFn | None = None,
        *,
        task: Any = None,
        extras: tuple = (),
    ):
        self.storage, self.archive, self.strategy, self.operator, self.eval_fn = storage, archive, strategy, operator, eval_fn
        self.task, self.extras = task, tuple(extras)
        self.history = HistoryCommands(storage, archive)

    def lookup(self, id: str) -> Individual:
        """An individual by id: anything in the history, else an archive member."""
        return self.history.lookup(id)

    def _best(self) -> Individual | None:
        best = getattr(self.archive, "best", None)
        return best() if callable(best) else None

    def _eval_fn(self) -> EvalFn:
        if self.eval_fn is None:
            raise UsageError("this run has no eval function", hint="set task.evaluate in config.yaml")
        return self.eval_fn

    @command
    def status(self) -> dict:
        """The archive's size and best individual, and what the run has created, evaluated and spent."""
        return {
            "archive": type(self.archive).__name__,
            "members": len(self.archive),
            "best": self._best(),
            "strategy": type(self.strategy).__name__,
            "operator": None if self.operator is None else type(self.operator).__name__,
            "counts": dict(history.counts(self.storage)),
            "usage": dict(history.usage(self.storage)),
            "pending": [i.id for i in history.pending(self.storage)],
        }

    @command
    def select_individuals(self, n: int = 1) -> list:
        """Ask the strategy what to build on next: each selection has parents and inspirations.

        An empty list means the archive is empty: seed it with a genome that has no parent.
        """
        return background.run(self.strategy.select(self.archive, n))

    @command
    def evaluate_genome(
        self, genome: Any, parents: list[str] | None = None, name: str = "", description: str = ""
    ) -> Individual:
        """Make an individual from a genome, evaluate it, and offer it to the archive.

        With no parents it is a seed. `name` is one line saying what this version is, `description`
        what changed and why. The result has its evaluation and info.admitted.
        """
        parents = list(parents or [])
        for parent in parents:
            self.lookup(parent)
        child = Individual.new(genome, parents, name=name, description=description)
        if parents:
            history.record_created(self.storage, [child], parents)
        return background.run(self._evaluate_and_tell([child]))[0]

    async def _evaluate_and_tell(self, individuals: list[Individual]) -> list[Individual]:
        evaluated = await evokit.evaluate(self.storage, individuals, eval_fn=self._eval_fn())
        admitted = {i.id for i in await evokit.tell(self.storage, self.archive, evaluated)}
        return [i.with_info(admitted=i.id in admitted) for i in evaluated]

    def seed(self, seed: Individual) -> Individual:
        """Evaluate `seed` and add it to the archive. Raises UsageError, recording nothing, if it fails its evaluation."""
        (individual,) = background.run(evokit.evaluate(self.storage, [seed], eval_fn=self._eval_fn()))
        if not individual.evaluation.objectives:
            raise UsageError(
                f"the seed failed its evaluation: {truncate(individual.evaluation.feedback, 2000)}",
                hint="fix the task's files or its evaluator, then run the command again",
            )
        background.run(evokit.tell(self.storage, self.archive, [individual]))
        return individual

    @command
    def list_individuals(self) -> list[Individual]:
        """The archive's members (best first, for ranked archives)."""
        return self.archive.members()

    @command
    def best_individual(self) -> Individual | None:
        """The archive's best member."""
        return self._best()

    @command
    def list_events(self, limit: int | None = 100) -> list[dict]:
        """The run's history, oldest first: created, evaluated (and whether admitted), discarded, notes."""
        records = [_summary(r) for r in history.records(self.storage)]
        return records[-limit:] if limit else records

    @command(surfaces=("cli", "tools"))
    def run(
        self, workers: int = 1, max_evaluations: int | None = None, max_cost: float | None = None, max_failures: int | None = 10
    ) -> Individual | None:
        """Run ask, evaluate and tell with the run's operator until a limit is reached or nothing can be selected.

        Limits count the whole run, seeds included; `max_failures` counts operator failures in a
        row. The operator is checked first, and only one process at a time can run a run.
        Returns the best individual.
        """
        if self.operator is None:
            raise UsageError("this run has no operator", hint="you are the operator: use select-individuals and evaluate-directory")
        if not len(self.archive):
            raise UsageError("the archive is empty", hint="a run is seeded with its task's files when it is created")
        eval_fn = self._eval_fn()
        limits = evokit.Limits(max_evaluations, max_cost, max_failures)

        async def worker() -> None:
            while not limits.reached(self.storage):
                asked = await evokit.ask(self.storage, self.archive, self.strategy, self.operator)
                if asked.exhausted:
                    return
                evaluated = await evokit.evaluate(self.storage, asked.individuals, eval_fn=eval_fn)
                await evokit.tell(self.storage, self.archive, evaluated)

        async def go() -> None:
            await asyncio.gather(*(worker() for _ in range(workers)))

        background.run(self.operator.check())
        with self.storage.lock("driver"):
            background.run(go())
        history.note(self.storage, "stop", limits.reached(self.storage) or "nothing left to select")
        return self._best()
