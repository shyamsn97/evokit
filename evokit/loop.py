"""The three functions every recipe is built from:

    asked = await evokit.ask(storage, archive, strategy, operator)
    evaluated = await evokit.evaluate(storage, asked.individuals, eval_fn=eval_fn)
    admitted = await evokit.tell(storage, archive, evaluated)
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import traceback
from concurrent.futures import Executor
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable, Iterable

from evokit import history
from evokit.archive import Archive
from evokit.errors import OperatorError
from evokit.operator import Operator
from evokit.records import Asked, Evaluation, Failure, Individual
from evokit.storage import Storage
from evokit.strategy import Strategy
from evokit.util import new_id, parse_duration, truncate

log = logging.getLogger("evokit")

FEEDBACK_LIMIT = 4000

EvalFn = Callable[[Individual, Path], "Evaluation | Awaitable[Evaluation]"]
"""`eval_fn(individual, workdir) -> Evaluation`, sync or async."""


def is_async(fn: Callable) -> bool:
    return inspect.iscoroutinefunction(fn) or inspect.iscoroutinefunction(getattr(fn, "__call__", None))


async def call_eval_fn(
    eval_fn: EvalFn, individual: Individual, workdir: Path, *, timeout: str | float | None = None, executor: Executor | None = None
) -> Evaluation:
    """Await an async eval_fn; run a sync one in `executor` (default: threads)."""
    if is_async(eval_fn):
        call = eval_fn(individual, workdir)
    else:
        call = asyncio.get_running_loop().run_in_executor(executor, eval_fn, individual, workdir)
    evaluation = await asyncio.wait_for(call, parse_duration(timeout))
    if not isinstance(evaluation, Evaluation):
        raise TypeError(f"eval_fn must return an Evaluation, not {type(evaluation).__name__}")
    return evaluation


async def ask(storage: Storage, archive: Archive, strategy: Strategy, operator: Operator, n: int = 1) -> Asked:
    """Up to n individuals to evaluate. Children a crash left unevaluated come first; otherwise select n
    times and create from each selection. Every creation and failure is recorded with its cost."""
    if pending := history.claim_pending(storage, n):
        return Asked(tuple(pending))
    selections = await strategy.select(archive, n)
    if not selections:
        return Asked((), exhausted=True)
    individuals: list[Individual] = []
    failures: list[Failure] = []
    for selection in selections:
        try:
            offspring = await operator.create(selection, archive)
        except OperatorError as exc:
            failures.append(Failure(selection, exc.reason, exc.usage))
            reason = truncate(exc.reason, FEEDBACK_LIMIT)
            if exc.logs:
                folder = f"failed/{new_id()}"
                _write_logs(storage, folder, exc.logs)
                reason += f"\n\nlogs: {folder}/"
            history.note(storage, "failed", reason, exc.usage)
            continue
        parents = tuple(p.id for p in selection.parents)
        for child in offspring.children:
            _write_logs(storage, f"artifacts/{child.id}", offspring.logs)
        history.record_created(storage, offspring.children, parents, offspring.usage)
        for child in offspring.children:
            log.info("created %s from %s", child.id, ", ".join(parents) or "nothing")
        individuals += offspring.children
    return Asked(tuple(individuals), tuple(failures))


def _write_logs(storage: Storage, folder: str, logs) -> None:
    for name, data in logs.items():
        storage.write(f"{folder}/{name}", data)


async def evaluate(
    storage: Storage,
    individuals: Iterable[Individual],
    *,
    eval_fn: EvalFn,
    timeout: str | float | None = None,
    executor: Executor | None = None,
) -> list[Individual]:
    """Evaluate each individual concurrently, in its artifact folder. A crash or a timeout becomes an
    evaluation with no objectives."""

    async def one(individual: Individual) -> Individual:
        with storage.local(f"artifacts/{individual.id}") as workdir:
            try:
                evaluation = await call_eval_fn(eval_fn, individual, workdir, timeout=timeout, executor=executor)
            except asyncio.TimeoutError:
                evaluation = Evaluation(feedback=f"evaluation timed out after {timeout}")
            except Exception:
                evaluation = Evaluation(feedback=truncate(traceback.format_exc(), FEEDBACK_LIMIT))
        return individual.with_evaluation(evaluation)

    return list(await asyncio.gather(*(one(i) for i in individuals)))


@dataclass(frozen=True)
class Limits:
    """When a loop stops, counted over the whole run (seeds included), so they hold across restarts.

    `max_failures` stops an operator that fails every time: failed calls are neither evaluations
    nor (often) costly, so without it such a loop never reaches the other limits.
    """

    max_evaluations: int | None = None
    max_cost: float | None = None
    max_failures: int | None = 10

    def reached(self, storage: Storage) -> str | None:
        """Why the loop should stop, or None."""
        if self.max_evaluations is not None and history.counts(storage)["evaluated"] >= self.max_evaluations:
            return f"{self.max_evaluations} evaluations"
        if self.max_cost is not None and history.usage(storage)["cost_usd"] >= self.max_cost:
            return f"${self.max_cost:g} spent"
        if self.max_failures is not None and history.failures_in_a_row(storage) >= self.max_failures:
            last = truncate(history.last_note(storage, "failed") or "", 1000)
            return f"{self.max_failures} operator failures in a row; the last: {last}"
        return None


async def tell(storage: Storage, archive: Archive, individuals: Iterable[Individual]) -> list[Individual]:
    """Add evaluated individuals to the archive and record them. Returns the ones the archive admitted."""
    admitted = []
    for individual in individuals:
        if individual.evaluation is None:
            raise ValueError(f"individual {individual.id} has no evaluation; evaluate it before telling it")
        added = bool(archive.add(individual))
        history.record_evaluated(storage, individual, added=added)
        scores = ", ".join(f"{k}={v:.6g}" for k, v in individual.evaluation.objectives.items()) or "failed"
        log.info("evaluated %s: %s%s", individual.id, scores, " (added)" if added else "")
        if added:
            admitted.append(individual)
    return admitted


__all__ = ["EvalFn", "Limits", "ask", "call_eval_fn", "evaluate", "tell"]
