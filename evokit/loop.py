"""What every recipe is built from:

    @evokit.eval_fn(timeout="10m")
    def score(individual) -> Evaluation: ...

    history = evokit.History(storage)
    asked = await evokit.ask(history, archive, strategy, operator)
    evaluations = await asyncio.gather(*map(score, asked.individuals))
    admitted = await evokit.tell(history, archive, asked.individuals, evaluations)
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import logging
import traceback
from concurrent.futures import Executor
from dataclasses import dataclass, replace
from typing import Awaitable, Callable, Iterable

from evokit.archive import Archive
from evokit.errors import OperatorError
from evokit.history import History
from evokit.operator import Operator
from evokit.records import Asked, Evaluation, Failure, Individual
from evokit.strategy import Strategy
from evokit.util import new_id, parse_duration, truncate

log = logging.getLogger("evokit")

FEEDBACK_LIMIT = 4000

EvalFn = Callable[[Individual], "Evaluation | Awaitable[Evaluation]"]
"""`eval_fn(individual) -> Evaluation`, sync or async."""


def is_async(fn: Callable) -> bool:
    return inspect.iscoroutinefunction(fn) or inspect.iscoroutinefunction(getattr(fn, "__call__", None))


def eval_fn(fn: EvalFn | None = None, *, timeout: str | float | None = None, executor: Executor | None = None):
    """Make `fn(individual) -> Evaluation`, sync or async, an async eval function that never raises:
    a crash, a timeout or a wrong return type becomes an evaluation with no objectives.

    A sync `fn` runs in `executor` (default: threads), so a slow evaluation does not block others.
    Usable bare (`@eval_fn`) or with options (`@eval_fn(timeout="10m")`).
    """

    def wrap(fn: EvalFn) -> Callable[[Individual], Awaitable[Evaluation]]:
        limit = parse_duration(timeout)

        @functools.wraps(fn)
        async def wrapped(individual: Individual) -> Evaluation:
            try:
                if is_async(fn):
                    call = fn(individual)
                else:
                    call = asyncio.get_running_loop().run_in_executor(executor, fn, individual)
                evaluation = await asyncio.wait_for(call, limit)
            except asyncio.TimeoutError:
                return Evaluation(feedback=f"evaluation timed out after {timeout}")
            except Exception:
                return Evaluation(feedback=truncate(traceback.format_exc(), FEEDBACK_LIMIT))
            if not isinstance(evaluation, Evaluation):
                return Evaluation(feedback=f"eval_fn must return an Evaluation, not {type(evaluation).__name__}")
            return evaluation

        wrapped.is_eval_fn = True
        return wrapped

    return wrap if fn is None else wrap(fn)


def as_eval_fn(fn: EvalFn) -> Callable[[Individual], Awaitable[Evaluation]]:
    """`fn` decorated with eval_fn, unless it already is."""
    return fn if getattr(fn, "is_eval_fn", False) else eval_fn(fn)


async def ask(history: History, archive: Archive, strategy: Strategy, operator: Operator, n: int = 1) -> Asked:
    """Up to n individuals to evaluate. Children a crash left unevaluated come first; otherwise select n
    times and create from each selection. Every creation and failure is recorded with its cost."""
    if pending := history.claim_pending(n):
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
                history.write_logs(folder, exc.logs)
                reason += f"\n\nlogs: {folder}/"
            history.note("failed", reason, exc.usage)
            continue
        parents = tuple(p.id for p in selection.parents)
        for child in offspring.children:
            history.write_logs(f"artifacts/{child.id}", offspring.logs)
        history.record_created(offspring.children, parents, offspring.usage)
        for child in offspring.children:
            log.info("created %s from %s", child.id, ", ".join(parents) or "nothing")
        individuals += offspring.children
    return Asked(tuple(individuals), tuple(failures))


@dataclass(frozen=True)
class Limits:
    """When a loop stops, counted over the whole run (seeds included), so they hold across restarts.

    `max_failures` stops an operator that fails every time: failed calls are neither evaluations
    nor (often) costly, so without it such a loop never reaches the other limits.
    """

    max_evaluations: int | None = None
    max_cost: float | None = None
    max_failures: int | None = 10

    def reached(self, history: History) -> str | None:
        """Why the loop should stop, or None."""
        if self.max_evaluations is not None and history.counts()["evaluated"] >= self.max_evaluations:
            return f"{self.max_evaluations} evaluations"
        if self.max_cost is not None and history.usage()["cost_usd"] >= self.max_cost:
            return f"${self.max_cost:g} spent"
        if self.max_failures is not None and history.failures_in_a_row() >= self.max_failures:
            last = truncate(history.last_note("failed") or "", 1000)
            return f"{self.max_failures} operator failures in a row; the last: {last}"
        return None

    def remaining_cost(self, history: History) -> float | None:
        """US dollars left under `max_cost`, or None if there is no cost limit."""
        if self.max_cost is None:
            return None
        return max(0.0, self.max_cost - history.usage()["cost_usd"])


async def tell(
    history: History, archive: Archive, individuals: Iterable[Individual], evaluations: Iterable[Evaluation]
) -> list[Individual]:
    """Give each individual its evaluation, store the evaluation's logs in its artifact folder, add it
    to the archive and record it. Returns the ones the archive admitted, with their evaluations."""
    admitted = []
    for individual, evaluation in zip(individuals, evaluations, strict=True):
        history.write_logs(f"artifacts/{individual.id}", evaluation.logs)
        individual = individual.with_evaluation(replace(evaluation, logs={}))
        added = bool(archive.add(individual))
        history.record_evaluated(individual, added=added)
        scores = ", ".join(f"{k}={v:.6g}" for k, v in evaluation.objectives.items()) or "failed"
        log.info("evaluated %s: %s%s", individual.id, scores, " (added)" if added else "")
        if added:
            admitted.append(individual)
    return admitted


async def submit(history: History, archive: Archive, eval_fn: EvalFn, individual: Individual) -> Individual:
    """Record `individual` as created, score it with `eval_fn` and offer it to the archive: how an
    agent keeps a version, since the score always comes from `eval_fn`, never from the caller.
    Returns it with its evaluation and info["admitted"]."""
    history.record_created([individual], individual.parents)
    evaluation = await as_eval_fn(eval_fn)(individual)
    admitted = await tell(history, archive, [individual], [evaluation])
    return individual.with_evaluation(replace(evaluation, logs={})).with_info(admitted=bool(admitted))


__all__ = ["EvalFn", "Limits", "as_eval_fn", "ask", "eval_fn", "submit", "tell"]
