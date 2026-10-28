"""Evaluate functions: `evaluate(individual) -> Evaluation`, sync or async, the one place scores come from.

    @evokit.evaluator(timeout="10m")
    def score(individual) -> Evaluation: ...
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import traceback
from concurrent.futures import Executor
from typing import Awaitable, Callable

from evokit.records import Evaluation, Individual
from evokit.util import parse_duration, truncate

FEEDBACK_LIMIT = 4000

Evaluate = Callable[[Individual], "Evaluation | Awaitable[Evaluation]"]
"""`evaluate(individual) -> Evaluation`, sync or async."""


def is_async(fn: Callable) -> bool:
    return inspect.iscoroutinefunction(fn) or inspect.iscoroutinefunction(getattr(fn, "__call__", None))


def evaluator(fn: Evaluate | None = None, *, timeout: str | float | None = None, executor: Executor | None = None):
    """Make `fn(individual) -> Evaluation`, sync or async, an async evaluate function that never
    raises: a crash, a timeout or a wrong return type becomes an evaluation with no objectives.

    A sync `fn` runs in `executor` (default: threads), so a slow evaluation does not block others.
    Usable bare (`@evaluator`) or with options (`@evaluator(timeout="10m")`).
    """

    def wrap(fn: Evaluate) -> Callable[[Individual], Awaitable[Evaluation]]:
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
                return Evaluation(feedback=f"evaluate must return an Evaluation, not {type(evaluation).__name__}")
            return evaluation

        wrapped.is_evaluator = True
        return wrapped

    return wrap if fn is None else wrap(fn)


def as_evaluate(fn: Evaluate) -> Callable[[Individual], Awaitable[Evaluation]]:
    """`fn` decorated with evaluator, unless it already is."""
    return fn if getattr(fn, "is_evaluator", False) else evaluator(fn)


__all__ = ["Evaluate", "as_evaluate", "evaluator"]
