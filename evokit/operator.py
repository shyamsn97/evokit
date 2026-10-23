"""Operator: how are new individuals made? (variation)"""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, ClassVar

from evokit.records import Individual, Offspring, Selection

if TYPE_CHECKING:
    from evokit.archive import Archive


class Operator(ABC):
    """Makes children from a selection, and reports what that cost in `Offspring.usage`
    ({"cost_usd", "tokens", "seconds", ...}). Raises OperatorError(reason, usage) when it tried
    and produced nothing. `archive` is read-only: for checks against the current members
    (novelty, duplicates); most operators ignore it."""

    name: ClassVar[str] = "operator"

    @abstractmethod
    async def create(self, selection: Selection, archive: "Archive") -> Offspring: ...

    async def check(self) -> None:
        """Raise UsageError if this operator cannot create anything (bad credentials, an unknown
        model). Loops call it once, before they start. A no-op by default."""


class Mutate(Operator):
    """The simplest operator: a function from the parents' genomes to a new genome, sync or async.

    `fn(genome)` for one parent; with several parents (e.g. Tournament(parents=2)) it gets them all:
    `fn(genome_a, genome_b)`.
    """

    name = "mutate"

    def __init__(self, fn: Callable[..., Any], name: str | None = None):
        self.fn = fn
        self.label = name or getattr(fn, "__name__", "mutate")

    async def create(self, selection: Selection, archive: "Archive") -> Offspring:
        genome = self.fn(*(p.genome for p in selection.parents))
        if inspect.isawaitable(genome):
            genome = await genome
        child = Individual.new(genome, [p.id for p in selection.parents], {"operator": self.label})
        return Offspring((child,))
