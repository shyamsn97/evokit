from __future__ import annotations

import random
from typing import Mapping

from evokit.errors import OperatorError
from evokit.operator import Operator
from evokit.records import Offspring, Selection


class Choice(Operator):
    """Pick one operator per call, by fixed `weights` (e.g. an LLM ensemble of a fast and a strong
    model). Each child's info["operator"] names the one that made it."""

    name = "choice"

    def __init__(self, operators: Mapping[str, Operator], weights: Mapping[str, float] | None = None, seed: int | None = None):
        if not operators:
            raise ValueError("Choice needs at least one operator")
        self.operators = dict(operators)
        self.weights = {k: float((weights or {}).get(k, 1.0)) for k in self.operators}
        self._rng = random.Random(seed)

    async def create(self, selection: Selection, archive) -> Offspring:
        name = self._rng.choices(list(self.weights), weights=list(self.weights.values()))[0]
        try:
            offspring = await self.operators[name].create(selection, archive)
        except OperatorError as exc:
            raise OperatorError(f"[{name}] {exc.reason}", exc.usage) from exc
        return Offspring(tuple(c.with_info(operator=name) for c in offspring.children), offspring.usage)
