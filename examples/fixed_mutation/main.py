"""Classic fixed mutations as EvoKit operators: no model, no agent, no cost.

Maximizes the negated Rastrigin function in 10 dimensions with hill climbing. Two mutations
(small Gaussian steps and resetting one coordinate) are mixed by Choice.

    python examples/fixed_mutation/main.py
"""

from __future__ import annotations

import asyncio
import math
import random
from collections import Counter

import evokit as ek
from evokit import history
from evokit.archives import BestArchive
from evokit.operators import Choice
from evokit.storages import MemoryStorage
from evokit.strategies import Greedy

DIMENSIONS = 10
BOUND = 5.12


def rastrigin(x: list[float]) -> float:
    """Higher is better, so the negated Rastrigin value; the optimum is 0 at the origin."""
    return -(10 * len(x) + sum(v * v - 10 * math.cos(2 * math.pi * v) for v in x))


def gaussian(sigma: float, rng: random.Random):
    """Add N(0, sigma) noise to every coordinate, clipped to the bounds."""
    return lambda genome: [min(BOUND, max(-BOUND, v + rng.gauss(0, sigma))) for v in genome]


def reset(rng: random.Random):
    """Set one random coordinate to a uniform value in the bounds (escapes local optima)."""

    def mutate(genome):
        genome = list(genome)
        genome[rng.randrange(len(genome))] = rng.uniform(-BOUND, BOUND)
        return genome

    return mutate


def eval_fn(individual: ek.Individual, workdir) -> ek.Evaluation:
    return ek.Evaluation({"score": rastrigin(individual.genome)})


async def main(evaluations: int = 5000) -> None:
    rng = random.Random(0)
    seed = [rng.uniform(-BOUND, BOUND) for _ in range(DIMENSIONS)]
    storage = MemoryStorage()
    archive = BestArchive(storage=storage)
    operator = Choice(
        {"gaussian": ek.Mutate(gaussian(0.1, random.Random(1))), "reset": ek.Mutate(reset(random.Random(2)))}, seed=3
    )
    strategy = Greedy()

    await ek.tell(storage, archive, await ek.evaluate(storage, [ek.Individual.new(seed)], eval_fn=eval_fn))
    while history.counts(storage)["evaluated"] < evaluations:
        asked = await ek.ask(storage, archive, strategy, operator)
        await ek.tell(storage, archive, await ek.evaluate(storage, asked.individuals, eval_fn=eval_fn))

    print(f"seed score: {rastrigin(seed):.3f}")
    print(f"best score: {archive.best().objective('score'):.3f} after {history.counts(storage)['evaluated']} evaluations")
    made = Counter(i.info.get("operator") for i in history.evaluated(storage) if i.parents)
    kept = Counter(i.info.get("operator") for i in history.admitted(storage) if i.parents)
    print("admission rate:", {name: f"{kept[name]}/{made[name]}" for name in made})


if __name__ == "__main__":
    asyncio.run(main())
