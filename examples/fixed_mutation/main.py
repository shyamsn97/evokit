"""Classic fixed mutations: no model, no agent, no cost.

Maximizes the negated Rastrigin function in 10 dimensions with a genetic algorithm
(`recipes.ga`). Two mutations (small Gaussian steps and resetting one coordinate) are mixed.

    python examples/fixed_mutation/main.py
"""

from __future__ import annotations

import asyncio
import math
import random

import evokit as ek

DIMENSIONS = 10
BOUND = 5.12


def rastrigin(x: list[float]) -> float:
    """Higher is better, so the negated Rastrigin value; the optimum is 0 at the origin."""
    return -(10 * len(x) + sum(v * v - 10 * math.cos(2 * math.pi * v) for v in x))


def gaussian(genome: list[float], sigma: float, rng: random.Random) -> list[float]:
    """Add N(0, sigma) noise to every coordinate, clipped to the bounds."""
    return [min(BOUND, max(-BOUND, v + rng.gauss(0, sigma))) for v in genome]


def reset(genome: list[float], rng: random.Random) -> list[float]:
    """Set one random coordinate to a uniform value in the bounds (escapes local optima)."""
    genome = list(genome)
    genome[rng.randrange(len(genome))] = rng.uniform(-BOUND, BOUND)
    return genome


def mutation(rng: random.Random, reset_rate: float = 0.2):
    """A Gaussian step, or now and then (`reset_rate`) a reset."""

    def mutate(genome: list[float]) -> list[float]:
        return reset(genome, rng) if rng.random() < reset_rate else gaussian(genome, 0.1, rng)

    return mutate


@ek.evaluator
def score(individual: ek.Individual) -> ek.Evaluation:
    return ek.Evaluation({"score": rastrigin(individual.genome)})


async def main(evaluations: int = 5000) -> None:
    rng = random.Random(0)
    seed = [rng.uniform(-BOUND, BOUND) for _ in range(DIMENSIONS)]
    best = await ek.recipes.ga(ek.Task(seed, score), mutation(random.Random(1)), max_evaluations=evaluations, seed=2)

    print(f"seed score: {rastrigin(seed):.3f}")
    print(f"best score: {best.objective('score'):.3f}")


if __name__ == "__main__":
    asyncio.run(main())
