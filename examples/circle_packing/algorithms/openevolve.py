"""AlphaEvolve / OpenEvolve on circle packing: evokit.recipes.openevolve.

OpenEvolve's flagship example is this same task. Islands of MAP-Elites grids (code length by
diversity), a parent plus inspirations in every prompt, a fast and a strong model writing diffs
inside an EVOLVE-BLOCK, a two-stage evaluation cascade, and many evaluations in flight at once.

    cd examples/circle_packing
    python -m algorithms.openevolve --dir runs/openevolve-full-0 --variant full

Variants, one swap each: full, one-island, no-inspirations, single-model, no-grid.
Models: $EVOKIT_FAST_MODEL (default claude-haiku-4-5), $EVOKIT_STRONG_MODEL (claude-sonnet-4-5).
"""

from __future__ import annotations

import asyncio
import os
from typing import Mapping

from dataclasses import replace

from algorithms.common import arguments, model, packing_task, report

import evokit as ek
from evokit import recipes
from evokit.archives import IslandArchive
from evokit.files import ScriptEvaluator
from evokit.recipes.dgm import staged_eval

VARIANTS = ("full", "one-island", "no-inspirations", "single-model", "no-grid")
ARCHIVE_SWAPS = {"one-island": {"islands": 1}, "no-grid": {"features": (), "cell_size": 50}}
RECIPE_SWAPS = {"no-inspirations": {"inspirations": 0}}


def cascade_evaluate(files):
    """Stage 1 runs evaluate.py with a 30 s limit: crashes and invalid packings fail cheaply.
    Only programs that ran out of time get stage 2, the same check with the task's 5 minute limit."""
    quick = ScriptEvaluator(files, "python evaluate.py", timeout="30s")
    full = ScriptEvaluator(files, "python evaluate.py", timeout="5m")
    timed_out = lambda evaluation: not evaluation.objectives and "exceeded" in evaluation.feedback  # noqa: E731
    return staged_eval(quick, full, promote=timed_out)


async def run(
    dir: str,
    variant: str = "full",
    models: Mapping[str, ek.operators.Model] | None = None,
    workers: int = 8,
    max_evaluations: int = 300,
    max_cost: float | None = 5.0,
) -> ek.Individual | None:
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}")
    task = packing_task()
    task = replace(task, evaluate=cascade_evaluate(task.files))
    models = models or {
        "fast": model(os.environ.get("EVOKIT_FAST_MODEL", "claude-haiku-4-5")),
        "strong": model(os.environ.get("EVOKIT_STRONG_MODEL", "claude-sonnet-4-5")),
    }
    if variant == "single-model":
        models = {"fast": models["fast"]}
    return await recipes.openevolve(
        task, models, weights={"fast": 0.8, "strong": 0.2}, dir=dir, archive=IslandArchive(**ARCHIVE_SWAPS.get(variant, {})),
        workers=workers, max_evaluations=max_evaluations, max_cost=max_cost, **RECIPE_SWAPS.get(variant, {}),
    )


def print_curve(dir: str, every: int = 25) -> None:
    best, points = None, []
    evaluated = ek.open_experiment(dir).evaluated()
    for k, individual in enumerate(evaluated, 1):
        score = individual.objective("score")
        if score is not None and (best is None or score > best):
            best = score
        if k % every == 0 or k == len(evaluated):
            points.append(f"{k}: {best:.4f}" if best is not None else f"{k}: -")
    print("best score by evaluation: " + ", ".join(points))


def main() -> None:
    args = arguments("AlphaEvolve / OpenEvolve on circle packing", variants=VARIANTS, workers=8, max_evaluations=300, max_cost=5.0)
    asyncio.run(run(args.dir, args.variant, workers=args.workers, max_evaluations=args.max_evaluations, max_cost=args.max_cost))
    report(args.dir)
    print_curve(args.dir)
    archive = ek.open_experiment(args.dir).archive
    for island in range(archive.islands):
        print(archive.table(island))


if __name__ == "__main__":
    main()
