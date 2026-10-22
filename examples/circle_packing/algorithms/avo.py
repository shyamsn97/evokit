"""AVO (Agentic Variation Operators, NVIDIA 2026) on circle packing: evokit.recipes.avo.

One coding agent keeps a single conversation across the whole run, scores drafts with the real
evaluation, and commits; a commit that does not regress is kept. After five rejections in a
row, it is told to step back and try something different.

    cd examples/circle_packing
    python -m algorithms.avo --dir runs/avo-0 --max-cost 5
"""

from __future__ import annotations

import asyncio

from algorithms.common import arguments, claude_code, packing_task, report, seed_archive

from evokit import recipes
from evokit.archives import BestArchive
from evokit.storages import DirectoryStorage


async def run(directory: str, agent=None, max_cost: float = 5.0, max_evaluations: int | None = None):
    storage = DirectoryStorage(directory)
    task = packing_task(storage)
    archive = BestArchive("score", ties="accept", storage=storage)
    await seed_archive(storage, archive, task.project.seed_genome(), task.eval_fn)
    await recipes.avo(
        task.project, task.eval_fn, agent or claude_code(), storage=storage, archive=archive, instructions=task.instructions,
        max_cost=max_cost, max_evaluations=max_evaluations,
    )
    return storage, archive


def main() -> None:
    args = arguments("AVO on circle packing", max_cost=5.0)
    storage, archive = asyncio.run(run(args.dir, max_cost=args.max_cost, max_evaluations=args.max_evaluations))
    report(storage, archive)


if __name__ == "__main__":
    main()
