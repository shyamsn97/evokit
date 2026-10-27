"""AVO (Agentic Variation Operators, NVIDIA 2026) on circle packing: evokit.recipes.avo.

One coding agent works in a single conversation for the whole run: it reads earlier versions
with `evokit log/show/diff/grep`, scores drafts with the real evaluation as often as it likes,
and commits when a version is ready; a commit that does not regress is kept, and the agent hears
so at once. After five drafts and commits without a strictly better score, a supervisor reviews the run
and sends it new directions.

    cd examples/circle_packing
    python -m algorithms.avo --dir runs/avo-0 --max-cost 5
"""

from __future__ import annotations

import asyncio

from algorithms.common import arguments, claude_code, coding_agent, packing_task, report, seed_archive

import evokit as ek
from evokit import recipes
from evokit.archives import BestArchive
from evokit.storages import DirectoryStorage


async def run(directory: str, agent=None, max_cost: float = 5.0, max_evaluations: int | None = None, **options):
    storage = DirectoryStorage(directory)
    history = ek.History(storage)
    task = packing_task(storage)
    archive = BestArchive("score", ties="accept", storage=storage)
    await seed_archive(history, archive, task.project.seed_genome(), task.eval_fn)
    await recipes.avo(
        task.project, task.eval_fn, agent or claude_code(max_turns=300, timeout="3h"), history=history, archive=archive, instructions=task.instructions,
        max_cost=max_cost, max_evaluations=max_evaluations, **options,
    )
    return storage, archive


def main() -> None:
    args = arguments("AVO on circle packing", max_cost=5.0)
    agent = coding_agent(args.agent, long=True)
    storage, archive = asyncio.run(run(args.dir, agent, max_cost=args.max_cost, max_evaluations=args.max_evaluations))
    report(storage, archive)


if __name__ == "__main__":
    main()
