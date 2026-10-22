"""autoresearch (Karpathy, 2026) on circle packing: evokit.recipes.autoresearch.

One lineage: keep a change only if the score went up, or held with less code; never stop.

The agent you are talking to is the operator (the original setup): the CLI builds the run from
the task folder's config.yaml, which has no operator.

    cd examples/circle_packing
    evokit status --task . --dir runs/autoresearch-external-0
    evokit skill export claude --dir runs/autoresearch-external-0
    # then in Claude Code, in this directory:
    #   "You are the evokit operator for runs/autoresearch-external-0. Follow the evokit skill. Never stop."

Unattended, a headless coding agent makes each change:

    python -m algorithms.autoresearch --dir runs/autoresearch-0 --max-evaluations 20
"""

from __future__ import annotations

import asyncio

from algorithms.common import arguments, claude_code, packing_task, report, seed_archive

from evokit import recipes
from evokit.archives import SimplerBestArchive
from evokit.files import with_lines
from evokit.storages import DirectoryStorage

SIMPLICITY = (
    "Keep it simple: a change that scores the same with fewer lines of code is also kept, "
    "and a tiny gain that adds a lot of complexity is not worth it."
)


async def run(directory: str, agent=None, max_evaluations: int | None = None):
    """A fresh headless coding-agent session per change."""
    storage = DirectoryStorage(directory)
    task = packing_task(storage)
    eval_fn = with_lines(task.eval_fn)
    archive = SimplerBestArchive("score", size="lines", storage=storage)
    await seed_archive(storage, archive, task.project.seed_genome(), eval_fn)
    await recipes.autoresearch(
        task.project, eval_fn, agent or claude_code(), storage=storage, archive=archive,
        instructions=f"{task.instructions}\n\n{SIMPLICITY}", objective="score",
        results=storage.root / "results.tsv", max_evaluations=max_evaluations,
    )
    return storage, archive


def main() -> None:
    args = arguments("autoresearch on circle packing")
    storage, archive = asyncio.run(run(args.dir, max_evaluations=args.max_evaluations))
    report(storage, archive)


if __name__ == "__main__":
    main()
