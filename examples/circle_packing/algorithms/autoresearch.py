"""autoresearch (Karpathy, 2026) on circle packing: evokit.recipes.autoresearch.

One lineage: keep a change only if the score went up, or held with less code; never stop.

The agent you are talking to is the operator (the original setup):

    cd examples/circle_packing
    evokit status --task . --dir runs/autoresearch-external-0
    evokit skill export claude
    # then in Claude Code, in this directory:
    #   "You are the evokit operator for runs/autoresearch-external-0. Follow the evokit skill. Never stop."

Unattended, a headless coding agent makes each change:

    python -m algorithms.autoresearch --dir runs/autoresearch-0 --max-evaluations 20
"""

from __future__ import annotations

import asyncio
from dataclasses import replace

from algorithms.common import arguments, coding_agent, packing_task, report

from evokit import recipes

SIMPLICITY = (
    "Keep it simple: a change that scores the same with fewer lines of code is also kept, "
    "and a tiny gain that adds a lot of complexity is not worth it."
)


def main() -> None:
    args = arguments("autoresearch on circle packing")
    task = packing_task()
    task = replace(task, instructions=f"{task.instructions}\n\n{SIMPLICITY}")
    asyncio.run(recipes.autoresearch(task, coding_agent(args.agent), dir=args.dir, max_evaluations=args.max_evaluations, max_cost=args.max_cost))
    report(args.dir)


if __name__ == "__main__":
    main()
