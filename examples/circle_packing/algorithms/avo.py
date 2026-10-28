"""AVO (Agentic Variation Operators, NVIDIA 2026) on circle packing: evokit.recipes.avo.

One coding agent works in a single conversation for the whole experiment: it reads earlier
versions with `evokit log/show/diff/grep`, scores drafts with the real evaluation as often as it
likes, and submits when a version is ready; a submission that does not regress is kept, and the
agent hears so at once. After five drafts and submissions without a strictly better score, a
supervisor reviews the experiment and sends it new directions.

    cd examples/circle_packing
    python -m algorithms.avo --dir runs/avo-0 --max-cost 5
"""

from __future__ import annotations

import asyncio

from algorithms.common import arguments, coding_agent, packing_task, report

from evokit import recipes


def main() -> None:
    args = arguments("AVO on circle packing", max_cost=5.0)
    agent = coding_agent(args.agent, long=True)
    asyncio.run(recipes.avo(packing_task(), agent, dir=args.dir, max_cost=args.max_cost, max_evaluations=args.max_evaluations))
    report(args.dir)


if __name__ == "__main__":
    main()
