"""Darwin Goedel Machine (Zhang et al., 2025) on circle packing: evokit.recipes.dgm.

The genome is a coding agent (seed_agent/: an LLM loop with bash, view and edit tools, and no
packing knowledge). An agent is scored by letting it solve packing tasks (benchmark.py, the
domain-specific part), and the next agent is written by a parent agent that reads its own logs.
Every working agent stays in the archive, and parents are sampled by score, fewer children preferred.

    cd examples/circle_packing
    python -m algorithms.dgm.main --dir runs/dgm-0 --variant full

Variants, one swap each: full; no-self-improve (the seed agent always does the modifying);
no-open-ended (keep only the newest working agent, Greedy); greedy (Greedy over the full archive).
The agents call the Anthropic API ($ANTHROPIC_API_KEY; model in $DGM_MODEL) and run shell
commands, always in scratch copies.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from algorithms.common import arguments, seed_archive
from algorithms.dgm.benchmark import INSTRUCTIONS, PackingBenchmark

import evokit as ek
from evokit import recipes
from evokit.archives import LatestArchive, UnboundedArchive
from evokit.files import Files
from evokit.storages import DirectoryStorage

SEED_AGENT = Path(__file__).resolve().parent / "seed_agent"
VARIANTS = ("full", "no-self-improve", "no-open-ended", "greedy")


async def run(directory: str, variant: str = "full", workers: int = 4, max_children: int = 40, max_cost: float | None = 20.0):
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}")
    storage = DirectoryStorage(directory)
    history = ek.History(storage)
    seed_agent = Files.from_directory(SEED_AGENT)
    eval_fn = PackingBenchmark(storage)
    archive = LatestArchive("score", storage=storage) if variant == "no-open-ended" else UnboundedArchive("score", storage=storage)
    await seed_archive(history, archive, seed_agent, eval_fn)
    await recipes.dgm(
        eval_fn, history=history, archive=archive, instructions=INSTRUCTIONS, workers=workers,
        max_children=max_children, max_cost=max_cost,
        fixed_agent=seed_agent if variant == "no-self-improve" else None,
        open_ended=variant not in ("no-open-ended", "greedy"),
    )
    return storage, archive


def report(storage: ek.Storage, archive: ek.Archive) -> None:
    """Write the best agent, and the best packing any agent produced."""
    history = ek.History(storage)
    counts, usage = history.counts(), history.usage()
    print(f"{counts['evaluated']} agents evaluated ({counts['added']} admitted), ${usage['cost_usd']:.2f}")
    best = archive.best()
    if best is None:
        print("no working agent")
        return
    with storage.local("best_agent") as folder:
        best.genome.write_to(folder)
    print(f"best agent: {best.id} score {best.objective('score'):.3f}")
    solved = [i for i in history.evaluated() if i.objective("best_sum_26") is not None and storage.exists(f"artifacts/{i.id}/best_main.py")]
    if solved:
        top = max(solved, key=lambda i: i.objective("best_sum_26"))
        storage.write("best_main.py", storage.read(f"artifacts/{top.id}/best_main.py"))
        print(f"best packing (N=26): sum of radii {top.objective('best_sum_26'):.4f}, by agent {top.id}")


def main() -> None:
    args = arguments("DGM on circle packing", variants=VARIANTS, workers=4, max_evaluations=40, max_cost=20.0)
    storage, archive = asyncio.run(run(args.dir, args.variant, args.workers, args.max_evaluations, args.max_cost))
    report(storage, archive)


if __name__ == "__main__":
    main()
