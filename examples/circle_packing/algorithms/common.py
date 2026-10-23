"""What the algorithm examples share: the circle-packing task folder, models, seeding and run flags."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import evokit as ek
from evokit import history
from evokit.files import ClaudeCode
from evokit.operators import AnthropicModel

CIRCLE_PACKING = Path(__file__).resolve().parents[1]

PRICES = {"claude-haiku-4-5": (1.0, 5.0), "claude-sonnet-4-5": (3.0, 15.0), "claude-opus-4-1": (15.0, 75.0)}


def packing_task(storage: ek.Storage) -> ek.Task:
    """examples/circle_packing as a Task: main.py evolves, and evaluate.py (copied into the run
    directory's env/) scores it; an invalid packing exits non-zero, so it is never admitted."""
    return ek.load_task(CIRCLE_PACKING, environment=Path(storage.root) / "env")


def model(name: str | None = None) -> AnthropicModel:
    name = name or os.environ.get("EVOKIT_MODEL", "claude-sonnet-4-5")
    return AnthropicModel(name, price=PRICES.get(name))


def claude_code(max_turns: int = 40, timeout: str = "1h") -> ClaudeCode:
    return ClaudeCode(model=os.environ.get("EVOKIT_AGENT_MODEL", "sonnet"), max_turns=max_turns, timeout=timeout)


async def seed_archive(storage: ek.Storage, archive: ek.Archive, genome, eval_fn) -> None:
    """A fresh run starts from `genome`; a continued run's archive has restored itself."""
    if archive.members():
        return
    await ek.tell(storage, archive, await ek.evaluate(storage, [ek.Individual.new(genome, name="seed")], eval_fn=eval_fn))
    if not archive.members():
        feedback = history.evaluated(storage)[-1].evaluation.feedback
        raise SystemExit(f"the seed failed its evaluation:\n{feedback[:2000]}")


def arguments(description: str, variants=(), **defaults) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--dir", required=True, help="run directory (continues the run if it exists)")
    if variants:
        parser.add_argument("--variant", choices=list(variants), default=variants[0])
    parser.add_argument("--workers", type=int, default=defaults.get("workers", 1))
    parser.add_argument("--max-evaluations", type=int, default=defaults.get("max_evaluations"))
    parser.add_argument("--max-cost", type=float, default=defaults.get("max_cost", 5.0), help="US dollars")
    logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="evokit: %(message)s")
    return parser.parse_args()


def report(storage: ek.Storage, archive: ek.Archive) -> ek.Individual | None:
    """Print what the run did, and write the best main.py next to it."""
    counts, usage = history.counts(storage), history.usage(storage)
    print(f"{counts['evaluated']} evaluations ({counts['added']} admitted), ${usage['cost_usd']:.2f}")
    best = archive.best()
    if best is not None:
        print(f"best: {best.id} score {best.objective('score')}")
        if "main.py" in best.genome:
            storage.write("best_main.py", best.genome.text("main.py").encode())
    return best
