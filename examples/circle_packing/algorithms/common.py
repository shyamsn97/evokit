"""What the algorithm examples share: the circle-packing task folder, models, seeding and run flags."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import evokit as ek
from evokit.agents import ClaudeCode, CodingAgent, ReplAgent
from evokit.operators import AnthropicModel, OpenAIModel

CIRCLE_PACKING = Path(__file__).resolve().parents[1]

PRICES = {
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-sonnet-4-5": (3.0, 15.0),
    "claude-opus-4-1": (15.0, 75.0),
    "gpt-5": (1.25, 10.0),
    "gpt-5-mini": (0.25, 2.0),
}


def packing_task(storage: ek.Storage) -> ek.Task:
    """examples/circle_packing as a Task: main.py evolves, and evaluate.py (copied into the run
    directory's env/) scores it; an invalid packing exits non-zero, so it is never admitted."""
    return ek.load_task(CIRCLE_PACKING, environment=Path(storage.root) / "env")


def model(name: str | None = None) -> AnthropicModel:
    name = name or os.environ.get("EVOKIT_MODEL", "claude-sonnet-4-5")
    return AnthropicModel(name, price=PRICES.get(name))


def claude_code(max_turns: int = 40, timeout: str = "1h") -> ClaudeCode:
    """Claude Code with its own login (EVOKIT_AGENT_AUTH=env uses ANTHROPIC_API_KEY instead)."""
    return ClaudeCode(
        model=os.environ.get("EVOKIT_AGENT_MODEL", "sonnet"), max_turns=max_turns, timeout=timeout,
        auth=os.environ.get("EVOKIT_AGENT_AUTH", "login"),
    )


def coding_agent(spec: str, long: bool = False) -> CodingAgent:
    """`claude` (Claude Code), or `repl:MODEL` (ReplAgent on an OpenAI model, e.g. repl:gpt-5-mini).
    `long` is for one session that lasts a whole run (AVO)."""
    if spec == "claude":
        return claude_code(*((300, "3h") if long else (40, "1h")))
    if spec.startswith("repl:"):
        name = spec.removeprefix("repl:")
        return ReplAgent(OpenAIModel(name, price=PRICES.get(name)), max_steps=150 if long else 40, timeout="3h" if long else "1h")
    raise SystemExit(f"unknown agent {spec!r}: use claude or repl:MODEL")


async def seed_archive(history: ek.History, archive: ek.Archive, genome, eval_fn) -> None:
    """A fresh run starts from `genome`; a continued run's archive has restored itself."""
    if archive.members():
        return
    seed = ek.Individual.new(genome, name="seed")
    evaluation = await ek.as_eval_fn(eval_fn)(seed)
    await ek.tell(history, archive, [seed], [evaluation])
    if not archive.members():
        feedback = evaluation.feedback
        raise SystemExit(f"the seed failed its evaluation:\n{feedback[:2000]}")


def arguments(description: str, variants=(), **defaults) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--dir", required=True, help="run directory (continues the run if it exists)")
    if variants:
        parser.add_argument("--variant", choices=list(variants), default=variants[0])
    parser.add_argument("--workers", type=int, default=defaults.get("workers", 1))
    parser.add_argument("--max-evaluations", type=int, default=defaults.get("max_evaluations"))
    parser.add_argument("--max-cost", type=float, default=defaults.get("max_cost", 5.0), help="US dollars")
    parser.add_argument("--agent", default="claude", help="coding agent: claude, or repl:MODEL (e.g. repl:gpt-5-mini)")
    logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="evokit: %(message)s")
    return parser.parse_args()


def report(storage: ek.Storage, archive: ek.Archive) -> ek.Individual | None:
    """Print what the run did, and write the best main.py next to it."""
    history = ek.History(storage)
    counts, usage = history.counts(), history.usage()
    print(f"{counts['evaluated']} evaluations ({counts['added']} admitted), ${usage['cost_usd']:.2f}")
    best = archive.best()
    if best is not None:
        print(f"best: {best.id} score {best.objective('score')}")
        if "main.py" in best.genome:
            storage.write("best_main.py", best.genome.text("main.py").encode())
    return best
