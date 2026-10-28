"""Run an evokit recipe on one CO-Bench problem, then score the best solve.py on the held-out test set (heldout.py).

    cd examples/co_bench
    python setup.py                                                   # once
    python run.py bin-packing-one-dimensional --algorithm autoresearch --dir runs/bin-packing-autoresearch-0
    python run.py job-shop-scheduling --algorithm openevolve --model gpt-5-mini --dir runs/job-shop-openevolve-0
    python run.py aircraft-landing --algorithm avo --agent claude --max-cost 5 --dir runs/aircraft-avo-0
    python run.py job-shop-scheduling --algorithm evorlm --dir runs/job-shop-evorlm-0     # needs rlmflow

CO-Bench's protocol is 64 evaluations per problem with o3-mini (now retired by OpenAI), so 64 is the default
(--max-evaluations 64). Agents: `claude` (Claude Code, $EVOKIT_AGENT_MODEL, default sonnet) or
`repl:MODEL` (ReplAgent on an OpenAI model). --model is the OpenAI model for openevolve.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from dataclasses import replace
from pathlib import Path

import evokit as ek
from evokit import recipes
from evokit.agents import ClaudeCode, ReplAgent
from evokit.archives import BestArchive
from evokit.operators import OpenAIModel

from heldout import print_rows, test_run

HERE = Path(__file__).resolve().parent
ALGORITHMS = ("autoresearch", "avo", "evorlm", "openevolve")
PRICES = {
    "o3-mini": (1.1, 4.4),
    "o4-mini": (1.1, 4.4),
    "gpt-5": (1.25, 10.0),
    "gpt-5-mini": (0.25, 2.0),
    "gpt-4.1": (2.0, 8.0),
}


def openai_model(name: str) -> OpenAIModel:
    return OpenAIModel(name, price=PRICES.get(name))


def agent(spec: str, long: bool):
    if spec == "claude":
        turns, timeout = (300, "3h") if long else (40, "1h")
        return ClaudeCode(
            model=os.environ.get("EVOKIT_AGENT_MODEL", "sonnet"),
            max_turns=turns,
            timeout=timeout,
            auth="login",
        )
    if spec.startswith("repl:"):
        return ReplAgent(
            openai_model(spec.removeprefix("repl:")),
            max_steps=100 if long else 40,
            timeout="3h" if long else "1h",
        )
    raise SystemExit(f"unknown agent {spec!r}: use claude or repl:MODEL")


async def run(
    problem: str,
    algorithm: str,
    directory: str,
    agent_spec: str,
    model: str,
    max_evaluations: int,
    max_cost: float,
    workers: int,
    evaluate=None,
):
    folder = HERE / "tasks" / problem
    if not folder.is_dir():
        raise SystemExit(
            f"no task folder {folder.relative_to(HERE)}; run `python setup.py` first"
        )
    # heldout.py scores the best solve.py with the experiment's own copy of evaluate.py, in <dir>/env
    task = ek.Task.from_directory(folder, environment=Path(directory) / "env")
    if evaluate is not None:
        task = replace(task, evaluate=evaluate)
    limits = dict(max_evaluations=max_evaluations, max_cost=max_cost)
    if algorithm == "evorlm":
        from rlmflow import AgentConfig

        from evokit.recipes.experimental.evorlm import evorlm

        if agent_spec == "claude":
            os.environ.pop("ANTHROPIC_API_KEY", None)
            config = AgentConfig(harness="claude_code", model=os.environ.get("EVOKIT_AGENT_MODEL", "sonnet"), max_iters=18)
        else:
            config = AgentConfig(harness="repl", model=agent_spec.removeprefix("repl:"), max_iters=18)
        # the root holds generations * max_ops * eval_budget evaluations; the seed is the 64th
        await evorlm(task, config, dir=directory, generations=3, eval_budget=3,
                     max_ops=max(1, (max_evaluations - 1) // 9), max_cost=max_cost, max_tokens=30_000_000)
    elif algorithm == "autoresearch":
        archive = BestArchive()
        await recipes.autoresearch(task, agent(agent_spec, long=False), dir=directory, archive=archive, **limits)
    elif algorithm == "avo":
        await recipes.avo(task, agent(agent_spec, long=True), dir=directory, **limits)
    else:
        await recipes.openevolve(task, {"main": openai_model(model)}, dir=directory, workers=workers, **limits)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run an evokit recipe on a CO-Bench problem"
    )
    parser.add_argument(
        "problem", help="a folder in tasks/, e.g. bin-packing-one-dimensional"
    )
    parser.add_argument("--algorithm", choices=ALGORITHMS, default="autoresearch")
    parser.add_argument(
        "--dir", required=True, help="experiment directory (continues it if it exists)"
    )
    parser.add_argument(
        "--agent", default="claude", help="claude or repl:MODEL (autoresearch, avo, evorlm)"
    )
    parser.add_argument(
        "--model", default="gpt-5-mini", help="OpenAI model (openevolve)"
    )
    parser.add_argument("--max-evaluations", type=int, default=64)
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="model calls in flight (openevolve); evaluations still run one at a time",
    )
    parser.add_argument(
        "--max-cost", type=float, default=5.0, help="US dollars (avo, openevolve)"
    )
    args = parser.parse_args()
    logging.basicConfig(
        stream=sys.stderr, level=logging.INFO, format="evokit: %(message)s"
    )
    asyncio.run(
        run(
            args.problem,
            args.algorithm,
            args.dir,
            args.agent,
            args.model,
            args.max_evaluations,
            args.max_cost,
            args.workers,
        )
    )
    print_rows([test_run(args.dir)])


if __name__ == "__main__":
    main()
