"""EvoRLM on circle packing: evokit.recipes.experimental.evorlm (needs rlmflow).

One root agent runs the evolution: each generation is one `mutate` call whose ops are fresh agents
running the same policy over the parents they name, with evaluations carved from the root's.

    EVOKIT_HARNESS=claude_code python -m algorithms.evorlm --dir runs/evorlm-0 --generations 3
    EVOKIT_MODEL=gpt-5 python -m algorithms.evorlm --dir runs/evorlm-1

Claude Code (needs claude-agent-sdk) runs with its own login, as the AVO example does, unless
EVOKIT_AGENT_AUTH=env, and with hour-long shell timeouts; its token counts include cache reads,
hence the larger default budget.
"""

from __future__ import annotations

import asyncio
import os

from algorithms.common import arguments, packing_task, report
from rlmflow import AgentConfig

from evokit.recipes.experimental.evorlm import evorlm


def main() -> None:
    args = arguments("EvoRLM on circle packing", generations=3)
    harness = os.environ.get("EVOKIT_HARNESS", "repl")
    claude_code = harness == "claude_code"
    if claude_code:
        if os.environ.get("EVOKIT_AGENT_AUTH", "login") == "login":
            os.environ.pop("ANTHROPIC_API_KEY", None)
        for name in ("BASH_DEFAULT_TIMEOUT_MS", "BASH_MAX_TIMEOUT_MS"):  # a mutate cell can run for an hour
            os.environ.setdefault(name, "3600000")
    agent = AgentConfig(
        harness=harness,
        model=os.environ.get("EVOKIT_MODEL", "sonnet" if claude_code else "claude-sonnet-4-5"),
        max_iters=18,
    )
    max_tokens = int(os.environ.get("EVOKIT_MAX_TOKENS", 30_000_000 if claude_code else 3_000_000))
    asyncio.run(evorlm(packing_task(), agent, dir=args.dir, generations=args.generations, max_tokens=max_tokens))
    report(args.dir)


if __name__ == "__main__":
    main()
