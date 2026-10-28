"""self_modify: a coding agent that is the genome diagnoses its own evaluation logs and rewrites itself."""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path
from typing import Mapping

from evokit.agents.command import CommandAgent
from evokit.errors import OperatorError
from evokit.files.tree import Files
from evokit.records import Individual, merge_usage
from evokit.util import title_and_body, truncate

DIAGNOSE = """This directory is your own source code: a coding agent, run as \
`python {entrypoint} --dir WORKDIR --task TASK`. ./{logs} holds the logs of your last evaluation, where \
you scored {score}.
{instructions}
Read the logs and find what limited you as a coding agent: your tools, your prompt, how you plan, \
check your work, or use your steps. Do not try to solve the evaluation's tasks yourself. Write one \
concrete, general improvement to your own code into proposal.md: what to change, in which file, and \
why it will help. Do not change any other file."""

IMPLEMENT = """This directory is the source code of a coding agent. It is run as \
`python {entrypoint} --dir WORKDIR --task TASK`, and the last line it prints must stay a JSON object \
with "summary", "cost_usd" and "tokens". Implement this improvement to it:

{proposal}

Keep the agent working: afterwards, `python -c "import {module}"` must succeed."""


def script_agent(
    directory: str | Path,
    entrypoint: str = "agent.py",
    timeout: str | float | None = "15m",
    env: Mapping[str, str] | None = None,
) -> CommandAgent:
    """The agent whose code is in `directory`, run as `python ENTRYPOINT --dir . --task PROMPT`."""
    argv = [sys.executable, str(Path(directory) / entrypoint), "--dir", ".", "--task", "{prompt}"]
    return CommandAgent(argv, timeout, env)


async def self_modify(
    parent: Individual,
    experiment,
    *,
    fixed_agent: Files | None = None,
    instructions: str | None = None,
    entrypoint: str = "agent.py",
    logs: str = "logs",
    objective: str = "score",
    timeout: str | float | None = "15m",
) -> tuple[Individual, dict]:
    """The genome is a coding agent's repository; the parent agent writes the child.

    1. Diagnose: the parent agent reads its evaluation logs (`artifacts/<parent id>/logs` in the
       experiment, which the evaluate function writes) and writes proposal.md.
    2. Implement: the parent agent applies the proposal to a fresh copy of its own code.

    The agent runs as `python ENTRYPOINT --dir . --task PROMPT` (see script_agent). With
    `fixed_agent`, that agent does both steps for every parent instead, so the improver never
    gets better (the DGM ablation without self-improvement). `instructions` describe the
    evaluation, for the diagnosis (default: the task's).
    """
    root = Path(tempfile.mkdtemp(prefix="evokit-self-modify-"))
    try:
        (fixed_agent or parent.genome).write_to(root / "runner")
        agent = script_agent(root / "runner", entrypoint, timeout)

        parent.genome.write_to(root / "diagnose")
        with experiment.storage.local(f"artifacts/{parent.id}") as artifacts:
            if (artifacts / logs).is_dir():
                shutil.copytree(artifacts / logs, root / "diagnose" / logs)
        score = parent.objective(objective)
        if instructions is None:
            instructions = experiment.task.instructions if experiment.task else None
        prompt = DIAGNOSE.format(
            entrypoint=entrypoint,
            logs=logs,
            score=f"{score:.3f}" if isinstance(score, (int, float)) else "nothing",
            instructions=f"\n{instructions}\n" if instructions else "",
        )
        async with agent.session(root / "diagnose") as session:
            diagnosis = await session.prompt(prompt)
        usage = diagnosis.usage
        proposal_file = root / "diagnose" / "proposal.md"
        if diagnosis.error or not proposal_file.is_file():
            raise OperatorError(f"no proposal: {diagnosis.error or diagnosis.text}", usage)
        proposal = proposal_file.read_text()

        parent.genome.write_to(root / "implement")
        module = Path(entrypoint).stem
        async with agent.session(root / "implement") as session:
            implementation = await session.prompt(IMPLEMENT.format(entrypoint=entrypoint, proposal=proposal, module=module))
        usage = merge_usage(usage, implementation.usage)
        if implementation.error:
            raise OperatorError(f"implementing failed: {implementation.error}", usage)
        child = Files.from_directory(root / "implement", exclude=("proposal.md", f"{logs}/**"))
        if child == parent.genome:
            raise OperatorError("the agent did not change its code", usage)
    finally:
        shutil.rmtree(root, ignore_errors=True)
    name, description = title_and_body(implementation.text)
    info = {"operator": "self-modify", "proposal": truncate(proposal, 4000)}
    return Individual.new(child, [parent.id], info, name=name, description=truncate(description, 2000)), usage


__all__ = ["DIAGNOSE", "IMPLEMENT", "script_agent", "self_modify"]
