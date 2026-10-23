"""SelfModify: a coding agent that is the genome diagnoses its own evaluation logs and rewrites itself."""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path
from typing import Mapping

from evokit.errors import OperatorError
from evokit.files.agents import CommandAgent
from evokit.files.tree import Files
from evokit.operator import Operator
from evokit.records import Individual, Offspring, Selection, merge_usage
from evokit.storage import Storage
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


class SelfModify(Operator):
    """The genome is a coding agent's repository; the parent agent writes the child.

    1. Diagnose: the parent agent reads its evaluation logs (`artifacts/<parent id>/logs` in the
       storage, which the eval function writes) and writes proposal.md.
    2. Implement: the parent agent applies the proposal to a fresh copy of its own code.

    The agent runs as `python ENTRYPOINT --dir . --task PROMPT` (see script_agent). With
    `fixed_agent`, that agent does both steps for every parent instead, so the improver never
    gets better (the DGM ablation without self-improvement). `instructions` describe the
    evaluation, for the diagnosis.
    """

    name = "self-modify"

    def __init__(
        self,
        storage: Storage,
        fixed_agent: Files | None = None,
        instructions: str | None = None,
        entrypoint: str = "agent.py",
        logs: str = "logs",
        objective: str = "score",
        timeout: str | float | None = "15m",
    ):
        self.storage, self.fixed_agent, self.instructions = storage, fixed_agent, instructions
        self.entrypoint, self.logs, self.objective, self.timeout = entrypoint, logs, objective, timeout

    async def create(self, selection: Selection, archive) -> Offspring:
        if not selection.parents:
            raise OperatorError("SelfModify needs a parent agent")
        parent = selection.parents[0]
        root = Path(tempfile.mkdtemp(prefix="evokit-self-modify-"))
        try:
            (self.fixed_agent or parent.genome).write_to(root / "runner")
            agent = script_agent(root / "runner", self.entrypoint, self.timeout)

            parent.genome.write_to(root / "diagnose")
            with self.storage.local(f"artifacts/{parent.id}") as artifacts:
                if (artifacts / self.logs).is_dir():
                    shutil.copytree(artifacts / self.logs, root / "diagnose" / self.logs)
            score = parent.objective(self.objective)
            prompt = DIAGNOSE.format(
                entrypoint=self.entrypoint,
                logs=self.logs,
                score=f"{score:.3f}" if isinstance(score, (int, float)) else "nothing",
                instructions=f"\n{self.instructions}\n" if self.instructions else "",
            )
            diagnosis = await agent.run(root / "diagnose", prompt)
            usage = diagnosis.usage
            proposal_file = root / "diagnose" / "proposal.md"
            if diagnosis.error or not proposal_file.is_file():
                raise OperatorError(f"no proposal: {diagnosis.error or diagnosis.summary}", usage)
            proposal = proposal_file.read_text()

            parent.genome.write_to(root / "implement")
            module = Path(self.entrypoint).stem
            implementation = await agent.run(
                root / "implement", IMPLEMENT.format(entrypoint=self.entrypoint, proposal=proposal, module=module)
            )
            usage = merge_usage(usage, implementation.usage)
            if implementation.error:
                raise OperatorError(f"implementing failed: {implementation.error}", usage)
            child = Files.from_directory(root / "implement", exclude=("proposal.md", f"{self.logs}/**"))
            if child == parent.genome:
                raise OperatorError("the agent did not change its code", usage)
        finally:
            shutil.rmtree(root, ignore_errors=True)
        name, description = title_and_body(implementation.summary)
        info = {"operator": self.name, "proposal": truncate(proposal, 4000)}
        return Offspring((Individual.new(child, [parent.id], info, name=name, description=truncate(description, 2000)),), usage)
