from __future__ import annotations

import asyncio
import logging
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from evokit.agents.base import CodingAgent
from evokit.agents.tools import Tool, agent_tools, limited, tool_name
from evokit.errors import EnvironmentModified, OperatorError, UsageError
from evokit.files.evaluator import ScriptEvaluator
from evokit.files.project import Project
from evokit.files.tree import Files
from evokit.history import History
from evokit.loop import EvalFn
from evokit.operator import Operator
from evokit.operators.prompts import describe_individual, join_notes
from evokit.records import Individual, Offspring, Selection
from evokit.util import title_and_body, truncate

log = logging.getLogger("evokit")

AGENT_PROMPT = """You are the variation operator in an evolutionary search run by EvoKit.

Your working directory is a full copy of the project. Improve the program so that it scores better on the objective, then stop.

Editable files (the genome): {editable}
Only changes to these files are kept. Do not modify any other file; edits to them are rejected.

{individuals}
{others}{instructions}When you are done, stop and reply with a one-line title for this version, a blank line, then what you changed and why."""

READING = (
    "Every earlier version of the program, kept or rejected, with its score, can be read with the "
    "log, show, diff and grep tools. Check what was already tried before repeating an idea.\n\n"
)

SCORING = (
    "The score tool scores your current edits. {limit}The final score is computed separately after "
    "you finish, from the files as you leave them: before you stop, score your final version, and if "
    "your last change made it worse, put back the best version you measured.\n\n"
)


def clear_directory(directory: Path) -> None:
    for child in directory.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()


class CodingAgentOperator(Operator):
    """A coding agent edits a copy of the parent's project in place.

    The agent gets `tools(archive, workspace, parents)`: by default `agent_tools` (the population;
    with a `history`, the run's log, show, diff and grep; with an `eval_fn`, `score`, up to
    `max_draft_evals` times, None: no limit) plus the extra `tools` given here. Subclass and
    override `tools` to change them. `eval_fn` may be a command, run like ScriptEvaluator.

    With persistent=True the agent resumes one conversation across children in one workspace;
    such an operator serves one worker at a time, and `session` resumes an earlier conversation.
    `redirect(text)` adds `text` to the next prompt.
    """

    name = "coding-agent"

    def __init__(
        self,
        agent: CodingAgent,
        project: Project,
        eval_fn: EvalFn | str | Sequence[str] | None = None,
        history: History | None = None,
        max_draft_evals: int | None = 10,
        persistent: bool = False,
        instructions: str | None = None,
        prompt: Callable[..., str] | None = None,
        session: str | None = None,
        tools: Sequence[Tool] = (),
    ):
        self.agent = agent
        self.project = project
        if isinstance(eval_fn, (str, list, tuple)):
            eval_fn = ScriptEvaluator(project, eval_fn)
        self.eval_fn = eval_fn if max_draft_evals != 0 else None
        self.history = history
        self.max_draft_evals = max_draft_evals
        self.persistent = persistent
        self.instructions = instructions
        self.prompt = prompt
        self.session = session
        self.extra_tools = list(tools)
        self.pending_redirect: str | None = None
        self._running = 0

    async def check(self) -> None:
        await self.agent.check()

    def redirect(self, text: str) -> None:
        """Add `text` to the next prompt, e.g. to steer a persistent session in a new direction."""
        self.pending_redirect = text

    def tools(self, archive, workspace: Path, parents: list[str]) -> list[Tool]:
        """The agent's tools while it makes a child of `parents` in `workspace`."""
        tools = agent_tools(archive, self.history, self.eval_fn, project=self.project, workspace=workspace, parents=lambda: parents)
        if self.max_draft_evals is not None:
            tools = [limited(t, self.max_draft_evals) if tool_name(t) == "score" else t for t in tools]
        return [*tools, *self.extra_tools]

    def _prompt(self, individuals, tools: Sequence[Tool], other_dirs: Mapping[str, Path], redirect: str | None) -> str:
        if self.prompt:
            return self.prompt(individuals=individuals, tools=tools, other_dirs=other_dirs, redirect=redirect)
        names = {tool_name(t) for t in tools}
        described = "\n\n".join(describe_individual(i) for i in individuals[:1]) or "There is no current program: write it from scratch."
        others = "".join(f"Individual {pid} (for inspiration or recombination) is at {d}\n\n" for pid, d in other_dirs.items())
        if "log" in names:
            others += READING
        if "score" in names:
            limit = f"You can use it {self.max_draft_evals} times. " if self.max_draft_evals is not None else ""
            others += SCORING.format(limit=limit)
        notes = join_notes(self.instructions, redirect)
        return AGENT_PROMPT.format(
            editable=", ".join(self.project.editable),
            individuals=described + "\n",
            others=others,
            instructions=(notes + "\n\n") if notes else "",
        )

    async def create(self, selection: Selection, archive) -> Offspring:
        if self.persistent and self._running:
            raise UsageError(
                "a persistent agent session cannot serve two workers at once",
                hint="run one worker, or use persistent=False",
            )
        self._running += 1
        try:
            return await self._create(archive, (*selection.parents, *selection.inspirations), len(selection.parents))
        finally:
            self._running -= 1

    async def _create(self, archive, individuals: tuple[Individual, ...], n_parents: int) -> Offspring:
        base = individuals[0].genome if individuals else Files()
        redirect, self.pending_redirect = self.pending_redirect, None
        started = time.monotonic()
        if self.persistent:
            workspace, owned = self.project.scratch / "agent", False
            workspace.mkdir(parents=True, exist_ok=True)
            clear_directory(workspace)
        else:
            workspace, owned = Path(tempfile.mkdtemp(prefix="evokit-ws-")), True
        others_root = Path(tempfile.mkdtemp(prefix="evokit-inputs-")) if len(individuals) > 1 else None
        try:
            await asyncio.to_thread(self.project.write, base, workspace)
            other_dirs = {}
            for other in individuals[1:]:
                other.genome.write_to(others_root / other.id)
                other_dirs[other.id] = others_root / other.id
            parents = [i.id for i in individuals[:n_parents]]
            tools = self.tools(archive, workspace, parents)
            prompt = self._prompt(individuals, tools, other_dirs, redirect)
            session = self.session if self.persistent else None
            result = await self.agent.run(workspace, prompt, session=session, tools=tools)
            return await self._child(result, workspace, base, parents, individuals, redirect, started)
        finally:
            if owned:
                shutil.rmtree(workspace, ignore_errors=True)
            if others_root:
                shutil.rmtree(others_root, ignore_errors=True)

    async def _child(self, result, workspace, base, parents, individuals, redirect, started) -> Offspring:
        usage = {**result.usage, "seconds": time.monotonic() - started}
        if self.persistent and result.session:
            self.session = result.session
        logs = {"agent.jsonl": result.transcript.encode()} if result.transcript else {}
        failed = f"agent failed: {result.error}\n\n{truncate(result.trace or '', 4000)}" if result.error else None
        try:
            genome = await asyncio.to_thread(self.project.read, workspace)
        except EnvironmentModified as exc:
            raise OperatorError(failed or f"{exc.message}. {exc.hint}", usage, logs) from exc
        # An agent that fails after editing (turn limit, timeout, a crash) still leaves work worth scoring.
        if failed and (not individuals or genome == base):
            raise OperatorError(failed, usage, logs)
        if individuals and genome == base:
            raise OperatorError(f"agent made no changes to the genome: {truncate(result.summary, 2000)}", usage, logs)
        info: dict[str, Any] = {"operator": self.name, "session": result.session}
        if result.error:
            info["agent_error"] = truncate(result.error, 4000)
        if redirect:
            info["redirect"] = redirect
        name, description = title_and_body(result.summary)
        child = Individual.new(genome, parents, info, name=name, description=truncate(description, 4000))
        return Offspring((child,), usage, logs)
