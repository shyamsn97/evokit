from __future__ import annotations

import asyncio
import json
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from evokit.errors import EnvironmentModified, OperatorError, UsageError
from evokit.loop import EvalFn, call_eval_fn
from evokit.files.agents import CodingAgent
from evokit.files.evaluator import ScriptEvaluator
from evokit.files.project import Project
from evokit.files.tree import Files
from evokit.operator import Operator
from evokit.operators.prompts import describe_individual, join_notes
from evokit.records import Evaluation, Individual, Offspring, Selection, merge_usage
from evokit.util import truncate

AGENT_PROMPT = """You are the variation operator in an evolutionary search run by EvoKit.

Your working directory is a full copy of the project. Improve the program so that it scores better on the objective, then stop.

Editable files (the genome): {editable}
Only changes to these files are kept. Do not modify any other file; edits to them are rejected.

{individuals}
{others}{draft}{instructions}When you are done, stop and reply with a short summary of what you changed and why."""

class _DraftServer:
    """A localhost socket the agent calls (through a tiny script outside the workspace) to score drafts."""

    def __init__(self, project: Project, evaluate: EvalFn, workspace: Path, parents: list[str], limit: int):
        self.project, self.evaluate, self.workspace, self.parents, self.limit = project, evaluate, workspace, parents, limit
        self.count = 0
        self.usage: dict[str, float] = {}
        self._server: asyncio.AbstractServer | None = None
        self._dir = Path(tempfile.mkdtemp(prefix="evokit-draft-"))
        self.script = self._dir / "evokit-eval"

    async def __aenter__(self) -> "_DraftServer":
        self._server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        port = self._server.sockets[0].getsockname()[1]
        self.script.write_text(
            f"#!{sys.executable}\n"
            "import socket, sys\n"
            f"s = socket.create_connection(('127.0.0.1', {port}))\n"
            "s.sendall(b'eval\\n')\n"
            "data = b''.join(iter(lambda: s.recv(65536), b''))\n"
            "print(data.decode())\n"
            "sys.exit(0 if b'\"error\"' not in data else 1)\n"
        )
        self.script.chmod(0o755)
        return self

    async def __aexit__(self, *exc) -> None:
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        shutil.rmtree(self._dir, ignore_errors=True)

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            await reader.readline()
            if self.count >= self.limit:
                reply = {"error": f"draft evaluation limit reached ({self.limit}); finish and summarize"}
            else:
                try:
                    genome = await asyncio.to_thread(self.project.read, self.workspace)
                except EnvironmentModified as exc:
                    reply = {"error": exc.message, "hint": exc.hint}
                else:
                    self.count += 1
                    evaluation = await self._score(Individual.new(genome, self.parents, {"operator": "draft"}))
                    self.usage = merge_usage(self.usage, evaluation.usage)
                    reply = {**evaluation.to_dict(), "draft_evals_left": self.limit - self.count}
            writer.write(json.dumps(reply).encode())
            await writer.drain()
        finally:
            writer.close()

    async def _score(self, draft: Individual) -> Evaluation:
        workdir = Path(tempfile.mkdtemp(prefix="evokit-draft-work-"))
        try:
            return await call_eval_fn(self.evaluate, draft, workdir)
        except Exception:
            return Evaluation(feedback=truncate(traceback.format_exc(), 4000))
        finally:
            shutil.rmtree(workdir, ignore_errors=True)


def _clear(directory: Path) -> None:
    for child in directory.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()


class CodingAgentOperator(Operator):
    """A coding agent edits a copy of the parent's project in place.

    With `draft_evaluate` (an eval function, or a command run like ScriptEvaluator), the agent
    can score drafts through the command in $EVOKIT_DRAFT_EVAL, up to `max_draft_evals` times;
    their cost is part of the child's creation cost. With persistent=True the agent resumes one
    conversation across children in one workspace (AVO's single continuing agent); such an
    operator serves one worker at a time, and `session` resumes an earlier conversation.
    `redirect(text)` adds `text` to the next prompt (AVO's supervisor).
    """

    name = "coding-agent"

    def __init__(
        self,
        agent: CodingAgent,
        project: Project,
        draft_evaluate: EvalFn | str | Sequence[str] | None = None,
        max_draft_evals: int = 10,
        persistent: bool = False,
        instructions: str | None = None,
        prompt: Callable[..., str] | None = None,
        session: str | None = None,
    ):
        self.agent = agent
        self.project = project
        if isinstance(draft_evaluate, (str, list, tuple)):
            draft_evaluate = ScriptEvaluator(project, draft_evaluate)
        self.draft_evaluate = draft_evaluate
        self.max_draft_evals = max_draft_evals
        self.persistent = persistent
        self.instructions = instructions
        self.prompt = prompt
        self.session = session
        self.pending_redirect: str | None = None
        self._running = 0

    def redirect(self, text: str) -> None:
        """Add `text` to the next prompt, e.g. to steer a persistent session in a new direction."""
        self.pending_redirect = text

    def _prompt(self, individuals, draft_cmd: str | None, other_dirs: Mapping[str, Path], redirect: str | None) -> str:
        if self.prompt:
            return self.prompt(individuals=individuals, draft_command=draft_cmd, other_dirs=other_dirs, redirect=redirect)
        described = "\n\n".join(describe_individual(i) for i in individuals[:1]) or "There is no current program: write it from scratch."
        others = "".join(f"Individual {pid} (for inspiration or recombination) is at {d}\n\n" for pid, d in other_dirs.items())
        draft = (
            f"To score your current edits, run: {draft_cmd}\n"
            f"It prints the evaluation as JSON. You have {self.max_draft_evals} draft evaluations; "
            "the final score is computed separately after you finish.\n\n"
            if draft_cmd
            else ""
        )
        notes = join_notes(self.instructions, redirect)
        return AGENT_PROMPT.format(
            editable=", ".join(self.project.editable),
            individuals=described + "\n",
            others=others,
            draft=draft,
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
            return await self._create((*selection.parents, *selection.inspirations), len(selection.parents))
        finally:
            self._running -= 1

    async def _create(self, individuals: tuple[Individual, ...], n_parents: int) -> Offspring:
        base = individuals[0].genome if individuals else Files()
        redirect, self.pending_redirect = self.pending_redirect, None
        started = time.monotonic()
        if self.persistent:
            workspace, owned = self.project.scratch / "agent", False
            workspace.mkdir(parents=True, exist_ok=True)
            _clear(workspace)
        else:
            workspace, owned = Path(tempfile.mkdtemp(prefix="evokit-ws-")), True
        others_root = Path(tempfile.mkdtemp(prefix="evokit-inputs-")) if len(individuals) > 1 else None
        try:
            await asyncio.to_thread(self.project.write, base, workspace)
            other_dirs = {}
            for other in individuals[1:]:
                other.genome.write_to(others_root / other.id)
                other_dirs[other.id] = others_root / other.id
            return await self._run_agent(individuals, n_parents, base, workspace, other_dirs, redirect, started)
        finally:
            if owned:
                shutil.rmtree(workspace, ignore_errors=True)
            if others_root:
                shutil.rmtree(others_root, ignore_errors=True)

    async def _run_agent(self, individuals, n_parents, base, workspace, other_dirs, redirect, started) -> Offspring:
        parents = [i.id for i in individuals[:n_parents]]
        draft_server = (
            _DraftServer(self.project, self.draft_evaluate, workspace, parents, self.max_draft_evals)
            if self.draft_evaluate and self.max_draft_evals > 0
            else None
        )
        if draft_server:
            await draft_server.__aenter__()
        try:
            draft_cmd = str(draft_server.script) if draft_server else None
            prompt = self._prompt(individuals, draft_cmd, other_dirs, redirect)
            env = {"EVOKIT_DRAFT_EVAL": draft_cmd} if draft_cmd else {}
            result = await self.agent.run(workspace, prompt, session=self.session if self.persistent else None, env=env)
        finally:
            draft_evals = draft_server.count if draft_server else 0
            draft_usage = draft_server.usage if draft_server else {}
            if draft_server:
                await draft_server.__aexit__(None, None, None)
        usage = merge_usage(result.usage, draft_usage, {"draft_evals": draft_evals, "seconds": time.monotonic() - started})
        if self.persistent and result.session:
            self.session = result.session
        if result.error:
            raise OperatorError(f"agent failed: {result.error}\n\n{truncate(result.trace or '', 4000)}", usage)
        try:
            genome = await asyncio.to_thread(self.project.read, workspace)
        except EnvironmentModified as exc:
            raise OperatorError(f"{exc.message}. {exc.hint}", usage) from exc
        if individuals and genome == base:
            raise OperatorError(f"agent made no changes to the genome: {truncate(result.summary, 2000)}", usage)
        info: dict[str, Any] = {
            "operator": self.name,
            "summary": truncate(result.summary, 4000),
            "session": result.session,
        }
        if redirect:
            info["redirect"] = redirect
        return Offspring((Individual.new(genome, parents, info),), usage)
