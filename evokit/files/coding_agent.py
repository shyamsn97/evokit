from __future__ import annotations

import asyncio
import functools
import json
import logging
import os
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping, Sequence

from evokit.errors import EnvironmentModified, OperatorError, UsageError
from evokit.loop import EvalFn, call_eval_fn
from evokit.files.agents import CodingAgent
from evokit.files.evaluator import ScriptEvaluator
from evokit.files.project import Project
from evokit.files.tree import Files
from evokit.interface.skill import install_skill
from evokit.operator import Operator
from evokit.operators.prompts import describe_individual, join_notes
from evokit.records import Evaluation, Individual, Offspring, Selection, merge_usage
from evokit.storages.directory import DirectoryStorage
from evokit.util import title_and_body, truncate

log = logging.getLogger("evokit")

AGENT_PROMPT = """You are the variation operator in an evolutionary search run by EvoKit.

Your working directory is a full copy of the project. Improve the program so that it scores better on the objective, then stop.

Editable files (the genome): {editable}
Only changes to these files are kept. Do not modify any other file; edits to them are rejected.

{individuals}
{others}{draft}{instructions}When you are done, stop and reply with a one-line title for this version, a blank line, then what you changed and why."""

class AgentCommands:
    """A localhost socket the agent calls through small scripts outside its workspace, one per
    command: each script sends its arguments, prints the JSON reply, and exits 1 on an error."""

    def __init__(self, handlers: Mapping[str, Callable[[list[str]], Awaitable[dict]]]):
        self.handlers = dict(handlers)
        self._server: asyncio.AbstractServer | None = None
        self._dir = Path(tempfile.mkdtemp(prefix="evokit-commands-"))
        self.scripts = {name: self._dir / name for name in self.handlers}

    async def __aenter__(self) -> "AgentCommands":
        self._server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        port = self._server.sockets[0].getsockname()[1]
        for name, script in self.scripts.items():
            script.write_text(
                f"#!{sys.executable}\n"
                "import json, socket, sys\n"
                f"s = socket.create_connection(('127.0.0.1', {port}))\n"
                f"s.sendall((json.dumps({{'command': {name!r}, 'args': sys.argv[1:]}}) + '\\n').encode())\n"
                "data = b''.join(iter(lambda: s.recv(65536), b''))\n"
                "print(data.decode())\n"
                "sys.exit(0 if b'\"error\"' not in data else 1)\n"
            )
            script.chmod(0o755)
        return self

    async def __aexit__(self, *exc) -> None:
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        shutil.rmtree(self._dir, ignore_errors=True)

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            request = json.loads(await reader.readline() or b"{}")
            handler = self.handlers.get(request.get("command"))
            try:
                reply = await handler(list(request.get("args") or [])) if handler else {"error": "unknown command"}
            except Exception as exc:
                log.exception("agent command %s failed", request.get("command"))
                reply = {"error": f"{type(exc).__name__}: {exc}"}
            writer.write(json.dumps(reply, default=str).encode())
            await writer.drain()
        finally:
            writer.close()


class Drafts:
    """The draft command: score the workspace's genome with the real evaluation, recording nothing.

    `limit` caps the calls (None: no cap); `stop()` returning a reason (e.g. the run's limits)
    refuses further drafts.
    """

    def __init__(
        self,
        project: Project,
        evaluate: EvalFn,
        workspace: Path,
        parents: list[str],
        limit: int | None = None,
        stop: Callable[[], str | None] | None = None,
    ):
        self.project, self.evaluate, self.workspace, self.parents = project, evaluate, workspace, parents
        self.limit, self.stop = limit, stop
        self.count = 0
        self.usage: dict[str, float] = {}
        self.last: Individual | None = None  # the last draft scored, with its evaluation

    async def __call__(self, args: list[str]) -> dict:
        reason = self.stop() if self.stop else None
        if reason:
            return {"error": f"the run is over ({reason}); stop now", "stop": reason}
        if self.limit is not None and self.count >= self.limit:
            return {"error": f"draft evaluation limit reached ({self.limit}); finish and summarize"}
        try:
            genome = await asyncio.to_thread(self.project.read, self.workspace)
        except EnvironmentModified as exc:
            return {"error": exc.message, "hint": exc.hint}
        self.count += 1
        draft = Individual.new(genome, self.parents, {"operator": "draft"})
        evaluation = await score_draft(self.evaluate, draft)
        self.last = draft.with_evaluation(evaluation)
        self.usage = merge_usage(self.usage, evaluation.usage)
        scores = ", ".join(f"{k}={v:.6g}" for k, v in evaluation.objectives.items()) or "failed"
        log.info("draft %d%s: %s", self.count, f"/{self.limit}" if self.limit is not None else "", scores)
        reply = evaluation.to_dict()
        if self.limit is not None:
            reply["draft_evals_left"] = self.limit - self.count
        return reply


async def score_draft(evaluate: EvalFn, draft: Individual) -> Evaluation:
    """Evaluate in a throwaway folder; a crash is a failed evaluation."""
    workdir = Path(tempfile.mkdtemp(prefix="evokit-draft-work-"))
    try:
        return await call_eval_fn(evaluate, draft, workdir)
    except Exception:
        return Evaluation(feedback=truncate(traceback.format_exc(), 4000))
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def run_access(agent: CodingAgent, workspace: Path, run_dir: Path | None) -> dict[str, str]:
    """For a run on local disk: the evokit skill in the workspace, and the environment for the read
    commands (log, show, diff, grep) to find the run and the `evokit` script."""
    if run_dir is None:
        return {}
    if agent.skill_dir:
        install_skill(workspace, agent.skill_dir)
    return {"EVOKIT_DIR": str(run_dir), "PATH": os.pathsep.join([str(_evokit_bin()), os.environ.get("PATH", "")])}


@functools.cache
def _evokit_bin() -> Path:
    """A directory holding only an `evokit` script for this interpreter, so the agent's `python`
    stays the one the evaluator gets."""
    directory = Path(tempfile.mkdtemp(prefix="evokit-bin-"))
    script = directory / "evokit"
    script.write_text(f"#!{sys.executable}\nfrom evokit.interface.cli import main\nmain()\n")
    script.chmod(0o755)
    return directory


def run_directory(storage: Any) -> Path | None:
    return storage.root if isinstance(storage, DirectoryStorage) else None


def clear_directory(directory: Path) -> None:
    for child in directory.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()


class CodingAgentOperator(Operator):
    """A coding agent edits a copy of the parent's project in place.

    With `draft_evaluate` (an eval function, or a command run like ScriptEvaluator), the agent
    can score drafts through the command in $EVOKIT_DRAFT_EVAL, up to `max_draft_evals` times
    (None: no limit); their cost is part of the child's creation cost. With persistent=True the agent resumes one
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
        max_draft_evals: int | None = 10,
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

    async def check(self) -> None:
        await self.agent.check()

    def redirect(self, text: str) -> None:
        """Add `text` to the next prompt, e.g. to steer a persistent session in a new direction."""
        self.pending_redirect = text

    def _prompt(
        self, individuals, draft_cmd: str | None, other_dirs: Mapping[str, Path], redirect: str | None, readable: bool = False
    ) -> str:
        if self.prompt:
            return self.prompt(individuals=individuals, draft_command=draft_cmd, other_dirs=other_dirs, redirect=redirect)
        history = (
            "Every earlier version of the program, kept or rejected, with its score, can be read with "
            "`evokit log`, `evokit show ID [PATH]`, `evokit diff A [B]` and `evokit grep PATTERN` "
            "(see the evokit skill). Check what was already tried before repeating an idea.\n\n"
            if readable
            else ""
        )
        described = "\n\n".join(describe_individual(i) for i in individuals[:1]) or "There is no current program: write it from scratch."
        others = "".join(f"Individual {pid} (for inspiration or recombination) is at {d}\n\n" for pid, d in other_dirs.items())
        limit = f"You have {self.max_draft_evals} draft evaluations. " if self.max_draft_evals is not None else ""
        draft = (
            f"To score your current edits, run: {draft_cmd}\n"
            f"It prints the evaluation as JSON. {limit}The final score is computed separately after you "
            "finish, from the files as you leave them: before you stop, score your final version, and if "
            "your last change made it worse, put back the best version you measured.\n\n"
            if draft_cmd
            else ""
        )
        notes = join_notes(self.instructions, redirect)
        return AGENT_PROMPT.format(
            editable=", ".join(self.project.editable),
            individuals=described + "\n",
            others=others + history,
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
        run_dir = run_directory(getattr(archive, "storage", None))
        try:
            return await self._create((*selection.parents, *selection.inspirations), len(selection.parents), run_dir)
        finally:
            self._running -= 1

    async def _create(self, individuals: tuple[Individual, ...], n_parents: int, run_dir: Path | None = None) -> Offspring:
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
            run_env = run_access(self.agent, workspace, run_dir)
            return await self._run_agent(individuals, n_parents, base, workspace, other_dirs, redirect, started, run_env)
        finally:
            if owned:
                shutil.rmtree(workspace, ignore_errors=True)
            if others_root:
                shutil.rmtree(others_root, ignore_errors=True)

    async def _run_agent(self, individuals, n_parents, base, workspace, other_dirs, redirect, started, run_env=None) -> Offspring:
        parents = [i.id for i in individuals[:n_parents]]
        drafts = (
            Drafts(self.project, self.draft_evaluate, workspace, parents, self.max_draft_evals)
            if self.draft_evaluate and self.max_draft_evals != 0
            else None
        )
        commands = AgentCommands({"evokit-eval": drafts} if drafts else {})
        async with commands:
            draft_cmd = str(commands.scripts["evokit-eval"]) if drafts else None
            prompt = self._prompt(individuals, draft_cmd, other_dirs, redirect, bool(run_env))
            env = {**(run_env or {}), **({"EVOKIT_DRAFT_EVAL": draft_cmd} if draft_cmd else {})}
            result = await self.agent.run(workspace, prompt, session=self.session if self.persistent else None, env=env)
        draft_evals = drafts.count if drafts else 0
        draft_usage = drafts.usage if drafts else {}
        usage = merge_usage(result.usage, draft_usage, {"draft_evals": draft_evals, "seconds": time.monotonic() - started})
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
