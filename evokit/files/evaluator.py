from __future__ import annotations

import asyncio
import json
import os
import re
import shlex
import shutil
import signal
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from evokit.files.project import Project
from evokit.files.tree import Files
from evokit.loop import as_eval_fn
from evokit.records import Evaluation, Individual
from evokit.util import parse_duration, reaped, spawned, truncate

FEEDBACK_LIMIT = 4000

_METRIC_LINE = re.compile(r"^\s*([A-Za-z_][\w.\-]*)\s*[:=]\s*(-?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|true|false|True|False)\s*$")


def parse_metrics(stdout: str) -> dict[str, Any] | None:
    """The last stdout line that is a JSON object wins; else every `name: number` line."""
    for line in reversed(stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                return data
    metrics: dict[str, Any] = {}
    for line in stdout.splitlines():
        match = _METRIC_LINE.match(line)
        if match:
            name, raw = match.groups()
            metrics[name] = raw.lower() == "true" if raw.lower() in ("true", "false") else float(raw)
    return metrics or None


class ScriptEvaluator:
    """An eval function: write the project (environment + genome) to a fresh directory and run
    a command there.

        eval_fn = ScriptEvaluator(project, "python evaluate.py", timeout="2m")

    The command prints its metrics (a JSON object, or `name: number` lines); the numbers become
    objectives. A non-zero exit is a failed evaluation (no objectives), with stderr as feedback:
    that is how a script rejects a program. Files the command writes to $EVOKIT_WORKDIR become
    the evaluation's logs, stored in the individual's artifact folder.
    """

    def __init__(
        self,
        project: Project,
        command: str | Sequence[str],
        timeout: str | float | None = None,
        env: Mapping[str, str] | None = None,
    ):
        self.project = project
        self.argv = shlex.split(command) if isinstance(command, str) else list(command)
        self.timeout = parse_duration(timeout)
        self.env = dict(env or {})

    async def __call__(self, individual: Individual) -> Evaluation:
        return await self.evaluate_files(individual.genome)

    async def evaluate_files(self, genome: Files) -> Evaluation:
        directory = Path(tempfile.mkdtemp(prefix="evokit-eval-"))
        workdir = Path(tempfile.mkdtemp(prefix="evokit-eval-work-"))
        try:
            await asyncio.to_thread(self.project.write, genome, directory)
            evaluation = await self.run(directory, {"EVOKIT_WORKDIR": str(workdir)})
            return replace(evaluation, logs=await asyncio.to_thread(read_folder, workdir))
        finally:
            await asyncio.to_thread(shutil.rmtree, directory, True)
            await asyncio.to_thread(shutil.rmtree, workdir, True)

    async def run(self, directory: Path, env: Mapping[str, str] | None = None) -> Evaluation:
        try:
            proc = await asyncio.create_subprocess_exec(
                *self.argv,
                cwd=directory,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                stdin=asyncio.subprocess.DEVNULL,
                env={**os.environ, **self.env, **(env or {})},
                start_new_session=True,
            )
        except OSError as exc:
            return Evaluation(feedback=f"could not start {self.argv[0]!r}: {exc}")
        spawned(proc)
        try:
            stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), self.timeout)
        except asyncio.TimeoutError:
            _kill(proc)
            await proc.wait()
            return Evaluation(feedback=f"evaluation exceeded {self.timeout:.0f}s")
        except asyncio.CancelledError:
            _kill(proc)
            raise
        finally:
            reaped(proc)
        stdout = stdout_b.decode(errors="replace")
        stderr = stderr_b.decode(errors="replace")
        metrics = parse_metrics(stdout)
        if proc.returncode != 0:
            reported = Evaluation.from_metrics(metrics) if metrics else Evaluation()
            tail = truncate(stderr.strip() or stdout.strip(), FEEDBACK_LIMIT)
            feedback = "\n".join(x for x in (reported.feedback, f"exit code {proc.returncode}", tail) if x)
            return Evaluation(feedback=feedback, usage=reported.usage)
        if not metrics:
            return Evaluation(feedback="evaluator printed no metrics\n" + (truncate(stdout + stderr, FEEDBACK_LIMIT) or ""))
        evaluation = Evaluation.from_metrics(metrics)
        if not evaluation.objectives:
            return replace(evaluation, feedback="\n".join(x for x in ("evaluator printed no numeric metrics", evaluation.feedback) if x))
        return evaluation

    def __repr__(self) -> str:
        return f"ScriptEvaluator({shlex.join(self.argv)!r})"


def read_folder(folder: Path) -> dict[str, bytes]:
    """Every file under `folder`, by its path relative to it."""
    return {p.relative_to(folder).as_posix(): p.read_bytes() for p in sorted(folder.rglob("*")) if p.is_file()}


def code_lines(text: str, comment: str = "#") -> int:
    """Non-blank lines that are not comments."""
    return sum(1 for line in text.splitlines() if line.strip() and not line.strip().startswith(comment))


def with_lines(eval_fn, name: str = "lines", comment: str = "#"):
    """An eval function that adds an objective `name`, the code lines of the genome's files, to
    every successful evaluation of `eval_fn`."""

    inner = as_eval_fn(eval_fn)

    async def evaluate_with_lines(individual: Individual) -> Evaluation:
        evaluation = await inner(individual)
        if not evaluation.objectives:
            return evaluation
        lines = sum(code_lines(individual.genome.text(p), comment) for p in individual.genome.paths())
        return replace(evaluation, objectives={**evaluation.objectives, name: float(lines)})

    return evaluate_with_lines


def _kill(proc: asyncio.subprocess.Process) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
