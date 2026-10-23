"""A small coding agent: the model writes ```repl Python blocks, which run in a persistent Python
process in the working directory, and it sees their output (after alexzhang13/rlm-minimal)."""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
import re
import signal
import sys
import uuid
from pathlib import Path
from typing import Callable, Mapping

from evokit.files.agents import AgentResult, CodingAgent
from evokit.operators.models import Model
from evokit.records import merge_usage
from evokit.util import parse_duration, reaped, spawned, truncate

log = logging.getLogger("evokit")

SYSTEM = """\
You are a coding agent working in a directory. You act by writing Python in repl blocks:

```repl
print(open("main.py").read())
```

Every block runs in one persistent Python process whose working directory is that directory, so
imports, variables and functions stay defined between blocks. You see what a block prints, and the
value of its last line if that is an expression. `sh(command)` runs a shell command in the
directory, prints its output and returns its exit code. Edit files by writing them from Python.

After your blocks, stop and wait for their output; do not guess it. When the task is done, reply
without a repl block: a short title for what you changed on the first line, a blank line, then what
you did and why."""

REPL_BLOCK = re.compile(r"```repl[ \t]*\n(.*?)```", re.DOTALL)

_CHILD = r'''
import ast, json, linecache, os, subprocess, sys, tempfile, traceback

requests = os.fdopen(os.dup(0))
replies = os.fdopen(os.dup(1), "w")
stdout, stderr = os.dup(1), os.dup(2)
os.dup2(os.open(os.devnull, os.O_RDONLY), 0)
sys.stdin = open(os.devnull)


def sh(command, timeout=None):
    """Run a shell command in the working directory, print its output, and return its exit code."""
    code = subprocess.run(command, shell=True, stderr=subprocess.STDOUT, timeout=timeout).returncode
    if code:
        print(f"[exit code {code}]")
    return code


namespace = {"__name__": "__main__", "__builtins__": __builtins__, "sh": sh}


def run(code, name):
    linecache.cache[name] = (len(code), None, code.splitlines(True), name)
    tree = ast.parse(code, name)
    last = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
    exec(compile(tree, name, "exec"), namespace)
    if last is not None:
        value = eval(compile(ast.Expression(last.value), name, "eval"), namespace)
        if value is not None:
            print(repr(value))


for number, line in enumerate(requests):
    with tempfile.TemporaryFile() as out:
        os.dup2(out.fileno(), 1)
        os.dup2(out.fileno(), 2)
        try:
            run(json.loads(line)["code"], f"<block {number + 1}>")
        except BaseException:
            kind, error, trace = sys.exc_info()
            while trace and trace.tb_frame.f_code.co_filename == "<string>":
                trace = trace.tb_next
            if isinstance(error, SyntaxError):
                trace = None
            traceback.print_exception(kind, error, trace)
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(stdout, 1)
            os.dup2(stderr, 2)
        out.seek(0)
        output = out.read().decode(errors="replace")
    replies.write(json.dumps({"output": output}) + "\n")
    replies.flush()
'''


class _Repl:
    """The persistent Python process. Its output is everything written to file descriptors 1 and 2,
    so the output of subprocesses (`sh`, os.system) is included."""

    def __init__(self, directory: Path, env: Mapping[str, str]):
        self.directory, self.env = directory, dict(env)
        self.proc: asyncio.subprocess.Process | None = None

    async def start(self) -> None:
        self.proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-u",
            "-c",
            _CHILD,
            cwd=self.directory,
            env={**os.environ, **self.env},
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            start_new_session=True,
            limit=1 << 26,
        )
        spawned(self.proc)

    async def close(self) -> None:
        if self.proc is None:
            return
        try:
            os.killpg(self.proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        await self.proc.wait()
        reaped(self.proc)
        self.proc = None

    async def restart(self) -> None:
        await self.close()
        await self.start()

    async def run(self, code: str, timeout: float | None) -> str:
        assert self.proc and self.proc.stdin and self.proc.stdout
        try:
            self.proc.stdin.write(json.dumps({"code": code}).encode() + b"\n")
            await self.proc.stdin.drain()
            line = await asyncio.wait_for(self.proc.stdout.readline(), timeout)
        except asyncio.TimeoutError:
            await self.restart()
            return f"[timed out after {timeout:.0f}s; the Python process was restarted, so its variables are gone]"
        except (BrokenPipeError, ConnectionResetError):
            line = b""
        if not line:
            code = await self.proc.wait()
            await self.restart()
            return f"[the Python process exited (code {code}) and was restarted, so its variables are gone]"
        return json.loads(line)["output"]


class ReplAgent(CodingAgent):
    """A coding agent on any `Model` (e.g. `OpenAIModel` for any OpenAI-compatible endpoint).

    Each reply's ```repl blocks run in order in a persistent Python process in the directory, and
    their output (at most `max_output` characters per block) is the next message. A reply without
    a block ends the run and is its summary. Sessions resume the conversation, not the process.
    """

    on_cost: Callable[[float | None], None] | None = None

    def __init__(
        self,
        model: Model,
        max_steps: int = 50,
        block_timeout: str | float | None = "10m",
        timeout: str | float | None = "1h",
        max_budget_usd: float | None = None,
        max_output: int = 10_000,
        system: str = SYSTEM,
    ):
        self.model, self.max_steps, self.max_output, self.system = (
            model,
            max_steps,
            max_output,
            system,
        )
        self.block_timeout, self.timeout = (
            parse_duration(block_timeout),
            parse_duration(timeout),
        )
        self.max_budget_usd = max_budget_usd
        self.sessions: dict[str, list[dict[str, str]]] = {}

    def with_budget(self, usd: float | None) -> "ReplAgent":
        """A copy that stops before a model call once it has spent `usd` (it can overshoot by one call)."""
        if usd is None:
            return self
        agent = copy.copy(self)
        agent.max_budget_usd = (
            usd if self.max_budget_usd is None else min(usd, self.max_budget_usd)
        )
        return agent

    def with_meter(self, on_cost: Callable[[float | None], None]) -> "ReplAgent":
        """A copy that reports its cost after every model call."""
        agent = copy.copy(self)
        agent.on_cost = on_cost
        return agent

    async def check(self) -> None:
        await self.model.check()

    async def run(self, directory, prompt, session=None, env=None) -> AgentResult:
        earlier = self.sessions.get(session or "")
        if earlier:
            prompt = (
                "[The Python process was restarted: earlier variables are gone.]\n\n"
                + prompt
            )
        messages = list(earlier or [{"role": "system", "content": self.system}])
        messages.append({"role": "user", "content": prompt})
        session = session if earlier else uuid.uuid4().hex[:12]
        usage: dict[str, float] = {}
        repl = _Repl(Path(directory), env or {})
        await repl.start()
        try:
            error = await asyncio.wait_for(
                self._steps(messages, usage, repl), self.timeout
            )
        except asyncio.TimeoutError:
            error = f"timed out after {self.timeout:.0f}s"
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        finally:
            await repl.close()
            self.sessions[session] = messages
        replies = [m["content"] for m in messages if m["role"] == "assistant"]
        return AgentResult(
            summary=replies[-1] if replies else "",
            usage=usage,
            session=session,
            error=error and truncate(error, 2000),
            transcript="".join(json.dumps(m) + "\n" for m in messages),
        )

    async def _steps(
        self, messages: list[dict[str, str]], usage: dict[str, float], repl: _Repl
    ) -> str | None:
        """Model calls and blocks until a reply has no block; an error message if it stops early."""
        for _ in range(self.max_steps):
            if (
                self.max_budget_usd is not None
                and usage.get("cost_usd", 0) >= self.max_budget_usd
            ):
                return f"spent its ${self.max_budget_usd:g} budget"
            reply = await self.model.chat(messages)
            usage.update(merge_usage(usage, reply.usage, {"turns": 1}))
            if self.on_cost is not None:
                self.on_cost(usage.get("cost_usd"))
            messages.append({"role": "assistant", "content": reply.text})
            blocks = REPL_BLOCK.findall(reply.text)
            if not blocks:
                return None
            outputs = []
            for code in blocks:
                log.info("agent: repl %s", " ".join(code.split())[:120])
                output = await repl.run(code, self.block_timeout)
                outputs.append(truncate(output, self.max_output) or "(no output)")
            messages.append(
                {
                    "role": "user",
                    "content": "\n\n".join(
                        f"Output of block {i + 1}:\n{o}" for i, o in enumerate(outputs)
                    ),
                }
            )
        return f"no final reply after {self.max_steps} steps"
