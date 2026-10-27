"""ReplAgent: a coding agent on any chat model. The model writes ```repl Python blocks, which run in
a persistent Python process in its directory, and it reads what they print (after rlmflow's REPL
and alexzhang13/rlm-minimal). The interpreter is in `repl_server.py`; `ReplProcess` runs it in a
subprocess in the agent's directory, so each agent has its own working directory and a stuck block
can be killed.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
import re
import shutil
import sys
import uuid
from pathlib import Path
from typing import Callable, Mapping

from evokit.agents import repl_server
from evokit.agents.base import AgentResult, CodingAgent, kill
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

After your blocks, stop and wait for their output; do not guess it. Nothing you describe is done
until a block has done it: files change only when your code writes them. When the task is done and
you have checked the files, reply without a repl block, which ends the session: a short title for
what you changed on the first line, a blank line, then what you did and why."""

CONFIRM = """\
Your reply had no repl block, so nothing ran and no file changed. If you meant to do something, \
write the repl block now. If the task is done, reply again without a block, with only your final \
summary: a short title on the first line, a blank line, then what you did and why."""

REPL_BLOCK = re.compile(r"```repl[ \t]*\n(.*?)```", re.DOTALL)


SERVER = Path(repl_server.__file__).read_text()


class ReplProcess:
    """`repl_server.serve()` run by `python` in a subprocess whose working directory is `directory`."""

    def __init__(self, directory: Path, env: Mapping[str, str], python: str):
        self.directory, self.env, self.python = directory, dict(env), python
        self.proc: asyncio.subprocess.Process | None = None

    async def start(self) -> None:
        self.proc = await asyncio.create_subprocess_exec(
            self.python,
            "-c",
            SERVER,
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
        if self.proc is not None:
            await kill(self.proc)
            reaped(self.proc)
            self.proc = None

    async def run(self, code: str, timeout: float | None) -> str:
        """The block's output; if it times out or the process dies, a note saying so, and a new process."""
        assert self.proc and self.proc.stdin and self.proc.stdout
        try:
            self.proc.stdin.write(json.dumps({"code": code}).encode() + b"\n")
            await self.proc.stdin.drain()
            line = await asyncio.wait_for(self.proc.stdout.readline(), timeout)
        except asyncio.TimeoutError:
            await self.close()
            await self.start()
            return f"[timed out after {timeout:.0f}s; the Python process was restarted, so its variables are gone]"
        except (BrokenPipeError, ConnectionResetError):
            line = b""
        if not line:
            code = await self.proc.wait()
            await self.close()
            await self.start()
            return f"[the Python process exited (code {code}) and was restarted, so its variables are gone]"
        return json.loads(line)["output"]


class ReplAgent(CodingAgent):
    """A coding agent on any `Model` (e.g. `OpenAIModel` for any OpenAI-compatible endpoint).

    Each reply's ```repl blocks run in order in a `ReplProcess` in the directory, and their output
    (at most `max_output` characters per block) is the next message. A reply without a block is
    answered with `CONFIRM` (models often say what they will do and stop); a second one in a row
    ends the run and is its summary. Sessions resume the conversation, not the process.

    `python` runs the blocks: by default the `python` on PATH, the one a task's `python evaluate.py`
    uses, so the agent can import what the program imports.
    """

    on_cost: Callable[[float | None], None] | None = None
    tool_style = "python"

    def __init__(
        self,
        model: Model,
        max_steps: int = 50,
        block_timeout: str | float | None = "10m",
        timeout: str | float | None = "1h",
        max_budget_usd: float | None = None,
        max_output: int = 10_000,
        system: str = SYSTEM,
        python: str | None = None,
    ):
        self.model, self.max_steps, self.max_output, self.system = model, max_steps, max_output, system
        self.block_timeout, self.timeout = parse_duration(block_timeout), parse_duration(timeout)
        self.max_budget_usd = max_budget_usd
        self.python = python or shutil.which("python") or sys.executable
        self.sessions: dict[str, list[dict[str, str]]] = {}

    def with_budget(self, usd: float | None) -> "ReplAgent":
        """A copy that stops before a model call once it has spent `usd` (it can overshoot by one call)."""
        if usd is None:
            return self
        agent = copy.copy(self)
        agent.max_budget_usd = usd if self.max_budget_usd is None else min(usd, self.max_budget_usd)
        return agent

    def with_meter(self, on_cost: Callable[[float | None], None]) -> "ReplAgent":
        """A copy that reports its cost after every model call."""
        agent = copy.copy(self)
        agent.on_cost = on_cost
        return agent

    async def check(self) -> None:
        await self.model.check()

    async def _run(self, directory, prompt, session=None, env=None) -> AgentResult:
        earlier = self.sessions.get(session or "")
        if earlier:
            prompt = "[The Python process was restarted: earlier variables are gone.]\n\n" + prompt
        messages = list(earlier or [{"role": "system", "content": self.system}])
        messages.append({"role": "user", "content": prompt})
        session = session if earlier else uuid.uuid4().hex[:12]
        usage: dict[str, float] = {}
        repl = ReplProcess(Path(directory), env or {}, self.python)
        await repl.start()
        try:
            error = await asyncio.wait_for(self._steps(messages, usage, repl), self.timeout)
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

    async def _steps(self, messages: list[dict[str, str]], usage: dict[str, float], repl: ReplProcess) -> str | None:
        """Model calls and blocks until a reply has no block; an error message if it stops early."""
        for _ in range(self.max_steps):
            if self.max_budget_usd is not None and usage.get("cost_usd", 0) >= self.max_budget_usd:
                return f"spent its ${self.max_budget_usd:g} budget"
            reply = await self.model.chat(messages)
            usage.update(merge_usage(usage, reply.usage, {"turns": 1}))
            if self.on_cost is not None:
                self.on_cost(usage.get("cost_usd"))
            messages.append({"role": "assistant", "content": reply.text})
            blocks = REPL_BLOCK.findall(reply.text)
            if not blocks:
                if messages[-2]["content"] == CONFIRM:
                    return None
                messages.append({"role": "user", "content": CONFIRM})
                continue
            outputs = []
            for code in blocks:
                log.info("agent: repl %s", " ".join(code.split())[:120])
                output = await repl.run(code, self.block_timeout)
                outputs.append(truncate(output, self.max_output) or "(no output)")
            messages.append(
                {"role": "user", "content": "\n\n".join(f"Output of block {i + 1}:\n{o}" for i, o in enumerate(outputs))}
            )
        return f"no final reply after {self.max_steps} steps"
