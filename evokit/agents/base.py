"""The coding agent interface: an agent edits a directory in place, given a prompt, calling tools if given."""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Sequence

STOP_GRACE = 60.0
"""Seconds an agent may keep going once `stop()` gives a reason, to finish what it is doing."""


@dataclass(frozen=True)
class AgentResult:
    summary: str
    usage: dict[str, float] = field(default_factory=dict)
    session: str | None = None
    trace: str | None = None
    error: str | None = None
    transcript: str | None = None  # the agent's full session, e.g. Claude Code's stream-json events


class CodingAgent(ABC):
    tool_style = "shell"
    """How the agent calls tools: "shell" (an `evokit` command) or "python" (functions in its session)."""

    async def run(
        self,
        directory: Path,
        prompt: str,
        session: str | None = None,
        env: Mapping[str, str] | None = None,
        *,
        tools: Sequence[Callable] = (),
        stop: Callable[[], str | None] | None = None,
        messages: Callable[[], str | None] | None = None,
    ) -> AgentResult:
        """Edit `directory` in place. `session` resumes an earlier conversation if supported.

        `tools` are functions the agent can call while it works (see `evokit.agents.tools`); the
        prompt tells it how. Once `stop()` gives a reason, tool calls are refused with it, and the
        agent is stopped `STOP_GRACE` seconds later if it is still going. `messages()` returning
        text passes it to the agent with its next tool result.
        """
        from evokit.agents.tools import ToolServer, describe

        directory, env = Path(directory), dict(env or {})
        if not tools and stop is None:
            return await self._run(directory, prompt, session, env)
        async with ToolServer(tools, stop=stop, messages=messages) as server:
            if "PATH" in env:
                server.env["PATH"] = server.env["PATH"].split(os.pathsep, 1)[0] + os.pathsep + env["PATH"]
            if tools:
                prompt = f"{prompt}\n\n{describe(tools, self.tool_style)}"
            return await _until(self._run(directory, prompt, session, {**env, **server.env}), stop)

    @abstractmethod
    async def _run(self, directory: Path, prompt: str, session: str | None, env: Mapping[str, str]) -> AgentResult:
        """The agent itself: edit `directory` in place, with `env` added to its environment."""

    async def check(self) -> None:
        """Raise UsageError if the agent cannot run. Called once, before a loop starts. A no-op by default."""

    def with_budget(self, usd: float | None) -> "CodingAgent":
        """This agent, stopping itself after spending `usd` in one run if it can. As it is by default."""
        return self

    def with_meter(self, on_cost: Callable[[float | None], None]) -> "CodingAgent":
        """This agent, calling `on_cost` with its cost so far as a run goes on (None: it cannot tell),
        if it can. As it is by default: its cost is only known from the run's result."""
        return self


async def _until(work, stop: Callable[[], str | None] | None) -> AgentResult:
    """`work`'s result; once `stop()` gives a reason, it gets `STOP_GRACE` seconds more, then is cancelled."""
    task = asyncio.ensure_future(work)
    try:
        while stop is not None:
            done, _ = await asyncio.wait({task}, timeout=1.0)
            if done:
                break
            if reason := stop():
                done, _ = await asyncio.wait({task}, timeout=STOP_GRACE)
                if not done:
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await task
                    return AgentResult("", error=f"stopped: {reason}")
                break
        return await task
    except asyncio.CancelledError:
        task.cancel()
        raise


async def kill(proc: asyncio.subprocess.Process) -> None:
    """Kill a process started with `start_new_session=True`, and everything it started."""
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    await proc.wait()
