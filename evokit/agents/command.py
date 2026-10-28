"""CommandAgent: any command-line coding agent, one process per turn."""

from __future__ import annotations

import asyncio
import json
import os
import shlex
from pathlib import Path
from typing import Mapping, Sequence

from evokit.agents.harness import Agent, Turn, kill
from evokit.util import parse_duration, reaped, spawned, truncate


class CommandAgent(Agent):
    """Any CLI agent. `{prompt}` in argv is substituted; otherwise the prompt goes to stdin.

    If the last stdout line is a JSON object, its `summary`, `cost_usd`, `tokens`, and
    `session` fields are used. Examples: `["codex", "exec", "{prompt}"]`,
    `["cursor-agent", "-p", "{prompt}"]`.
    """

    def __init__(self, argv: str | Sequence[str], timeout: str | float | None = "1h", env: Mapping[str, str] | None = None):
        self.argv = shlex.split(argv) if isinstance(argv, str) else list(argv)
        self.timeout = parse_duration(timeout)
        self.env = dict(env or {})

    async def _run(self, cwd, prompt, session, env, budget) -> Turn:
        uses_arg = any("{prompt}" in a for a in self.argv)
        argv = [a.replace("{prompt}", prompt).replace("{session}", session or "") for a in self.argv]
        try:
            code, out, err = await _exec(
                argv, Path(cwd), {**self.env, **env}, None if uses_arg else prompt.encode(), self.timeout
            )
        except asyncio.TimeoutError:
            return Turn("", error=f"agent timed out after {self.timeout:.0f}s", session=session)
        except FileNotFoundError:
            return Turn("", error=f"{argv[0]!r} not found")
        data = {}
        lines = out.strip().splitlines()
        if lines and lines[-1].startswith("{"):
            try:
                data = json.loads(lines[-1])
            except json.JSONDecodeError:
                data = {}
        usage = {k: float(data[k]) for k in ("cost_usd", "tokens") if isinstance(data.get(k), (int, float))}
        return Turn(
            text=str(data.get("summary") or truncate(out.strip(), 4000)),
            usage=usage,
            session=data.get("session") or session,
            trace=truncate(err, 4000) if err else None,
            error=None if code == 0 else truncate(err.strip() or out.strip() or f"exit code {code}", 2000),
        )


async def _exec(argv: Sequence[str], cwd: Path, env: Mapping[str, str], stdin: bytes | None, timeout: float | None):
    proc = await asyncio.create_subprocess_exec(
        *argv,
        cwd=cwd,
        env={**os.environ, **env},
        stdin=asyncio.subprocess.PIPE if stdin is not None else asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    spawned(proc)
    try:
        out, err = await asyncio.wait_for(proc.communicate(stdin), timeout)
    except (asyncio.TimeoutError, asyncio.CancelledError):
        await kill(proc)
        raise
    finally:
        reaped(proc)
    return proc.returncode, out.decode(errors="replace"), err.decode(errors="replace")
