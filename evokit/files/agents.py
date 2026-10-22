"""Coding agents used by CodingAgentOperator: they edit a directory given a prompt."""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import signal
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

from evokit.util import parse_duration, truncate


@dataclass(frozen=True)
class AgentResult:
    summary: str
    usage: dict[str, float] = field(default_factory=dict)
    session: str | None = None
    trace: str | None = None
    error: str | None = None


class CodingAgent(ABC):
    @abstractmethod
    async def run(
        self, directory: Path, prompt: str, session: str | None = None, env: Mapping[str, str] | None = None
    ) -> AgentResult:
        """Edit `directory` in place. `session` resumes an earlier conversation if supported."""


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
    try:
        out, err = await asyncio.wait_for(proc.communicate(stdin), timeout)
    except (asyncio.TimeoutError, asyncio.CancelledError):
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        await proc.wait()
        raise
    return proc.returncode, out.decode(errors="replace"), err.decode(errors="replace")


class ClaudeCode(CodingAgent):
    """Headless Claude Code (`claude -p --output-format json`), resumable with `--resume`."""

    def __init__(
        self,
        model: str | None = None,
        max_turns: int = 50,
        permission_mode: str = "bypassPermissions",
        max_budget_usd: float | None = None,
        timeout: str | float | None = "1h",
        extra_args: Sequence[str] = (),
        binary: str = "claude",
    ):
        self.model, self.max_turns, self.permission_mode = model, max_turns, permission_mode
        self.max_budget_usd, self.timeout = max_budget_usd, parse_duration(timeout)
        self.extra_args, self.binary = list(extra_args), binary

    def argv(self, prompt: str, session: str | None) -> list[str]:
        argv = [self.binary, "-p", prompt, "--output-format", "json", "--max-turns", str(self.max_turns)]
        argv += ["--permission-mode", self.permission_mode]
        if self.model:
            argv += ["--model", self.model]
        if self.max_budget_usd is not None:
            argv += ["--max-budget-usd", str(self.max_budget_usd)]
        if session:
            argv += ["--resume", session]
        return argv + self.extra_args

    async def run(self, directory, prompt, session=None, env=None) -> AgentResult:
        try:
            code, out, err = await _exec(self.argv(prompt, session), Path(directory), env or {}, None, self.timeout)
        except asyncio.TimeoutError:
            return AgentResult("", error=f"claude timed out after {self.timeout:.0f}s")
        except FileNotFoundError:
            return AgentResult("", error=f"{self.binary!r} not found; install Claude Code or pass binary=")
        try:
            data = json.loads(out.strip().splitlines()[-1]) if out.strip() else {}
        except json.JSONDecodeError:
            data = {}
        usage_raw = data.get("usage") or {}
        tokens = sum(
            int(usage_raw.get(k, 0) or 0)
            for k in ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
        )
        usage = {"tokens": tokens, "turns": data.get("num_turns", 0)}
        if data.get("total_cost_usd") is not None:
            usage["cost_usd"] = float(data["total_cost_usd"])
        error = None
        if code != 0 or data.get("is_error"):
            error = truncate(data.get("result") or err.strip() or out.strip() or f"exit code {code}", 2000)
        return AgentResult(
            summary=data.get("result") or "",
            usage=usage,
            session=data.get("session_id"),
            trace=truncate(err, 4000) if err else None,
            error=error,
        )


class CommandAgent(CodingAgent):
    """Any CLI agent. `{prompt}` in argv is substituted; otherwise the prompt goes to stdin.

    If the last stdout line is a JSON object, its `summary`, `cost_usd`, `tokens`, and
    `session` fields are used. Examples: `["codex", "exec", "{prompt}"]`,
    `["cursor-agent", "-p", "{prompt}"]`.
    """

    def __init__(self, argv: str | Sequence[str], timeout: str | float | None = "1h", env: Mapping[str, str] | None = None):
        self.argv = shlex.split(argv) if isinstance(argv, str) else list(argv)
        self.timeout = parse_duration(timeout)
        self.env = dict(env or {})

    async def run(self, directory, prompt, session=None, env=None) -> AgentResult:
        uses_arg = any("{prompt}" in a for a in self.argv)
        argv = [a.replace("{prompt}", prompt).replace("{session}", session or "") for a in self.argv]
        try:
            code, out, err = await _exec(
                argv, Path(directory), {**self.env, **(env or {})}, None if uses_arg else prompt.encode(), self.timeout
            )
        except asyncio.TimeoutError:
            return AgentResult("", error=f"agent timed out after {self.timeout:.0f}s")
        except FileNotFoundError:
            return AgentResult("", error=f"{argv[0]!r} not found")
        data = {}
        lines = out.strip().splitlines()
        if lines and lines[-1].startswith("{"):
            try:
                data = json.loads(lines[-1])
            except json.JSONDecodeError:
                data = {}
        usage = {k: float(data[k]) for k in ("cost_usd", "tokens") if isinstance(data.get(k), (int, float))}
        return AgentResult(
            summary=str(data.get("summary") or truncate(out.strip(), 4000)),
            usage=usage,
            session=data.get("session") or session,
            trace=truncate(err, 4000) if err else None,
            error=None if code == 0 else truncate(err.strip() or out.strip() or f"exit code {code}", 2000),
        )
