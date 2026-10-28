"""ClaudeCode: headless Claude Code as a coding agent."""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
from pathlib import Path
from typing import Literal, Mapping, Sequence

from evokit.agents.harness import Agent, Turn, kill, scratch
from evokit.errors import UsageError
from evokit.util import parse_duration, reaped, spawned, truncate

log = logging.getLogger("evokit")

NO_SHELL = ("Read", "Edit", "Write", "Glob", "Grep", "Bash(evokit *)")
"""`allowed_tools` for an agent that only reads and edits files and calls its session's tools (the
`evokit` command): it cannot run other commands, so it cannot run code except through those tools."""


class ClaudeCode(Agent):
    """Headless Claude Code (`claude -p --output-format stream-json`), resumable with `--resume`.

    The final `result` event is the answer. Claude Code can stay alive after printing it, so the
    process gets `exit_grace` seconds to exit and is then killed.

    `auth="env"` runs it with the environment as it is, so an ANTHROPIC_API_KEY there takes
    precedence; `auth="login"` removes that key, so Claude Code uses its own login.

    `allowed_tools` (Claude Code tool rules, e.g. `NO_SHELL`) limits it to those tools: every
    other call is denied (`--permission-mode dontAsk`). None: every tool, as `permission_mode` says.
    """

    exit_grace = 30.0
    takes_system = True

    def __init__(
        self,
        model: str | None = None,
        max_turns: int = 50,
        permission_mode: str = "bypassPermissions",
        max_budget_usd: float | None = None,
        timeout: str | float | None = "1h",
        extra_args: Sequence[str] = (),
        binary: str = "claude",
        auth: Literal["env", "login"] = "env",
        allowed_tools: Sequence[str] | None = None,
    ):
        if auth not in ("env", "login"):
            raise ValueError(f"auth must be 'env' or 'login', not {auth!r}")
        self.model, self.max_turns, self.permission_mode = model, max_turns, permission_mode
        self.max_budget_usd, self.timeout = max_budget_usd, parse_duration(timeout)
        self.extra_args, self.binary, self.auth = list(extra_args), binary, auth
        self.allowed_tools = None if allowed_tools is None else list(allowed_tools)

    def environment(self, env: Mapping[str, str]) -> dict[str, str]:
        environment = {**os.environ, **env}
        if self.auth == "login":
            environment.pop("ANTHROPIC_API_KEY", None)
        return environment

    async def check(self) -> None:
        """One one-turn call with these settings (binary, model, auth); its cost is not recorded."""
        probe = copy.copy(self)
        probe.max_turns = 1
        async with scratch(probe) as session:
            result = await session.prompt("Reply with OK.")
        if result.error:
            raise UsageError(
                f"Claude Code cannot run: {result.error}",
                hint=f"try `{self.binary} -p hi` in this shell; check the model ({self.model or 'default'}) and auth ({self.auth})",
            )

    def argv(self, prompt: str, session: str | None, budget: float | None = None, system: str | None = None) -> list[str]:
        argv = [self.binary, "-p", prompt, "--output-format", "stream-json", "--verbose", "--max-turns", str(self.max_turns)]
        if self.allowed_tools is None:
            argv += ["--permission-mode", self.permission_mode]
        else:
            argv += ["--permission-mode", "dontAsk", "--allowedTools", ",".join(self.allowed_tools)]
        if self.model:
            argv += ["--model", self.model]
        budgets = [b for b in (self.max_budget_usd, budget) if b is not None]
        if budgets:
            argv += ["--max-budget-usd", str(min(budgets))]
        if system:
            argv += ["--append-system-prompt", system]
        if session:
            argv += ["--resume", session]
        return argv + self.extra_args

    async def _exec(
        self, argv: list[str], cwd: Path, env: Mapping[str, str], transcript: list[str]
    ) -> tuple[int | None, dict, str]:
        """(exit code, the result event or {}, stderr). Every event is appended to `transcript` as it
        arrives, so a timed-out session keeps what it did."""
        proc = await asyncio.create_subprocess_exec(
            *argv,
            cwd=cwd,
            env=self.environment(env),
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
            limit=1 << 26,
        )
        spawned(proc)
        stderr = asyncio.create_task(proc.stderr.read())
        data: dict = {}

        async def result_event() -> dict:
            async for line in proc.stdout:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    event = None
                transcript.append(line.decode(errors="replace"))
                if not isinstance(event, dict):
                    continue
                if event.get("type") == "result":
                    return event
                for step in _steps(event):
                    log.info("agent: %s", step)
            return {}

        try:
            data = await asyncio.wait_for(result_event(), self.timeout)
            try:
                await asyncio.wait_for(proc.wait(), self.exit_grace if data else None)
            except asyncio.TimeoutError:
                await kill(proc)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            await kill(proc)
            stderr.cancel()
            raise
        finally:
            reaped(proc)
        return proc.returncode, data, (await stderr).decode(errors="replace")

    async def _run(self, cwd, prompt, session, env, budget, system=None) -> Turn:
        lines: list[str] = []
        try:
            code, data, err = await self._exec(self.argv(prompt, session, budget, system), Path(cwd), env, lines)
        except asyncio.TimeoutError:
            return Turn("", error=f"claude timed out after {self.timeout:.0f}s", transcript="".join(lines) or None, session=session)
        except FileNotFoundError:
            return Turn("", error=f"{self.binary!r} not found; install Claude Code or pass binary=")
        usage_raw = data.get("usage") or {}
        tokens = sum(
            int(usage_raw.get(k, 0) or 0)
            for k in ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
        )
        usage = {"tokens": tokens, "turns": data.get("num_turns", 0)}
        if data.get("total_cost_usd") is not None:
            usage["cost_usd"] = float(data["total_cost_usd"])
        error = None
        subtype = str(data.get("subtype") or "")
        if data.get("is_error") or subtype.startswith("error") or not data:
            reason = data.get("result") or (subtype if subtype.startswith("error") else "") or err.strip()
            error = truncate(reason or f"claude printed no result (exit code {code})", 2000)
        return Turn(
            text=data.get("result") or "",
            usage=usage,
            session=data.get("session_id") or session,
            trace=truncate(err, 4000) if err else None,
            error=error,
            transcript="".join(lines) or None,
        )


_TOOL_ARGUMENT = ("file_path", "command", "pattern", "path", "url", "query", "description")


def _steps(event: dict) -> list[str]:
    """One short line per tool call or message in a Claude Code `assistant` event."""
    if event.get("type") != "assistant":
        return []
    steps = []
    for block in (event.get("message") or {}).get("content") or []:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "tool_use":
            arguments = block.get("input") or {}
            argument = next((str(arguments[k]) for k in _TOOL_ARGUMENT if arguments.get(k)), "")
            steps.append(f"{block.get('name', 'tool')} {_one_line(argument, 120)}".rstrip())
        elif block.get("type") == "text" and block.get("text", "").strip():
            steps.append(repr(_one_line(block["text"], 200)))
    return steps


def _one_line(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."
