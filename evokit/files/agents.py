"""Coding agents used by CodingAgentOperator: they edit a directory given a prompt."""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
import shlex
import signal
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal, Mapping, Sequence

from evokit.errors import UsageError
from evokit.util import parse_duration, reaped, spawned, truncate

log = logging.getLogger("evokit")


@dataclass(frozen=True)
class AgentResult:
    summary: str
    usage: dict[str, float] = field(default_factory=dict)
    session: str | None = None
    trace: str | None = None
    error: str | None = None
    transcript: str | None = None  # the agent's full session, e.g. Claude Code's stream-json events


class CodingAgent(ABC):
    skill_dir: str | None = None
    """Where in its working directory the agent loads skills from (e.g. ".claude/skills"); None if it doesn't."""

    @abstractmethod
    async def run(
        self, directory: Path, prompt: str, session: str | None = None, env: Mapping[str, str] | None = None
    ) -> AgentResult:
        """Edit `directory` in place. `session` resumes an earlier conversation if supported."""

    async def check(self) -> None:
        """Raise UsageError if the agent cannot run. Called once, before a loop starts. A no-op by default."""

    def with_budget(self, usd: float | None) -> "CodingAgent":
        """This agent, stopping itself after spending `usd` in one run if it can. As it is by default."""
        return self

    def with_meter(self, on_cost: Callable[[float | None], None]) -> "CodingAgent":
        """This agent, calling `on_cost` with its cost so far as a run goes on (None: it cannot tell),
        if it can. As it is by default: its cost is only known from the run's result."""
        return self


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
        await _kill(proc)
        raise
    finally:
        reaped(proc)
    return proc.returncode, out.decode(errors="replace"), err.decode(errors="replace")


async def _kill(proc: asyncio.subprocess.Process) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    await proc.wait()


class ClaudeCode(CodingAgent):
    """Headless Claude Code (`claude -p --output-format stream-json`), resumable with `--resume`.

    The final `result` event is the answer. Claude Code can stay alive after printing it, so the
    process gets `exit_grace` seconds to exit and is then killed.

    `auth="env"` runs it with the environment as it is, so an ANTHROPIC_API_KEY there takes
    precedence; `auth="login"` removes that key, so Claude Code uses its own login.
    """

    exit_grace = 30.0
    skill_dir = ".claude/skills"
    on_cost: Callable[[float | None], None] | None = None

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
    ):
        if auth not in ("env", "login"):
            raise ValueError(f"auth must be 'env' or 'login', not {auth!r}")
        self.model, self.max_turns, self.permission_mode = model, max_turns, permission_mode
        self.max_budget_usd, self.timeout = max_budget_usd, parse_duration(timeout)
        self.extra_args, self.binary, self.auth = list(extra_args), binary, auth

    def environment(self, env: Mapping[str, str]) -> dict[str, str]:
        environment = {**os.environ, **env}
        if self.auth == "login":
            environment.pop("ANTHROPIC_API_KEY", None)
        return environment

    def with_budget(self, usd: float | None) -> "ClaudeCode":
        """A copy whose `--max-budget-usd` is at most `usd`."""
        if usd is None:
            return self
        agent = copy.copy(self)
        agent.max_budget_usd = usd if self.max_budget_usd is None else min(usd, self.max_budget_usd)
        return agent

    def with_meter(self, on_cost: Callable[[float | None], None]) -> "ClaudeCode":
        """A copy that reports its cost after every API message (see `_CostMeter`)."""
        agent = copy.copy(self)
        agent.on_cost = on_cost
        return agent

    async def check(self) -> None:
        """One one-turn call with these settings (binary, model, auth); its cost is not recorded."""
        probe = copy.copy(self)
        probe.max_turns = 1
        with tempfile.TemporaryDirectory(prefix="evokit-check-") as directory:
            result = await probe.run(Path(directory), "Reply with OK.")
        if result.error:
            raise UsageError(
                f"Claude Code cannot run: {result.error}",
                hint=f"try `{self.binary} -p hi` in this shell; check the model ({self.model or 'default'}) and auth ({self.auth})",
            )

    def argv(self, prompt: str, session: str | None) -> list[str]:
        argv = [self.binary, "-p", prompt, "--output-format", "stream-json", "--verbose", "--max-turns", str(self.max_turns)]
        argv += ["--permission-mode", self.permission_mode]
        if self.model:
            argv += ["--model", self.model]
        if self.max_budget_usd is not None:
            argv += ["--max-budget-usd", str(self.max_budget_usd)]
        if session:
            argv += ["--resume", session]
        if self.on_cost is not None:
            argv += ["--include-partial-messages"]
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
        meter = _CostMeter()

        async def result_event() -> dict:
            async for line in proc.stdout:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    event = None
                if isinstance(event, dict) and event.get("type") == "stream_event":
                    if self.on_cost is not None and meter.update(event.get("event") or {}):
                        self.on_cost(meter.cost())
                    continue
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
                await _kill(proc)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            await _kill(proc)
            stderr.cancel()
            raise
        finally:
            reaped(proc)
        return proc.returncode, data, (await stderr).decode(errors="replace")

    async def run(self, directory, prompt, session=None, env=None) -> AgentResult:
        lines: list[str] = []
        try:
            code, data, err = await self._exec(self.argv(prompt, session), Path(directory), env or {}, lines)
        except asyncio.TimeoutError:
            return AgentResult("", error=f"claude timed out after {self.timeout:.0f}s", transcript="".join(lines) or None)
        except FileNotFoundError:
            return AgentResult("", error=f"{self.binary!r} not found; install Claude Code or pass binary=")
        _learn_rates(data.get("modelUsage"))
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
        return AgentResult(
            summary=data.get("result") or "",
            usage=usage,
            session=data.get("session_id"),
            trace=truncate(err, 4000) if err else None,
            error=error,
            transcript="".join(lines) or None,
        )


_RATES: dict[str, float] = {}
"""USD per input token for each model, as Claude Code charged it in a finished run (its result's
`modelUsage`); `ClaudeCode.check()` teaches the run's model before a loop starts."""


def _weight(input_tokens, output_tokens, cache_writes, cache_reads) -> float:
    """Tokens in input-token units: Anthropic's prices are in these ratios for every model."""
    return (input_tokens or 0) + 5 * (output_tokens or 0) + 1.25 * (cache_writes or 0) + 0.1 * (cache_reads or 0)


def _learn_rates(model_usage) -> None:
    for model, u in (model_usage or {}).items():
        weight = _weight(u.get("inputTokens"), u.get("outputTokens"), u.get("cacheCreationInputTokens"), u.get("cacheReadInputTokens"))
        if weight > 0 and u.get("costUSD") is not None:
            _RATES[model] = float(u["costUSD"]) / weight


class _CostMeter:
    """A Claude Code run's cost so far, from each API message's token counts (with
    --include-partial-messages: `message_start`, and `message_delta` for the final output tokens),
    at the rates in `_RATES`. Exact for the agent's own messages; Claude Code's small background
    calls (e.g. Haiku, 1-2% of a session) are not in the stream and only show in the result."""

    def __init__(self):
        self.messages: dict[str, tuple[str, dict]] = {}
        self.current: str | None = None

    def update(self, event: dict) -> bool:
        """True if the cost may have changed."""
        if event.get("type") == "message_start":
            message = event.get("message") or {}
            self.current = message.get("id")
            self.messages[self.current] = (message.get("model", ""), dict(message.get("usage") or {}))
            return True
        if event.get("type") == "message_delta" and self.current in self.messages:
            self.messages[self.current][1].update(event.get("usage") or {})
            return True
        return False

    def cost(self) -> float | None:
        total = 0.0
        for model, u in self.messages.values():
            if model not in _RATES:
                return None
            total += _RATES[model] * _weight(
                u.get("input_tokens"), u.get("output_tokens"), u.get("cache_creation_input_tokens"), u.get("cache_read_input_tokens")
            )
        return total


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
