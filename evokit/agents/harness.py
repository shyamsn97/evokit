"""How EvoKit drives a coding agent (a harness: Claude Code, Codex, pi, ReplAgent, your own).

Shaped like the harnesses' own SDKs (pi, tau, the Claude Agent SDK): a session in a directory,
prompts that stream events, messages mid-turn, stopping, and resuming by id.

    async with agent.session("work", tools=[add]) as session:
        async for event in session.prompt("Improve main.py"):
            match event:
                case ToolCall(name="add"):
                    session.send("Try a different data structure.")  # reaches the agent at its next step
                case Turn():
                    print(event.text, event.usage)
        turn = await session.prompt("Now tidy it up.")             # just the Turn
        turn = await session.prompt("Name it.", output=Name)        # turn.output: a Name, from the finish tool

`agent.prompt(text)` and `agent.send(text)` talk to the conversation the agent was started in
(`experiment.start(agent)` binds one); an agent that was not started answers in a session of its
own. `agent.fresh()` is the same agent with no conversation, e.g. a reviewer.

Every tool call goes through `Session.call`, so events, messages and stopping work the same in
every harness; a transport (the `evokit` command, SDK tools, ReplAgent's Python functions) only
carries calls to it. Events cover the session's tools, not the harness's own (Bash, Edit): those
belong to the user's harness configuration.

`system` is added to the harness's own system prompt. `output` (a dataclass) asks for a
structured result: the agent gets a `finish` tool taking the dataclass's fields, and the Turn's
`output` is what it passed.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import copy
import dataclasses
import inspect
import logging
import os
import signal
import tempfile
import typing
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, AsyncContextManager, AsyncIterator, Callable, Mapping, Sequence

from evokit.errors import EvoError, UsageError

log = logging.getLogger("evokit")

STOP_GRACE = 60.0
"""Seconds an agent may keep going after `Session.stop`, to finish what it is doing."""

FINISH = "finish"
REMIND = "You have not called {finish} yet. Call it now, with your result, then end your turn."

Tool = Callable[..., Any]


@dataclass(frozen=True)
class Turn:
    """What one prompt produced; the last event of `Session.prompt`."""

    text: str                                               # the agent's final reply
    usage: dict[str, float] = field(default_factory=dict)  # this turn's cost_usd, tokens, ...
    transcript: str | None = None                           # this turn's events, as JSON lines
    error: str | None = None                                # set if the turn failed or was stopped
    session: str | None = None                              # the conversation's id, to resume it
    trace: str | None = None                                # the harness's stderr, for diagnosis
    calls: tuple[str, ...] = ()                             # the session's tools it called, in order
    stopped: str | None = None                              # why `Session.stop` ended it, if it did
    output: Any = None                                      # with `prompt(output=...)`: what it passed to finish


@dataclass(frozen=True)
class ToolCall:
    """The agent called one of EvoKit's tools. `reply` is what it gets back: {"result"} or
    {"error"}. The reply is sent once the loop body has handled the event, so a `send` or
    `stop` there reaches the agent with this very reply."""

    name: str
    args: dict[str, Any]
    reply: dict[str, Any]


Event = ToolCall | Turn


class Prompt:
    """One prompt in flight: `async for event in prompt` yields each ToolCall, then the Turn;
    `await prompt` gives just the Turn."""

    def __init__(self, events: AsyncIterator[Event]):
        self._events = events

    def __aiter__(self) -> AsyncIterator[Event]:
        return self._events

    def __await__(self):
        return self._turn().__await__()

    async def _turn(self) -> Turn:
        async with contextlib.aclosing(self._events) as events:
            async for event in events:
                if isinstance(event, Turn):
                    return event
        raise AssertionError("a prompt always ends with its Turn")


class Session:
    """A conversation with a harness in one directory. Made by `Agent.session`; use `async with`."""

    sends_natively = False
    """True if the harness takes `send` text mid-turn; otherwise it goes with the next tool reply."""

    tool_style = "shell"
    """How this session's agent is told about tools: "shell" (the `evokit` command) or "python"."""

    def __init__(
        self,
        cwd: str | os.PathLike,
        tools: Sequence[Tool] = (),
        resume: str | None = None,
        env: Mapping[str, str] | None = None,
        system: str | None = None,
    ):
        from evokit.agents.tools import tool_name

        self.cwd = Path(cwd)
        self.tools = {tool_name(t): t for t in tools}
        self.id = resume
        self.env = dict(env or {})
        self.system = system
        self._output: Any = None
        self._sent: list[str] = []
        self._calls: list[str] = []
        self._stopping: str | None = None
        self._task: asyncio.Task | None = None
        self._timer: asyncio.TimerHandle | None = None
        self._events: asyncio.Queue | None = None


    # what callers use

    def prompt(self, text: str, budget: float | None = None, output: type | None = None) -> Prompt:
        """Run one turn: send `text` (after anything sent and not yet delivered), spending at most
        `budget` US dollars if the harness can enforce it. Yields a ToolCall for each call of the
        session's tools, then the Turn; or await it for the Turn. With `output` (a dataclass), the
        agent ends by calling `finish` with its fields, and `Turn.output` is that dataclass (None,
        with `Turn.error` set, if it never did)."""
        return Prompt(self._turn(text, budget, output))

    async def _turn(self, text: str, budget: float | None, output: type | None = None) -> AsyncIterator[Event]:
        if self._task is not None:
            raise UsageError("a turn is already running", hint="send() to it, or wait for its Turn")
        if self._sent:
            text = "\n\n".join([*self._sent, text])
            self._sent.clear()
        self._stopping = None
        self._calls = []
        self._output = None
        if output is not None:
            self.tools[FINISH] = self._finish_tool(output)
            text = f"{text}\n\n{self._describe_finish()}"
        events = self._events = asyncio.Queue()
        task = self._task = asyncio.ensure_future(self._finished(text, budget, output))
        task.add_done_callback(lambda _: events.put_nowait(None))
        try:
            while (item := await events.get()) is not None:
                event, handled = item
                try:
                    yield event
                finally:
                    if not handled.done():
                        handled.set_result(None)
            try:
                turn = task.result()
            except asyncio.CancelledError:
                turn = Turn("", error=f"stopped: {self._stopping or 'interrupted'}", session=self.id)
            self.id = turn.session or self.id
            if output is not None and self._output is None and turn.error is None:
                turn = replace(turn, error=f"the agent ended its turn without calling {FINISH}")
            yield replace(turn, calls=tuple(self._calls), stopped=self._stopping, output=self._output)
        finally:
            if output is not None:
                self.tools.pop(FINISH, None)
            await self._end_turn()

    async def _finished(self, text: str, budget: float | None, output: type | None) -> Turn:
        """The turn; with `output`, reminded once to call finish if it ended without."""
        turn = await self._prompt(text, budget)
        if output is None or self._output is not None or turn.error or self._stopping is not None:
            return turn
        self.id = turn.session or self.id
        again = await self._prompt(REMIND.format(finish=self._finish_usage()), budget)
        usage = {k: turn.usage.get(k, 0) + again.usage.get(k, 0) for k in {*turn.usage, *again.usage}}
        transcript = (turn.transcript or "") + (again.transcript or "") or None
        return replace(again, usage=usage, transcript=transcript)

    def _finish_tool(self, output: type) -> Tool:
        if not dataclasses.is_dataclass(output):
            raise UsageError(f"output must be a dataclass, not {output!r}")
        hints = typing.get_type_hints(output)

        def default(f: dataclasses.Field) -> Any:
            if f.default is not dataclasses.MISSING:
                return f.default
            if f.default_factory is not dataclasses.MISSING:
                return f.default_factory()
            return inspect.Parameter.empty

        params = [
            inspect.Parameter(f.name, inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=hints.get(f.name, Any), default=default(f))
            for f in dataclasses.fields(output)
        ]

        def finish(**fields: Any) -> str:
            self._output = output(**fields)
            return "recorded; end your turn now"

        finish.__signature__ = inspect.Signature(params)
        finish.__annotations__ = {p.name: p.annotation for p in params}
        finish.__doc__ = f"Finish your turn with your result. {inspect.getdoc(output) or ''}".strip()
        return finish

    def _finish_usage(self) -> str:
        from evokit.agents.tools import usage

        tool = self.tools[FINISH]
        return f"`{usage(tool)}`" if self.tool_style == "shell" else f"`{FINISH}{inspect.signature(tool)}`"

    def _describe_finish(self) -> str:
        from evokit.agents.tools import tool_doc

        return f"When you are done, call {self._finish_usage()} once, then end your turn. {tool_doc(self.tools[FINISH])}"

    def send(self, text: str) -> None:
        """Tell the agent `text` at its next step: natively if the harness can, else with its next
        tool reply, or else before the next prompt."""
        self._sent.append(text)

    def stop(self, reason: str = "stopped", grace: float | None = None) -> None:
        """End the current turn: from now on every tool call is refused with `reason`, and the turn is
        interrupted `grace` seconds from now (default STOP_GRACE), so the agent can finish what it is doing."""
        if self._stopping is not None:
            return
        self._stopping = reason
        if self._task is not None and not self._task.done():
            delay = STOP_GRACE if grace is None else grace
            self._timer = asyncio.get_running_loop().call_later(delay, self._task.cancel)

    # what transports use

    async def call(self, name: str, kwargs: Mapping[str, Any] | None = None, args: Sequence[Any] = ()) -> dict:
        """Run one tool call: {"result"} or {"error", "hint"?}, plus "stop" once the session is
        stopping and "message" with text sent to the agent."""
        from evokit.agents.tools import call, encode

        bound: dict[str, Any] = {}
        tool = self.tools.get(name)
        if self._stopping is not None:
            reply: dict[str, Any] = {"error": f"stopped ({self._stopping}); end your turn now"}
        elif self._output is not None:
            reply = {"error": f"you already called {FINISH}; end your turn now"}
        elif tool is None:
            reply = {"error": f"unknown tool {name!r}", "hint": f"tools: {', '.join(self.tools)}"}
        else:
            self._calls.append(name)
            try:
                bound = dict(inspect.signature(tool).bind(*args, **(kwargs or {})).arguments)
            except TypeError as exc:
                reply = {"error": str(exc), "hint": f"{name}{inspect.signature(tool)}"}
            else:
                try:
                    reply = {"result": encode(await call(tool, bound))}
                except EvoError as exc:
                    reply = {"error": exc.message, **({"hint": exc.hint} if exc.hint else {})}
                except Exception as exc:
                    log.exception("tool %s failed", name)
                    reply = {"error": f"{type(exc).__name__}: {exc}"}
        if self._events is not None:
            handled = asyncio.get_running_loop().create_future()
            self._events.put_nowait((ToolCall(name, bound, reply), handled))
            await handled
        if self._stopping is not None:
            reply["stop"] = self._stopping
        if self._sent and not self.sends_natively:
            reply["message"] = "\n\n".join(self._sent)
            self._sent.clear()
        return reply

    async def call_argv(self, name: str, argv: Sequence[str]) -> dict:
        """`call`, with its arguments as command-line words (the `evokit` command)."""
        from evokit.agents.tools import parse_argv

        tool = self.tools.get(name)
        if tool is None or self._stopping is not None or self._output is not None:
            return await self.call(name)
        try:
            kwargs = parse_argv(tool, argv)
        except UsageError as exc:
            return {"error": exc.message, **({"hint": exc.hint} if exc.hint else {})}
        return await self.call(name, kwargs)

    # what harnesses implement

    async def _prompt(self, text: str, budget: float | None) -> Turn:
        """One turn in the harness, spending at most `budget` US dollars if it can enforce that.
        Cancelled to stop it."""
        raise NotImplementedError

    async def _open(self) -> None:
        """Start the harness (processes, clients)."""

    async def _close(self) -> None:
        """Release the harness."""

    async def _end_turn(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        task, self._task = self._task, None
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        events, self._events = self._events, None
        while events is not None and not events.empty():
            item = events.get_nowait()
            if item is not None and not item[1].done():
                item[1].set_result(None)

    async def __aenter__(self) -> "Session":
        await self._open()
        return self

    async def __aexit__(self, *exc) -> None:
        try:
            await self._end_turn()
        finally:
            await self._close()


class TurnSession(Session):
    """The session of an agent that runs one process per turn, resumed by session id (`Agent._run`).

    EvoKit's tools reach it as the `evokit` command on its PATH (or, for ReplAgent, Python
    functions), described in the first prompt."""

    def __init__(self, agent: "Agent", cwd, tools=(), resume=None, env=None, system=None):
        super().__init__(cwd, tools, resume, env, system)
        self.agent = agent
        self.tool_style = agent.tool_style
        self._server = None
        self._described = False
        self._introduced = resume is not None

    async def _open(self) -> None:
        from evokit.agents.tools import ShellTools

        self._server = await ShellTools(self).__aenter__()

    async def _close(self) -> None:
        if self._server is not None:
            await self._server.__aexit__(None, None, None)

    async def _prompt(self, text: str, budget: float | None) -> Turn:
        from evokit.agents.tools import describe

        listed = {n: t for n, t in self.tools.items() if n != FINISH}
        if listed and not self._described:
            text = f"{text}\n\n{describe(list(listed.values()), self.agent.tool_style)}"
            self._described = True
        env = dict(self.env)
        if self._server:
            tools_dir, _, inherited = self._server.env["PATH"].partition(os.pathsep)
            env.update(self._server.env, PATH=tools_dir + os.pathsep + env.get("PATH", inherited))
        if self.agent.takes_system:
            return await self.agent._run(self.cwd, text, self.id, env, budget, system=self.system)
        if self.system and not self._introduced:
            text = f"{self.system}\n\n{text}"
        self._introduced = True
        return await self.agent._run(self.cwd, text, self.id, env, budget)


Opener = Callable[["Agent"], AsyncContextManager[Any]]
"""`opener(agent)`: a conversation (anything with `prompt(text)` and `send(text)`) for an agent
that was not started."""

_bound: contextvars.ContextVar[Mapping[int, Any]] = contextvars.ContextVar("evokit_bound", default={})
_opener: contextvars.ContextVar[Opener | None] = contextvars.ContextVar("evokit_opener", default=None)


@contextlib.contextmanager
def bind(agent: "Agent", conversation: Any, opener: Opener | None = None):
    """Within the block (and the asyncio tasks it starts), `agent.prompt` and `agent.send` go to
    `conversation`, and an agent that was not started gets its conversation from `opener`."""
    bound = _bound.set({**_bound.get(), id(agent): conversation})
    opened = _opener.set(opener or _opener.get())
    try:
        yield conversation
    finally:
        _opener.reset(opened)
        _bound.reset(bound)


@contextlib.asynccontextmanager
async def scratch(agent: "Agent"):
    """A session of its own in a fresh, empty directory: for an agent that only answers."""
    with tempfile.TemporaryDirectory(prefix="evokit-agent-") as directory:
        async with agent.session(directory) as session:
            yield session


class Agent:
    """A coding harness EvoKit can drive.

    Implement either `_run` (one process per turn, resumed by session id; the default session
    serves its tools as an `evokit` command), or `session` (for a harness with its own SDK)."""

    tool_style = "shell"
    """How the agent is told about its tools: "shell" (the `evokit` command) or "python" (functions)."""

    takes_system = False
    """True if `_run` takes `system=` (added to the harness's system prompt); otherwise the default
    session puts the system prompt before the first prompt."""

    def session(
        self,
        cwd: str | os.PathLike,
        tools: Sequence[Tool] = (),
        resume: str | None = None,
        env: Mapping[str, str] | None = None,
        system: str | None = None,
    ) -> Session:
        """A new (or, with `resume`, continued) conversation in `cwd` that can call `tools`, with
        `env` added to its environment and `system` to its system prompt."""
        return TurnSession(self, cwd, tools, resume, env, system)

    async def _run(self, cwd: Path, prompt: str, session: str | None, env: Mapping[str, str], budget: float | None) -> Turn:
        """One turn: work in `cwd` on `prompt`, continuing `session` if given, with `env` added to the
        environment, spending at most `budget` US dollars if it can. With `takes_system`, it also
        gets `system=`."""
        raise NotImplementedError(f"{type(self).__name__} implements neither _run nor session")

    async def check(self) -> None:
        """Raise UsageError if the harness cannot run (not installed, no credentials). A no-op by default."""

    def prompt(self, text: str, output: type | None = None) -> Prompt:
        """Send `text` to the conversation the agent was started in; an agent that was not started
        answers in a conversation of its own. Iterate it for the events, or await it for the Turn.
        With `output` (a dataclass), `Turn.output` is the agent's structured result (see `Session.prompt`)."""
        return Prompt(self._prompt(text, output))

    async def _prompt(self, text: str, output: type | None = None) -> AsyncIterator[Event]:
        conversation = _bound.get().get(id(self))
        async with contextlib.AsyncExitStack() as stack:
            if conversation is None:
                conversation = await stack.enter_async_context((_opener.get() or scratch)(self))
            prompt = conversation.prompt(text) if output is None else conversation.prompt(text, output=output)
            events = await stack.enter_async_context(contextlib.aclosing(aiter(prompt)))
            async for event in events:
                yield event

    def send(self, text: str) -> None:
        """Tell the started agent `text` at its next step (see `Session.send`)."""
        conversation = _bound.get().get(id(self))
        if conversation is None:
            raise UsageError("this agent was not started", hint="send inside `async with experiment.start(agent):`")
        conversation.send(text)

    def fresh(self) -> "Agent":
        """The same agent with no conversation: prompting it starts afresh, e.g. as a reviewer."""
        return copy.copy(self)


async def kill(proc: asyncio.subprocess.Process) -> None:
    """Kill a process started with `start_new_session=True`, and everything it started."""
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    await proc.wait()


__all__ = ["STOP_GRACE", "Agent", "Event", "Prompt", "Session", "Tool", "ToolCall", "Turn", "TurnSession", "bind", "kill", "scratch"]
