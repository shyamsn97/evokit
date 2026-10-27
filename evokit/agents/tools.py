"""Tools: plain functions an agent calls while it works, and how an agent in another process calls them.

`agent_tools(...)` is the one place the standard tools are listed: read the population and the
run, score a version, and (for an agent that decides what is kept) submit one. `CodingAgent.run`
takes any list of functions as `tools=` and serves them through a `ToolServer`: the agent's shell
gets an `evokit` command (`evokit NAME ARGS...`), and ReplAgent's Python gets functions of the
same names. A tool's name, parameters and docstring are what the agent is told.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import json
import logging
import os
import shutil
import sys
import tempfile
import typing
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from evokit.errors import EvoError, UsageError
from evokit.loop import as_eval_fn, submit as submit_individual
from evokit.records import Evaluation, Individual
from evokit.util import NAME_LIMIT, title_and_body, truncate

logger = logging.getLogger("evokit")

Tool = Callable[..., Any]


def best(archive) -> dict | None:
    """The best individual in the population: its id, name and scores."""
    found = archive.best()
    return _summary(found) if found else None


def members(archive, top: int | None = None) -> list[dict]:
    """The population, best first: each member's id, name and scores."""
    return [_summary(m) for m in archive.members()[:top]]


def log(history, id: str | None = None, all: bool = False, limit: int | None = 20) -> list[dict]:
    """ID and its ancestors, nearest first, with names and scores, like `git log`; without ID, from
    the last individual the archive admitted. With --all, every evaluated individual, newest first,
    rejected ones too; `member` says whether it is still in the archive."""
    return history.log(history.lookup(id) if id else None, all=all, limit=limit)


def show(history, id: str, path: str | None = None) -> Any:
    """An individual's name, description, parents, evaluation and files; with PATH, that file."""
    return history.show(history.lookup(id), path)


def diff(history, a: str, b: str | None = None) -> str:
    """What changed from A to B, as a unified diff; with only A, from its parent to A."""
    return history.diff(history.lookup(a), history.lookup(b) if b else None)


def grep(history, pattern: str, limit: int = 20) -> list[dict]:
    """Individuals whose name, description or files match a regex, rejected ones too, so you can
    see what was already tried and how it scored."""
    return history.grep(pattern, limit)


READ_TOOLS = (log, show, diff, grep)
"""The tools that read a run's history, like git; each takes the History first."""


def agent_tools(
    archive,
    history=None,
    eval_fn=None,
    *,
    project=None,
    workspace: Path | None = None,
    parents: Callable[[], list[str]] | None = None,
    submit: bool = False,
    info: Mapping[str, Any] | None = None,
) -> list[Tool]:
    """The tools every agent gets.

    Always: `best` and `members` (the population). With a `history`: `log`, `show`, `diff` and
    `grep` (the run). With an `eval_fn`: `score`, which scores a version and keeps nothing (its
    cost is noted in the history). With `submit=True` (an agent that decides what is kept, like
    AVO's): `submit`, which scores a version and offers it to the archive.

    With a `project` and `workspace`, the version scored or submitted is the workspace's files,
    with `parents()` as its parents (default: the archive's best); otherwise the agent passes the
    genome itself, as JSON. `info` goes into every submitted individual's info.
    """
    files = project is not None and workspace is not None
    default_parents = parents or (lambda: [archive.best().id] if archive.best() else [])
    tools: list[Tool] = [functools.partial(best, archive), functools.partial(members, archive)]
    if history is not None:
        tools += [functools.partial(tool, history) for tool in READ_TOOLS]

    if eval_fn is not None:
        score_fn = as_eval_fn(eval_fn)

        async def _score(genome) -> dict:
            evaluation = await score_fn(Individual.new(genome, default_parents(), {"operator": "draft"}))
            if history is not None:
                history.note("draft", _scores(evaluation), {**evaluation.usage, "draft_evals": 1})
            return _evaluation(evaluation)

        if files:

            async def score() -> dict:
                """Score the files in your working directory as they are now, without keeping them."""
                return await _score(await asyncio.to_thread(project.read, workspace))

        else:

            async def score(genome: Any) -> dict:
                """Score a genome (as JSON) without keeping it."""
                return await _score(genome)

        tools.append(score)

    if submit:
        if history is None or eval_fn is None:
            raise ValueError("submit needs a history and an eval_fn")

        async def _submit(genome, parent_ids, name, description) -> dict:
            if len(name) > NAME_LIMIT or "\n" in name:
                name, rest = title_and_body(name)
                description = "\n\n".join(p for p in (rest, description) if p)
            description = truncate(description, 4000)
            child = Individual.new(genome, parent_ids, dict(info or {}), name=name, description=description)
            result = await submit_individual(history, archive, eval_fn, child)
            return {"id": result.id, "kept": result.info["admitted"], **_evaluation(result.evaluation)}

        if files:

            async def submit(name: str, description: str = "") -> dict:
                """Score the files in your working directory and offer them to the population: `kept` says
                whether they were kept. `name` is one line saying what this version is, `description` what
                changed and why."""
                genome = await asyncio.to_thread(project.read, workspace)
                best = archive.best()
                if best is not None and genome == best.genome:
                    raise UsageError("nothing changed since the best version", hint="edit the files first")
                return await _submit(genome, default_parents(), name, description)

        else:

            async def submit(genome: Any, name: str, description: str = "", parents: list[str] | None = None) -> dict:
                """Score a genome (as JSON) and offer it to the population: `kept` says whether it was kept."""
                return await _submit(genome, parents if parents is not None else default_parents(), name, description)

        tools.append(submit)

    return tools


def limited(tool: Tool, n: int) -> Tool:
    """`tool`, refusing every call after the first `n`."""
    calls = 0

    @functools.wraps(tool)
    async def limited_tool(*args, **kwargs):
        nonlocal calls
        if calls >= n:
            raise UsageError(f"{tool_name(tool)} can be called {n} times and was; finish and summarize")
        calls += 1
        return await call(tool, inspect.signature(tool).bind(*args, **kwargs).arguments)

    return limited_tool


def _summary(individual: Individual) -> dict:
    out: dict[str, Any] = {"id": individual.id, "name": individual.name, "parents": list(individual.parents)}
    if individual.evaluation is not None:
        out["objectives"] = dict(individual.evaluation.objectives)
    return out


def _evaluation(evaluation: Evaluation) -> dict:
    return {"objectives": dict(evaluation.objectives), "feedback": truncate(evaluation.feedback, 4000)}


def _scores(evaluation: Evaluation) -> str:
    return ", ".join(f"{k}={v:.6g}" for k, v in evaluation.objectives.items()) or "failed"


# calling a tool from text arguments


def tool_name(tool: Tool) -> str:
    fn = getattr(tool, "func", tool)  # functools.partial
    return getattr(tool, "__name__", None) or fn.__name__


def tool_doc(tool: Tool) -> str:
    return inspect.getdoc(getattr(tool, "func", tool)) or ""


def _hints(tool: Tool) -> dict[str, Any]:
    fn = inspect.unwrap(getattr(tool, "func", tool))
    try:
        return typing.get_type_hints(fn)
    except Exception:
        return {}


def parameters(tool: Tool) -> list[inspect.Parameter]:
    return [p for p in inspect.signature(tool).parameters.values() if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)]


def _is(hint: Any, kind: type) -> bool:
    return hint is kind or kind in typing.get_args(hint)


def _coerce(value: str, hint: Any) -> Any:
    if _is(hint, bool):
        return value.lower() in ("1", "true", "yes", "y")
    if _is(hint, int):
        return int(value)
    if _is(hint, float):
        return float(value)
    if hint is str or _is(hint, str) or hint is inspect.Parameter.empty:
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _is_list(hint: Any) -> bool:
    return typing.get_origin(hint) is list or list in [typing.get_origin(a) for a in typing.get_args(hint)]


def _positional(param: inspect.Parameter, hint: Any) -> bool:
    """Required parameters, and optional text ones, can be given without their --name."""
    return param.default is param.empty or (_is(hint, str) and not _is_list(hint))


def _flag(param: inspect.Parameter, hint: Any) -> str:
    name = param.name.replace("_", "-")
    return name[:-1] if _is_list(hint) and name.endswith("s") else name  # --parent ID, once per parent


def parse_argv(tool: Tool, argv: Sequence[str], name: str | None = None) -> dict[str, Any]:
    """Keyword arguments for `tool` from command-line words: required and text parameters can be
    given in order; `--name value` (or `--name=value`) sets any by name, `--flag` a boolean, and a
    list parameter takes its singular `--name value` once per item."""
    params, hints = parameters(tool), _hints(tool)
    by_flag = {p.name.replace("_", "-"): p for p in params} | {_flag(p, hints.get(p.name)): p for p in params}
    positional = [p for p in params if _positional(p, hints.get(p.name))]
    kwargs: dict[str, Any] = {}
    words = list(argv)
    while words:
        word = words.pop(0)
        if word.startswith("--") and len(word) > 2:
            flag, has_value, inline = word[2:].partition("=")
            param = by_flag.get(flag)
            if param is None:
                raise UsageError(f"unknown option --{flag}", hint=usage(tool, name))
            hint = hints.get(param.name)
            if _is(hint, bool) and not has_value:
                kwargs[param.name] = True
                continue
            if not has_value and not words:
                raise UsageError(f"--{flag} needs a value", hint=usage(tool, name))
            value = inline if has_value else words.pop(0)
            if _is_list(hint):
                kwargs.setdefault(param.name, []).append(value)
            else:
                kwargs[param.name] = _coerce(value, hint)
            continue
        free = [p for p in positional if p.name not in kwargs]
        if not free:
            raise UsageError(f"too many arguments: {word!r}", hint=usage(tool, name))
        kwargs[free[0].name] = _coerce(word, hints.get(free[0].name))
    missing = [p.name for p in params if p.default is p.empty and p.name not in kwargs]
    if missing:
        raise UsageError(f"missing {', '.join(missing)}", hint=usage(tool, name))
    return kwargs


def usage(tool: Tool, name: str | None = None, command: str = "evokit") -> str:
    """How to call `tool` from the shell, e.g. `evokit show ID [PATH]`."""
    words = [command, name or tool_name(tool)]
    hints = _hints(tool)
    for p in parameters(tool):
        hint, word = hints.get(p.name), p.name.upper()
        if _is(hint, bool):
            words.append(f"[--{_flag(p, hint)}]")
        elif _is_list(hint):
            words.append(f"[--{_flag(p, hint)} {word[:-1] if word.endswith('S') else word} ...]")
        elif p.default is p.empty:
            words.append(word)
        elif _positional(p, hint):
            words.append(f"[{word}]")
        else:
            words.append(f"[--{_flag(p, hint)} {word}]")
    return " ".join(words)


def describe(tools: Sequence[Tool], style: str = "shell") -> str:
    """What the agent is told about its tools: one line each, as shell commands or Python functions."""
    lines = []
    for tool in tools:
        doc = " ".join(tool_doc(tool).split())
        if style == "python":
            signature = str(inspect.signature(tool)).replace("'", "")
            lines.append(f"- `{tool_name(tool)}{signature}`: {doc}")
        else:
            lines.append(f"- `{usage(tool)}`: {doc}")
    if style == "python":
        head = "These functions are defined in your Python session; each returns its result, or raises RuntimeError with the reason:"
    else:
        head = "You can run these commands; each prints its result as JSON (or text), and exits 1 on an error:"
    return head + "\n" + "\n".join(lines)


def encode(value: Any) -> Any:
    """JSON-compatible data for a tool's result."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Individual):
        return {**_summary(value), "description": value.description}
    if isinstance(value, Evaluation):
        return _evaluation(value)
    if isinstance(value, Mapping):
        return {str(k): encode(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [encode(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return repr(value)


async def call(tool: Tool, kwargs: Mapping[str, Any]) -> Any:
    result = tool(**kwargs)
    if inspect.isawaitable(result):
        result = await result
    return result


# serving tools to an agent in another process

SCRIPT = """#!{python}
import json, os, socket, sys
args = sys.argv[1:]
request = {{"tool": args[0] if args else "help", "argv": args[1:]}}
s = socket.create_connection(("127.0.0.1", {port}))
s.sendall((json.dumps(request) + "\\n").encode())
reply = json.loads(b"".join(iter(lambda: s.recv(65536), b"")).decode() or "{{}}")
if "error" in reply:
    print(json.dumps(reply, indent=2), file=sys.stderr)
    sys.exit(1)
result = reply.pop("result", None)
if isinstance(result, dict):
    print(json.dumps({{**result, **reply}}, indent=2))
else:
    print(result if isinstance(result, str) else json.dumps(result, indent=2))
    for key, value in reply.items():
        print(f"[{{key}}] {{value}}")
"""


class ToolServer:
    """`tools` on a localhost socket, for an agent running in another process.

    `env` puts an `evokit` command on the agent's PATH (`evokit NAME ARGS...`, `evokit help`) and
    tells ReplAgent's Python where to find them. `stop()` returning a reason refuses every call and
    is added to every reply as `stop`; `messages()` returning text adds it to the next reply as
    `message` (e.g. a supervisor's directions).
    """

    def __init__(
        self,
        tools: Sequence[Tool],
        stop: Callable[[], str | None] | None = None,
        messages: Callable[[], str | None] | None = None,
    ):
        self.tools = {tool_name(t): t for t in tools}
        self.stop, self.messages = stop, messages
        self._server: asyncio.AbstractServer | None = None
        self._dir = Path(tempfile.mkdtemp(prefix="evokit-tools-"))
        self.env: dict[str, str] = {}

    async def __aenter__(self) -> "ToolServer":
        self._server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        port = self._server.sockets[0].getsockname()[1]
        script = self._dir / "evokit"
        script.write_text(SCRIPT.format(python=sys.executable, port=port))
        script.chmod(0o755)
        spec = {"port": port, "tools": {name: str(inspect.signature(t)) for name, t in self.tools.items()}}
        self.env = {"PATH": f"{self._dir}{os.pathsep}{os.environ.get('PATH', '')}", "EVOKIT_TOOLS": json.dumps(spec)}
        return self

    async def __aexit__(self, *exc) -> None:
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        shutil.rmtree(self._dir, ignore_errors=True)

    async def handle(self, name: str, argv: Sequence[str] | None = None, args=(), kwargs=None) -> dict:
        """One call: by command-line words (`argv`) or by Python arguments."""
        if name == "help":
            return {"result": describe(list(self.tools.values()))}
        tool = self.tools.get(name)
        if tool is None:
            return {"error": f"unknown tool {name!r}", "hint": f"tools: {', '.join(self.tools)}"}
        reason = self.stop() if self.stop else None
        if reason:
            return {"error": f"the run is over ({reason}); stop now", "stop": reason}
        try:
            if argv is not None:
                kwargs = parse_argv(tool, argv)
            else:
                try:
                    kwargs = dict(inspect.signature(tool).bind(*args, **(kwargs or {})).arguments)
                except TypeError as exc:
                    raise UsageError(str(exc), hint=f"{name}{inspect.signature(tool)}") from None
            reply: dict[str, Any] = {"result": encode(await call(tool, kwargs))}
        except EvoError as exc:
            reply = {"error": exc.message, **({"hint": exc.hint} if exc.hint else {})}
        except Exception as exc:
            logger.exception("tool %s failed", name)
            reply = {"error": f"{type(exc).__name__}: {exc}"}
        if self.messages and (message := self.messages()):
            reply["message"] = message
        if self.stop and (reason := self.stop()):
            reply["stop"] = reason
        return reply

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            request = json.loads(await reader.readline() or b"{}")
            reply = await self.handle(
                request.get("tool", ""), request.get("argv"), request.get("args") or (), request.get("kwargs")
            )
            writer.write(json.dumps(reply, default=str).encode())
            await writer.drain()
        finally:
            writer.close()


__all__ = ["READ_TOOLS", "Tool", "ToolServer", "agent_tools", "describe", "limited", "parse_argv", "tool_doc", "tool_name", "usage"]
