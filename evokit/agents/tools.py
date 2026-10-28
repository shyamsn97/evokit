"""Tools: plain functions an agent calls while it works, and how an agent in another process calls them.

A session's tools are the functions it was given (`experiment.start` gives an agent the
experiment's: read it, remember, score, submit). `ShellTools` serves them as an `evokit` command
on the agent's PATH (`evokit NAME ARGS...`), and ReplAgent's Python gets functions of the same
names. A tool's name, parameters and docstring are what the agent is told.
"""

from __future__ import annotations

import asyncio
import dataclasses
import datetime as _dt
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

from evokit.errors import UsageError
from evokit.records import Evaluation, Individual

logger = logging.getLogger("evokit")

Tool = Callable[..., Any]


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


# calling a tool from text arguments


def tool_name(tool: Tool) -> str:
    fn = getattr(tool, "func", tool)  # functools.partial
    return getattr(tool, "__name__", None) or fn.__name__


def tool_doc(tool: Tool) -> str:
    return inspect.getdoc(getattr(tool, "func", tool)) or ""


def tool_summary(tool: Tool) -> str:
    """The first paragraph of a tool's docstring, on one line."""
    return " ".join(tool_doc(tool).split("\n\n")[0].split())


def _hints(tool: Tool) -> dict[str, Any]:
    fn = inspect.unwrap(getattr(tool, "func", tool))
    fn = inspect.unwrap(getattr(fn, "func", fn))  # a wrapped functools.partial
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
    """Required parameters, and optional text ones that are not keyword-only, can be given without their --name."""
    if param.kind is param.KEYWORD_ONLY:
        return False
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
    """JSON-compatible data for a tool's or command's result."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Individual):
        return {
            "id": value.id,
            "name": value.name,
            "description": value.description,
            "genome": encode(value.genome),
            "parents": list(value.parents),
            "evaluation": encode(value.evaluation),
            "info": encode(dict(value.info)),
        }
    if isinstance(value, Evaluation):
        return value.to_dict()
    if hasattr(value, "describe"):
        return encode(value.describe())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (_dt.datetime, _dt.date)):
        return value.isoformat()
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: encode(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {str(k): encode(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [encode(v) for v in value]
    if hasattr(value, "to_dict"):
        return encode(value.to_dict())
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


class ShellTools:
    """A session's tools as an `evokit` command on PATH (`evokit NAME ARGS...`, `evokit help`), for an
    agent in another process. `env` puts the command on the agent's PATH and tells ReplAgent's
    Python where to find the tools (EVOKIT_TOOLS). Every call goes to `session.call`."""

    def __init__(self, session):
        self.session = session
        self._server: asyncio.AbstractServer | None = None
        self._dir = Path(tempfile.mkdtemp(prefix="evokit-tools-"))
        self.env: dict[str, str] = {}

    async def __aenter__(self) -> "ShellTools":
        self._server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        port = self._server.sockets[0].getsockname()[1]
        script = self._dir / "evokit"
        script.write_text(SCRIPT.format(python=sys.executable, port=port))
        script.chmod(0o755)
        tools = {"finish": "(**fields)"} | {name: str(inspect.signature(t)) for name, t in self.session.tools.items()}
        spec = {"port": port, "tools": tools}
        self.env = {"PATH": f"{self._dir}{os.pathsep}{os.environ.get('PATH', '')}", "EVOKIT_TOOLS": json.dumps(spec)}
        return self

    async def __aexit__(self, *exc) -> None:
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        shutil.rmtree(self._dir, ignore_errors=True)

    async def handle(self, request: Mapping[str, Any]) -> dict:
        """One request: {"tool", "argv"} from the command, or {"tool", "args", "kwargs"} from Python."""
        name = request.get("tool", "")
        if name == "help":
            return {"result": describe(list(self.session.tools.values()))}
        if request.get("argv") is not None:
            return await self.session.call_argv(name, request["argv"])
        return await self.session.call(name, request.get("kwargs") or {}, request.get("args") or ())

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            reply = await self.handle(json.loads(await reader.readline() or b"{}"))
            writer.write(json.dumps(reply, default=str).encode())
            await writer.drain()
        finally:
            writer.close()


__all__ = ["ShellTools", "Tool", "call", "describe", "encode", "limited", "parse_argv", "tool_doc", "tool_name", "tool_summary", "usage"]
