"""`evokit mcp`: the experiment's commands (`experiment_tools`) served as MCP tools over stdio."""

from __future__ import annotations

import inspect
import json
import typing
from typing import Any, Callable

from evokit.agents.tools import parameters, tool_doc
from evokit.errors import EvoError, UsageError
from evokit.agents.tools import encode
from evokit.interface.commands import experiment_tools


def _tool_error() -> type[Exception]:
    try:
        from mcp.server.mcpserver.exceptions import ToolError
    except ImportError:
        try:
            from mcp.server.fastmcp.exceptions import ToolError
        except ImportError:
            return ValueError
    return ToolError


def _mcp_tool(name: str, fn: Callable, release: Callable[[], None]) -> Callable:
    """`fn` (a command bound to its experiment) as a plain function MCP can describe: keyword
    arguments, JSON results, errors as ToolError. `release` gives the experiment back after each
    call, so the server never keeps other processes from driving it."""
    tool_error = _tool_error()
    params = [p.replace(kind=inspect.Parameter.KEYWORD_ONLY) for p in parameters(fn)]
    try:
        hints = typing.get_type_hints(inspect.unwrap(getattr(fn, "func", fn)))
    except Exception:
        hints = {}

    def tool(**kwargs) -> Any:
        try:
            return encode(fn(**kwargs))
        except EvoError as exc:
            raise tool_error(json.dumps(exc.to_dict())) from exc
        finally:
            release()

    tool.__signature__ = inspect.Signature([p.replace(annotation=hints.get(p.name, p.annotation)) for p in params])
    tool.__annotations__ = {p.name: hints.get(p.name, p.annotation) for p in params}
    tool.__name__ = name.replace("-", "_")
    tool.__doc__ = tool_doc(fn)
    return tool


def build_server(directory: str, task: str | None = None):
    try:
        from mcp.server.mcpserver import MCPServer as FastMCP  # mcp >= 2
    except ImportError:
        try:
            from mcp.server.fastmcp import FastMCP  # mcp 1.x
        except ImportError as exc:
            raise UsageError("the MCP server needs the `mcp` package", hint="pip install 'evokit[mcp]'") from exc
    from evokit.interface.cli import experiment_for
    from evokit.interface.skill import render_skill

    experiment = experiment_for(directory, task)
    server = FastMCP("evokit", instructions=render_skill(experiment, "mcp"))
    for name, fn in experiment_tools(experiment).items():
        tool = _mcp_tool(name, fn, experiment.close)
        server.add_tool(tool, name=tool.__name__, description=tool.__doc__)
    return server


def serve(directory: str, task: str | None = None) -> None:
    build_server(directory, task).run()
