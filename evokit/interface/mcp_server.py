"""`evokit mcp`: the same @command registry served as MCP tools over stdio."""

from __future__ import annotations

import inspect
import json
from typing import TYPE_CHECKING, Any, Callable

from evokit.errors import EvoError, UsageError
from evokit.interface.commands import Command, commands_of, component_commands

if TYPE_CHECKING:
    from evokit.interface.setup import Setup


def _tool_error() -> type[Exception]:
    try:
        from mcp.server.mcpserver.exceptions import ToolError
    except ImportError:
        try:
            from mcp.server.fastmcp.exceptions import ToolError
        except ImportError:
            return ValueError
    return ToolError


def _tool(cmd: Command, setup: "Setup") -> Callable | None:
    from evokit.interface.cli import cli_params, resolve_arguments

    params = cli_params(cmd.func)
    if params is None:
        return None
    tool_error = _tool_error()

    def tool(**kwargs) -> Any:
        try:
            return cmd.shape(cmd.func(**resolve_arguments(cmd.func, setup, kwargs)))
        except EvoError as exc:
            raise tool_error(json.dumps(exc.to_dict())) from exc

    tool.__signature__ = inspect.Signature([p.replace(kind=inspect.Parameter.KEYWORD_ONLY) for p in params])
    tool.__annotations__ = {p.name: p.annotation for p in params}
    tool.__name__ = cmd.name.replace("-", "_")
    tool.__doc__ = cmd.doc
    return tool


def build_server(directory: str, task: str | None = None):
    try:
        from mcp.server.mcpserver import MCPServer as FastMCP  # mcp >= 2
    except ImportError:
        try:
            from mcp.server.fastmcp import FastMCP  # mcp 1.x
        except ImportError as exc:
            raise UsageError("the MCP server needs the `mcp` package", hint="pip install 'evokit[mcp]'") from exc
    from evokit.interface.setup import open_setup
    from evokit.interface.skill import render_skill

    setup = open_setup(directory, task)
    server = FastMCP("evokit", instructions=render_skill(setup, "mcp"))
    cmds = [(c.name.replace("-", "_"), c) for obj in setup.objects for c in commands_of(obj, "mcp")]
    for slot, extra in component_commands(setup.objects, "mcp").items():
        cmds += [(f"{slot}_{c.name.replace('-', '_')}", c) for c in extra]
    for name, cmd in cmds:
        tool = _tool(cmd, setup)
        if tool is not None:
            server.add_tool(tool, name=name, description=cmd.doc)
    return server


def serve(directory: str, task: str | None = None) -> None:
    build_server(directory, task).run()
