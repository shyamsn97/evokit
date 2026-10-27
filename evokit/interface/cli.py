"""The `evokit` CLI: `evokit COMMAND ARGS... [--dir DIR] [--task TASK] [--format FORMAT]`, where a
command is one of `run_tools` (the run's commands, by name) or a built-in (init, mcp, skill export).
Arguments are parsed from the command's signature, the same way an agent's `evokit` tools are."""

from __future__ import annotations

import inspect
import json
import logging
import os
import sys
import traceback
from typing import Any, Callable

from evokit import __version__
from evokit.agents.tools import parse_argv, tool_doc, usage
from evokit.errors import EvoError, UsageError
from evokit.interface.run import COMMANDS, READ, encode, run_tools
from evokit.task import init_task, open_history, open_run

DEFAULT_DIR = os.environ.get("EVOKIT_DIR", ".evokit")
DEFAULT_TASK = os.environ.get("EVOKIT_TASK")
FORMATS = ("auto", "json", "text")
GLOBAL = {"--dir": DEFAULT_DIR, "--task": DEFAULT_TASK, "--format": "auto"}


def emit(value: Any, fmt: str = "auto") -> None:
    data = encode(value)
    if fmt == "auto":
        fmt = "text" if sys.stdout.isatty() else "json"
    if fmt == "json":
        sys.stdout.write(json.dumps(data, separators=(",", ":"), default=str) + "\n")
    elif isinstance(data, str):
        sys.stdout.write(data if data.endswith("\n") else data + "\n")
    else:
        sys.stdout.write(json.dumps(data, indent=2, default=str) + "\n")
    sys.stdout.flush()


def emit_error(error: EvoError) -> None:
    sys.stderr.write(json.dumps(error.to_dict(), default=str) + "\n")
    sys.stderr.flush()


# built-in commands


def init(name: str) -> dict:
    """Create a task folder NAME from the template (TASK.md, config.yaml, evaluate.py, main.py) that runs as it is."""
    directory = init_task(name)
    return {
        "task": str(directory),
        "files": sorted(p.name for p in directory.iterdir()),
        "next": f"edit the files, then `evokit status --task {name} --dir {name}/runs/0`",
    }


def skill_export(target: str = "claude", to: str = ".", stdout: bool = False) -> Any:
    """Write the evokit skill (SKILL.md for claude or cursor, an AGENTS.md section for codex) into a project."""
    from evokit.interface.skill import export_skill, render_skill

    if target not in ("claude", "cursor", "codex"):
        raise UsageError(f"unknown target {target!r}", hint="claude, cursor or codex")
    if stdout:
        sys.stdout.write(render_skill(None, target))
        return None
    return {"written": str(export_skill(target, to))}


BUILTINS: dict[str, Callable] = {"init": init, "skill export": skill_export}


def split_options(tokens: list[str]) -> tuple[dict[str, str | None], list[str]]:
    """The global options (--dir, --task, --format), wherever they are, and the other words."""
    options: dict[str, str | None] = dict(GLOBAL)
    rest, words = [], list(tokens)
    while words:
        word = words.pop(0)
        flag, has_value, value = word.partition("=")
        if flag in GLOBAL:
            if not has_value:
                if not words:
                    raise UsageError(f"{flag} needs a value")
                value = words.pop(0)
            options[flag] = value
        else:
            rest.append(word)
    if options["--format"] not in FORMATS:
        raise UsageError(f"unknown format {options['--format']!r}", hint=", ".join(FORMATS))
    return options, rest


def commands(directory: str, task: str | None, name: str | None) -> dict[str, Callable]:
    """The commands for `name`: the read commands open any run directory; the others open the run
    with its task (creating and seeding it from `task` if it is new)."""
    if name in READ:
        return run_tools(open_history(directory))
    return run_tools(open_run(directory, task))


def help_text(tools: dict[str, Callable]) -> str:
    lines = [
        "Usage: evokit COMMAND [ARGS]... [--dir DIR] [--task TASK_DIR] [--format auto|json|text]",
        "",
        "Evolutionary search where the variation operator can be a coding agent.",
        "",
        "Commands:",
    ]
    for name, tool in {**BUILTINS, **tools}.items():
        lines.append(f"  {usage(tool, name)}")
        lines.append(f"      {tool_doc(tool).splitlines()[0] if tool_doc(tool) else ''}")
    lines.append(f"  {usage(lambda: None, 'mcp')}")
    lines.append("      Serve this run's commands as MCP tools over stdio.")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    tokens = list(sys.argv[1:] if argv is None else argv)
    try:
        options, words = split_options(tokens)
        directory, task, fmt = options["--dir"], options["--task"], options["--format"]
        if not words or words[0] in ("-h", "--help", "help"):
            try:
                tools = commands(directory, None, None)
            except EvoError:
                tools = {name: functools_partial_none(fn) for name, fn in {**READ, **COMMANDS}.items()}
            sys.stdout.write(help_text(tools))
            return 0
        if words[0] == "--version":
            sys.stdout.write(f"evokit {__version__}\n")
            return 0
        name, args = words[0], words[1:]
        if name == "skill" and args:
            name, args = f"skill {args[0]}", args[1:]
        if name == "mcp":
            from evokit.interface.mcp_server import serve

            serve(directory, task)
            return 0
        if name == "run":
            logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="evokit: %(message)s")
        tool = BUILTINS.get(name)
        if tool is None:
            if name not in READ and name not in COMMANDS:
                tool = commands(directory, task, name).get(name)
                if tool is None:
                    raise UsageError(f"no command {name!r}", hint="see `evokit --help`")
            else:
                tool = commands(directory, task, name)[name]
        if "-h" in args or "--help" in args:
            sys.stdout.write(f"Usage: {usage(tool, name)}\n\n{tool_doc(tool)}\n")
            return 0
        result = tool(**parse_argv(tool, args, name))
        if result is not None or name != "skill export":
            emit(result, fmt)
    except EvoError as exc:
        emit_error(exc)
        return exc.code
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        emit_error(EvoError(f"{type(exc).__name__}: {exc}", hint=traceback.format_exc(limit=5)))
        return 1
    return 0


def functools_partial_none(fn: Callable) -> Callable:
    """`fn` with its first parameter (the run) left out, for help without a run."""
    import functools

    return functools.partial(fn, None)


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["main", "emit", "inspect"]
