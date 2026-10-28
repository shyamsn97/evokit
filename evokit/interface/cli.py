"""The `evokit` CLI: `evokit COMMAND ARGS... [--dir DIR] [--task TASK] [--format FORMAT]`, where a
command is one of `experiment_tools` (the experiment's commands, by name) or a built-in (init,
run, mcp, skill export). Arguments are parsed from the command's signature, the same way an
agent's `evokit` tools are; `evokit run` parses a recipe's options from the recipe's signature."""

from __future__ import annotations

import asyncio
import functools
import importlib
import importlib.util
import inspect
import json
import logging
import os
import shlex
import sys
import traceback
from pathlib import Path
from typing import Any, Callable

from evokit import __version__
from evokit.agents.tools import encode, parse_argv, tool_doc, tool_summary, usage
from evokit.errors import EvoError, UsageError
from evokit.experiment import Experiment, open_experiment, task_for
from evokit.interface.commands import COMMANDS, READ, experiment_tools
from evokit.task import init_task, read_config

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
        "next": f"edit the files, then `evokit status --task {name} --dir {name}/experiments/0`",
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


def run(recipe: str, task_dir: str | None = None) -> Any:
    """Run RECIPE (avo, autoresearch, dgm, ... or FILE.py:NAME) on the task in TASK_DIR, in --dir,
    continuing it if it exists. Options are the recipe's own (`--max-cost 5`, `--agent claude` or
    `--agent "COMMAND"`); config.yaml's `recipe:` section gives the recipe and its defaults, so
    `evokit run TASK_DIR` runs that. Returns the best individual."""


BUILTINS: dict[str, Callable] = {"init": init, "run": run, "skill export": skill_export}


def load_recipe(ref: str) -> Callable:
    """A recipe by name (evokit.recipes), as MODULE:NAME, or as FILE.py:NAME."""
    import evokit.recipes

    if ":" not in ref:
        found = getattr(evokit.recipes, ref.replace("-", "_"), None)
        if not inspect.iscoroutinefunction(found):
            raise UsageError(f"no recipe {ref!r}", hint="avo, autoresearch, dgm, hyperagents, openevolve, shinka, ga, or FILE.py:NAME")
        return found
    module_ref, _, name = ref.rpartition(":")
    if module_ref.endswith(".py"):
        path = Path(module_ref).resolve()
        if not path.is_file():
            raise UsageError(f"recipe file not found: {path}")
        spec = importlib.util.spec_from_file_location(f"_evokit_recipe_{abs(hash(str(path)))}", path)
        module = importlib.util.module_from_spec(spec)
        sys.path.insert(0, str(path.parent))
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(module_ref)
    found = getattr(module, name, None)
    if not callable(found):
        raise UsageError(f"{ref} is not a function")
    return found


def make_agent(spec: str):
    """`claude` (Claude Code), or a command run in the agent's directory with the prompt as {prompt}."""
    from evokit.agents import ClaudeCode, CommandAgent

    return ClaudeCode() if spec == "claude" else CommandAgent(shlex.split(spec))


def run_recipe(words: list[str], directory: str) -> Any:
    """`evokit run [RECIPE] TASK_DIR [--agent A] [--option value ...]`."""
    positional = []
    while words and not words[0].startswith("--"):
        positional.append(words.pop(0))
    if not 1 <= len(positional) <= 2:
        raise UsageError("usage: evokit run [RECIPE] TASK_DIR [--option value ...]", hint="e.g. evokit run avo examples/circle_packing --max-cost 5")
    task_dir = positional[-1]
    config = dict(read_config(task_dir).get("recipe") or {})
    name = positional[0] if len(positional) == 2 else config.pop("name", None)
    if len(positional) == 2:
        config.pop("name", None)
    if not name:
        raise UsageError(f"no recipe for {task_dir}", hint="name it (`evokit run avo TASK_DIR`), or set recipe: {name: avo} in config.yaml")
    recipe = load_recipe(name)
    params = list(inspect.signature(recipe).parameters.values())
    first, *rest = params
    agent_param = next((p for p in rest if p.name == "agent"), None)
    options = [p for p in rest if p is not agent_param and p.name != "dir" and p.kind is not p.VAR_KEYWORD]
    if agent_param is not None:
        options.append(inspect.Parameter("agent", inspect.Parameter.KEYWORD_ONLY, default="claude", annotation=str))
    parser = functools.wraps(recipe)(lambda **kw: kw)
    parser.__signature__ = inspect.Signature(options)
    defaults = [w for key, value in config.items() for w in (f"--{key.replace('_', '-')}", str(value))]
    kwargs = parse_argv(parser, [*defaults, *words], f"run {name}")
    if agent_param is not None:
        kwargs["agent"] = make_agent(kwargs.pop("agent", "claude"))
    if any(p.name == "dir" for p in rest):
        kwargs["dir"] = directory
    task = task_for(task_dir, directory)
    return asyncio.run(recipe(task, **kwargs))


def experiment_for(directory: str, task: str | None) -> Experiment:
    """The experiment in `directory`: an existing one (with `task`, or the task it was made from),
    or a new one on `task`, seeded with its files and kept by a BestArchive."""
    if (Path(directory) / "history" / "records.jsonl").is_file():
        return open_experiment(directory, task_for(task, directory) if task else None)
    if task is None:
        raise UsageError(f"no experiment in {Path(directory)}", hint="start one with --task TASK_DIR (create a task with `evokit init NAME`)")
    return task_for(task, directory).create_experiment(directory)


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
        lines.append(f"      {tool_summary(tool)}")
    lines.append(f"  {usage(lambda: None, 'mcp')}")
    lines.append("      Serve this experiment's commands as MCP tools over stdio.")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    tokens = list(sys.argv[1:] if argv is None else argv)
    try:
        options, words = split_options(tokens)
        directory, task, fmt = options["--dir"], options["--task"], options["--format"]
        if not words or words[0] in ("-h", "--help", "help"):
            tools = {name: functools.partial(fn, None) for name, fn in {**READ, **COMMANDS}.items()}
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
        if "-h" in args or "--help" in args:
            tool = BUILTINS.get(name) or {**READ, **COMMANDS}.get(name)
            if tool is None:
                raise UsageError(f"no command {name!r}", hint="see `evokit --help`")
            sys.stdout.write(f"Usage: {usage(tool, name)}\n\n{tool_doc(tool)}\n")
            return 0
        if name == "run":
            logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="evokit: %(message)s")
            emit(run_recipe(args, directory), fmt)
            return 0
        tool = BUILTINS.get(name)
        if tool is None:
            if name in READ:
                tool = experiment_tools(open_experiment(directory))[name]
            elif name in COMMANDS:
                tool = experiment_tools(experiment_for(directory, task))[name]
            else:
                raise UsageError(f"no command {name!r}", hint="see `evokit --help`")
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


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["emit", "load_recipe", "main", "run_recipe"]
