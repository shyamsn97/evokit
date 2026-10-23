"""The `evokit` CLI (click), generated from the @command registry on RunCommands and
FileCommands (and on the archive, strategy, operator, and eval function)."""

from __future__ import annotations

import inspect
import json
import logging
import os
import sys
import traceback
import types
import typing
from typing import Any, Callable, Literal

import click

from evokit import __version__
from evokit.errors import EvoError, UsageError
from evokit.files.commands import FileCommands
from evokit.interface.commands import Command, component_commands, encode, injected_arguments
from evokit.interface.history import HistoryCommands
from evokit.interface.run import RunCommands
from evokit.interface.setup import Setup, open_history, open_setup
from evokit.records import Individual
from evokit.task import init_task

DEFAULT_DIR = os.environ.get("EVOKIT_DIR", ".evokit")
DEFAULT_TASK = os.environ.get("EVOKIT_TASK")
FORMATS = ("auto", "json", "text")
CLI_NAMES = {"parents": "--parent"}
OWNERS: tuple[type, ...] = (RunCommands, FileCommands, HistoryCommands)
_HINTS = {"Individual": Individual, "RunCommands": RunCommands}


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


def hints(fn: Callable) -> dict[str, Any]:
    return typing.get_type_hints(fn, localns=_HINTS)


def _is_individual(annotation: Any) -> bool:
    return _unwrap_optional(annotation) is Individual


def _cli_annotation(annotation: Any) -> Any | None:
    """Map a Python API type to something a command line can carry; None means unsupported.

    An Individual parameter is passed as its id.
    """
    if annotation is Individual:
        return str
    if _is_individual(annotation):
        return str | None
    if annotation is Any or typing.get_origin(annotation) in (dict, typing.Mapping) or "Mapping" in repr(annotation):
        return None
    return annotation


def _injected(annotation: Any) -> bool:
    return annotation is RunCommands


def cli_params(fn: Callable) -> list[inspect.Parameter] | None:
    """The parameters a caller supplies: `self` and injected ones (the RunCommands) are left out,
    and so are optional ones a command line cannot carry (an executor)."""
    types = hints(fn)
    out = []
    for param in inspect.signature(fn).parameters.values():
        if param.name == "self" or _injected(types.get(param.name)):
            continue
        if "Executor" in repr(types.get(param.name)) and param.default is not inspect.Parameter.empty:
            continue
        annotation = _cli_annotation(types.get(param.name, str))
        if annotation is None:
            return None
        out.append(param.replace(annotation=annotation))
    return out


def resolve_arguments(fn: Callable, setup: Setup, kwargs: dict) -> dict:
    """Inject the RunCommands, and replace individual ids with the Individuals they name."""
    types = hints(fn)
    injected = injected_arguments(fn, setup.run)
    out = {}
    for name in inspect.signature(fn).parameters:
        if name in injected:
            out[name] = injected[name]
        elif name in kwargs:
            value = kwargs[name]
            out[name] = setup.history.lookup(value) if _is_individual(types.get(name)) and isinstance(value, str) else value
    return out


# signatures -> click parameters


def _unwrap_optional(annotation: Any) -> Any:
    args = [a for a in typing.get_args(annotation) if a is not type(None)]
    if typing.get_origin(annotation) in (typing.Union, types.UnionType) and len(args) == 1:
        return args[0]
    return annotation


def _click_type(annotation: Any) -> Any:
    if typing.get_origin(annotation) is Literal:
        return click.Choice([str(a) for a in typing.get_args(annotation)])
    return annotation if annotation in (int, float, str) else str


def positional(fn: Callable, param: inspect.Parameter) -> bool:
    """Required parameters, individuals (by id), and those the command names are positional."""
    meta = getattr(fn, "__evokit_command__", None) or {}
    return (
        param.default is inspect.Parameter.empty
        or _is_individual(hints(fn).get(param.name))
        or param.name in meta.get("positional", ())
    )


def _click_param(param: inspect.Parameter, is_positional: bool = False) -> click.Parameter:
    annotation = _unwrap_optional(param.annotation)
    flag = CLI_NAMES.get(param.name, "--" + param.name.replace("_", "-"))
    if param.default is inspect.Parameter.empty:
        return click.Argument([param.name], type=_click_type(annotation))
    if is_positional:
        return click.Argument([param.name], type=_click_type(annotation), required=False, default=param.default)
    if annotation is bool:
        return click.Option([flag, param.name], is_flag=True, default=bool(param.default))
    if typing.get_origin(annotation) is list:
        (inner,) = typing.get_args(annotation) or (str,)
        return click.Option([flag, param.name], type=_click_type(inner), multiple=True)
    return click.Option([flag, param.name], type=_click_type(annotation), default=param.default, show_default=param.default is not None)


def _global_options() -> list[click.Option]:
    return [
        click.Option(["--dir"], default=DEFAULT_DIR, show_default=True, help="run directory ($EVOKIT_DIR)"),
        click.Option(["--task"], default=DEFAULT_TASK, help="task folder; needed only when --dir is a new run ($EVOKIT_TASK)"),
        click.Option(["--format"], type=click.Choice(FORMATS), default="auto", show_default=True, help="output format"),
    ]


def _make(
    name: str,
    fn: Callable,
    params: list[inspect.Parameter],
    call: Callable[[Setup, dict], Any],
    opener: Callable[[str, str | None], Setup] = open_setup,
) -> click.Command:
    def callback(dir: str, task: str | None, format: str, **kwargs):
        for param in params:
            value = kwargs.get(param.name)
            if isinstance(value, tuple):
                kwargs[param.name] = list(value) if value else param.default
        emit(call(opener(dir, task), kwargs), format)

    doc = inspect.getdoc(fn) or ""
    click_params = [_click_param(p, positional(fn, p)) for p in params]
    summary = doc.splitlines()[0] if doc else ""
    return click.Command(name, callback=callback, params=[*click_params, *_global_options()], help=doc, short_help=summary)


def owner_of(setup: Setup, cls: type) -> Any:
    owner = setup.find(cls)
    if owner is None:
        raise UsageError(
            f"this run has no {cls.__name__}",
            hint=f"return a {cls.__name__} from the task's factory to enable these commands",
        )
    return owner


def class_commands(surface: str) -> list[tuple[str, dict, type, Callable]]:
    """(name, meta, owner class, function) for every command on the owner classes."""
    out = []
    for cls in OWNERS:
        for attr in dir(cls):
            fn = getattr(cls, attr)
            meta = getattr(fn, "__evokit_command__", None)
            if meta and surface in meta["surfaces"]:
                out.append((meta["name"], meta, cls, fn))
    return sorted(out, key=lambda c: c[0])


def _owner_command(name: str, meta: dict, cls: type, fn: Callable) -> click.Command | None:
    params = cli_params(fn)
    if params is None:
        return None

    def call(setup, kwargs):
        owner = owner_of(setup, cls)
        value = getattr(owner, fn.__name__)(**resolve_arguments(fn, setup, kwargs))
        return meta["result"](owner, value) if meta["result"] else value

    if cls is HistoryCommands:
        return _make(name, fn, params, call, lambda dir, task: open_history(dir))
    return _make(name, fn, params, call)


def _component_command(cmd: Command, setup: Setup) -> click.Command | None:
    params = cli_params(cmd.func)
    if params is None:
        return None
    return _make(cmd.name, cmd.func, params, lambda _, kwargs: cmd.shape(cmd.func(**resolve_arguments(cmd.func, setup, kwargs))))


# built-in commands


def _dir_option(f):
    return click.option("--dir", default=DEFAULT_DIR, show_default=True, help="run directory ($EVOKIT_DIR)")(f)


def _format_option(f):
    return click.option("--format", type=click.Choice(FORMATS), default="auto", show_default=True, help="output format")(f)


def _task_option(f):
    return click.option("--task", default=DEFAULT_TASK, help="task folder; needed only when --dir is a new run ($EVOKIT_TASK)")(f)


@click.command()
@click.argument("name")
@_format_option
def init(name: str, format: str):
    """Create a task folder NAME from the template (TASK.md, config.yaml, evaluate.py, main.py) that runs as it is."""
    directory = init_task(name)
    emit(
        {
            "task": str(directory),
            "files": sorted(p.name for p in directory.iterdir()),
            "next": f"edit the files, then `evokit status --task {name} --dir {name}/runs/0`",
        },
        format,
    )


@click.command()
@_dir_option
@_task_option
def mcp(dir: str, task: str | None):
    """Serve this run's commands as MCP tools over stdio."""
    from evokit.interface.mcp_server import serve

    serve(dir, task)


@click.group()
def skill():
    """Agent skill files."""


@skill.command()
@click.argument("target", type=click.Choice(["claude", "cursor", "codex"]), default="claude")
@click.option("--to", default=".", show_default=True, help="project root to write into")
@click.option("--stdout", "to_stdout", is_flag=True, help="print instead of writing")
@_format_option
def export(target: str, to: str, to_stdout: bool, format: str):
    """Write the evokit skill (SKILL.md, or an AGENTS.md section for codex) into a project."""
    from evokit.interface.skill import export_skill, render_skill

    if to_stdout:
        sys.stdout.write(render_skill(None, target))
        return
    emit({"written": str(export_skill(target, to))}, format)


def try_open(directory: str, task: str | None = None) -> Setup | None:
    try:
        return open_setup(directory, task)
    except Exception:
        return None


def _prescan(tokens: list[str], option: str, default: str | None) -> str | None:
    for i, token in enumerate(tokens):
        if token == option and i + 1 < len(tokens):
            return tokens[i + 1]
        if token.startswith(option + "="):
            return token.split("=", 1)[1]
    return default


def build_app(tokens: list[str] | None = None) -> click.Group:
    app = click.Group("evokit", help="Evolutionary search where the variation operator can be a coding agent.")
    click.version_option(__version__, prog_name="evokit")(app)
    app.add_command(init)
    for name, meta, cls, fn in class_commands("cli"):
        command = _owner_command(name, meta, cls, fn)
        if command is not None:
            app.add_command(command)
    app.add_command(mcp)
    app.add_command(skill)

    tokens = tokens or []
    component = bool(tokens) and not tokens[0].startswith("-") and tokens[0] not in app.commands
    # only a component command may create the run here; --help and built-in commands open existing runs
    setup = try_open(_prescan(tokens, "--dir", DEFAULT_DIR), _prescan(tokens, "--task", DEFAULT_TASK) if component else None)
    if setup is not None:
        for slot, cmds in component_commands(setup.objects, "cli").items():
            group = click.Group(slot, help=f"Commands contributed by the run's {slot}.")
            for cmd in cmds:
                command = _component_command(cmd, setup)
                if command is not None:
                    group.add_command(command)
            app.add_command(group)
    return app


def main(argv: list[str] | None = None) -> int:
    tokens = list(sys.argv[1:] if argv is None else argv)
    if tokens and tokens[0] == "run":
        logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="evokit: %(message)s")
    try:
        result = build_app(tokens).main(tokens, prog_name="evokit", standalone_mode=False)
    except click.ClickException as exc:
        emit_error(UsageError(exc.format_message(), hint="see `evokit --help`"))
        return 2
    except click.exceptions.Abort:
        return 130
    except EvoError as exc:
        emit_error(exc)
        return exc.code
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        emit_error(EvoError(f"{type(exc).__name__}: {exc}", hint=traceback.format_exc(limit=5)))
        return 1
    return result if isinstance(result, int) else 0


if __name__ == "__main__":
    sys.exit(main())
