"""One `@command` registry, rendered into CLI, MCP, SKILL.md, and Python tools."""

from __future__ import annotations

import dataclasses
import datetime as _dt
import functools
import inspect
import typing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from evokit.records import Evaluation, Individual

SURFACES = frozenset({"cli", "mcp", "tools"})


def command(
    fn: Callable | None = None,
    *,
    name: str | None = None,
    surfaces: Iterable[str] = SURFACES,
    result: Callable[[Any, Any], Any] | None = None,
):
    """Mark a method as an agent-facing command.

    `result(owner, value)` optionally reshapes the return value for JSON surfaces
    (CLI and MCP); Python tools always get the raw value.
    """

    def mark(f: Callable) -> Callable:
        f.__evokit_command__ = {
            "name": (name or f.__name__).replace("_", "-"),
            "surfaces": frozenset(surfaces),
            "result": result,
        }
        return f

    return mark(fn) if fn is not None else mark


@dataclass(frozen=True)
class Command:
    name: str
    func: Callable  # bound method
    surfaces: frozenset
    result: Callable | None
    owner: Any
    attr: str

    @property
    def doc(self) -> str:
        return inspect.getdoc(self.func) or ""

    @property
    def summary(self) -> str:
        return self.doc.splitlines()[0] if self.doc else ""

    @property
    def signature(self) -> inspect.Signature:
        return inspect.signature(self.func)

    def shape(self, value: Any) -> Any:
        return encode(self.result(self.owner, value) if self.result else value)


def commands_of(obj: Any, surface: str | None = None) -> list[Command]:
    found = []
    for attr in dir(type(obj)):
        raw = getattr(type(obj), attr, None)
        meta = getattr(raw, "__evokit_command__", None)
        if meta is None or (surface and surface not in meta["surfaces"]):
            continue
        found.append(Command(meta["name"], getattr(obj, attr), meta["surfaces"], meta["result"], obj, attr))
    return found


SLOTS = {"archive": "archive", "strategy": "strategy", "operator": "operator", "eval_fn": "evaluator"}


def components_of(objects: Iterable[Any]) -> dict[str, Any]:
    """The swappable parts (archive, strategy, operator, evaluator) of the run these objects belong to."""
    out: dict[str, Any] = {}
    for obj in objects:
        for attr, slot in SLOTS.items():
            component = getattr(obj, attr, None)
            if component is not None:
                out.setdefault(slot, component)
    return out


def component_commands(objects: Iterable[Any], surface: str | None = None) -> dict[str, list[Command]]:
    """Commands contributed by the archive, strategy, operator, and evaluator, by slot."""
    out = {}
    for slot, component in components_of(objects).items():
        cmds = commands_of(component, surface)
        if cmds:
            out[slot] = cmds
    return out


def encode(value: Any) -> Any:
    """Convert records and results to JSON-compatible data."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Individual):
        return {
            "id": value.id,
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


def injected_arguments(func: Callable, run: Any) -> dict[str, Any]:
    """Parameters typed RunCommands, filled in rather than passed by the caller."""
    from evokit.interface.run import RunCommands

    return {name: run for name, hint in typing.get_type_hints(func).items() if hint is RunCommands}


def tools(*objects: Any) -> dict[str, Callable]:
    """Plain synchronous callables for REPL drivers (RLMs, notebooks, scripts), keyed by method name.

    Pass a RunCommands and optionally e.g. FileCommands. Parameters typed RunCommands are filled in.
    """
    from evokit.interface.run import RunCommands

    run = next((o for o in objects if isinstance(o, RunCommands)), None)

    def bind(func: Callable) -> Callable:
        injected = injected_arguments(func, run)
        if not injected or run is None:
            return func
        rest = inspect.Signature([p for p in inspect.signature(func).parameters.values() if p.name not in injected])

        @functools.wraps(func)
        def call(*args, **kwargs):
            return func(**rest.bind(*args, **kwargs).arguments, **injected)

        call.__signature__ = rest
        return call

    out: dict[str, Callable] = {}
    for obj in objects:
        for cmd in commands_of(obj, "tools"):
            out[cmd.attr] = bind(cmd.func)
    for slot, cmds in component_commands(objects, "tools").items():
        for cmd in cmds:
            out[f"{slot}_{cmd.attr}"] = bind(cmd.func)
    return out
