"""Task: a folder that defines a problem, loaded into the objects a run is made of.

    packing/
      TASK.md        what the operator is told: the goal, the rules, what "better" means
      config.yaml    which files evolve, how they are scored, which components to use
      evaluate.py    the evaluator: part of the fixed environment, never evolved
      main.py        the seed: the genome files as they are now

`load_task(dir)` reads the folder into a Task: a Project (genome files plus the read-only
environment), an eval function (ScriptEvaluator over the `evaluate` command), the instructions
(TASK.md), and the `run` section naming the archive, strategy and operator. `task.open(storage)`
builds those components on the storage, as a Run; a new storage is seeded with the task's genome
files, an existing one continues where it stopped. `evokit init NAME` writes a working template.
"""

from __future__ import annotations

import contextlib
import importlib
import importlib.util
import inspect
import json
import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

import yaml

from evokit.archive import Archive
from evokit.archives import (
    BestArchive,
    IslandArchive,
    LatestArchive,
    PopulationArchive,
    SimplerBestArchive,
    UnboundedArchive,
)
from evokit.errors import UsageError
from evokit.agents import ClaudeCode, CodingAgentOperator, CommandAgent
from evokit.files import (
    DEFAULT_IGNORE,
    DEFAULT_PROTECTED,
    DiffOperator,
    FileRewriteOperator,
    Project,
    ScriptEvaluator,
)
from evokit.history import LOG, History
from evokit.loop import EvalFn, as_eval_fn, tell
from evokit.operator import Operator
from evokit.operators import AnthropicModel
from evokit.records import Individual
from evokit.storage import Storage
from evokit.storages import DirectoryStorage
from evokit.strategies import Greedy, IslandSampler, ProportionalSelection, Tournament
from evokit.strategy import Strategy
from evokit.util import background, truncate

CONFIG, INSTRUCTIONS = "config.yaml", "TASK.md"
RUN_TASK = "run/task.json"
TEMPLATE = Path(__file__).parent / "templates" / "task"
IGNORE = (*DEFAULT_IGNORE, ".evokit/**", "runs/**")
PROTECTED = (*DEFAULT_PROTECTED, CONFIG, INSTRUCTIONS)

ARCHIVES: dict[str, Callable[..., Archive]] = {
    "best": BestArchive,
    "simpler": SimplerBestArchive,
    "population": lambda size=10, **kw: PopulationArchive(size, **kw),
    "islands": IslandArchive,
    "unbounded": UnboundedArchive,
    "latest": LatestArchive,
}
STRATEGIES: dict[str, Callable[..., Strategy]] = {
    "greedy": lambda objective, **kw: Greedy(**kw),
    "tournament": lambda objective, **kw: Tournament(**kw),
    "proportional": lambda objective, **kw: ProportionalSelection(objective=objective, **kw),
    "islands": lambda objective, **kw: IslandSampler(**kw),
}
OPERATORS = ("none", "diff", "rewrite", "coding-agent")


@dataclass(frozen=True)
class Run:
    """A run opened from its storage: what the `evokit` commands work on. A run opened only to be
    read (`open_history`) has just its storage and history; `tools` are extra commands from the
    task's factory."""

    storage: Storage
    history: History
    archive: Archive | None = None
    strategy: Strategy | None = None
    operator: Operator | None = None
    eval_fn: EvalFn | None = None
    task: "Task | None" = None
    tools: tuple[Callable, ...] = ()


@dataclass(frozen=True)
class Task:
    directory: Path
    project: Project
    eval_fn: EvalFn
    instructions: str
    objective: str = "score"
    minimize: bool = False
    run: dict = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.directory.name

    def seed(self) -> Individual:
        """The task's genome files as they are now, as a fresh individual."""
        return Individual.new(self.project.seed_genome(), name="seed")

    def components(self, storage: Storage, history: History | None = None) -> tuple[Archive, Strategy, Operator | None]:
        """The archive (on `storage`, so it restores itself), strategy and operator the `run` section
        names; a coding agent operator reads the run through `history` (default: a new one)."""
        archive, strategy, operator, _, _ = self._build(storage, history)
        return archive, strategy, operator

    def _build(self, storage: Storage, history: History | None = None) -> tuple[Archive, Strategy, Operator | None, EvalFn, tuple]:
        if "factory" in self.run:
            return self._from_factory(storage)
        run = self.run
        try:
            archive = ARCHIVES[run.get("archive", "best")](
                objective=self.objective, minimize=self.minimize, storage=storage, **run.get("archive_options", {})
            )
            strategy = STRATEGIES[run.get("strategy", "greedy")](self.objective, **run.get("strategy_options", {}))
        except KeyError as exc:
            raise UsageError(f"{CONFIG}: unknown archive or strategy {exc}", hint=f"archives: {', '.join(ARCHIVES)}; strategies: {', '.join(STRATEGIES)}") from None
        return archive, strategy, self._operator(history or History(storage)), self.eval_fn, ()

    def _operator(self, history: History) -> Operator | None:
        kind, options = self.run.get("operator", "none"), dict(self.run.get("operator_options", {}))
        if kind == "none":
            return None
        if kind in ("diff", "rewrite"):
            price = self.run.get("price")
            model = AnthropicModel(self.run.get("model", "claude-sonnet-4-5"), price=tuple(price) if price else None)
            cls = DiffOperator if kind == "diff" else FileRewriteOperator
            return cls(model, self.project, instructions=self.instructions, **options)
        if kind == "coding-agent":
            agent = self.run.get("agent", "claude")
            agent = ClaudeCode() if agent == "claude" else CommandAgent(agent)
            if options.get("max_draft_evals"):
                options["eval_fn"] = self.eval_fn
            return CodingAgentOperator(agent, self.project, history=history, instructions=self.instructions, **options)
        raise UsageError(f"{CONFIG}: unknown operator {kind!r}", hint=f"operators: {', '.join(OPERATORS)}")

    def _from_factory(self, storage: Storage):
        """`factory: MODULE:NAME`: build(task, storage) returns the archive, strategy, optionally an
        operator and an eval function (default: the task's), and optionally a list of functions,
        which become extra commands."""
        with _working_directory(self.directory):
            build = load_factory(self.run["factory"], self.directory)
            returned = build(self, storage)
        objects = list(returned) if isinstance(returned, (tuple, list)) else [returned]
        archive = _one([o for o in objects if isinstance(o, Archive)], "archive")
        strategy = _one([o for o in objects if isinstance(o, Strategy)], "strategy")
        operator = _one([o for o in objects if isinstance(o, Operator)], "operator", required=False)
        eval_fn = _one([o for o in objects if callable(o) and not isinstance(o, type)], "eval function", required=False)
        if archive.storage is not storage:
            raise UsageError("the factory's archive does not use the run's storage", hint="build it with storage=storage")
        tools = tuple(tool for o in objects if isinstance(o, list) for tool in o)
        return archive, strategy, operator, eval_fn or self.eval_fn, tools

    def open(self, storage: Storage) -> Run:
        """A run of this task on `storage`. A new storage is seeded first: the task's genome files
        are evaluated and added to the archive (UsageError, and nothing recorded, if they fail). An
        existing one continues: its archive restores itself and nothing is re-run."""
        if storage.exists(RUN_TASK):
            recorded = Path(json.loads(storage.read(RUN_TASK))["task"])
            if recorded != self.directory:
                raise UsageError(f"this run belongs to the task {recorded}, not {self.directory}", hint="use another --dir")
        history = History(storage)
        archive, strategy, operator, eval_fn, tools = self._build(storage, history)
        run = Run(storage, history, archive, strategy, operator, eval_fn, self, tools)
        if not len(archive):
            seed(run, self.seed())
        if not storage.exists(RUN_TASK):
            storage.write(RUN_TASK, json.dumps({"task": str(self.directory)}).encode())
        return run


def seed(run: Run, individual: Individual) -> Individual:
    """Evaluate `individual` and add it to the archive. Raises UsageError, recording nothing, if it fails its evaluation."""
    evaluation = background.run(as_eval_fn(run.eval_fn)(individual))
    if not evaluation.objectives:
        raise UsageError(
            f"the seed failed its evaluation: {truncate(evaluation.feedback, 2000)}",
            hint="fix the task's files or its evaluator, then run the command again",
        )
    background.run(tell(run.history, run.archive, [individual], [evaluation]))
    return individual.with_evaluation(evaluation)


def load_task(directory: str | Path, environment: str | Path | None = None, ignore: Sequence[str] = ()) -> Task:
    """Read a task folder. `environment` is where the fixed files are snapshotted (a temporary folder
    by default; a run keeps its own in `<run>/env`), and `ignore` adds patterns that are neither genome
    nor environment."""
    directory = Path(directory).resolve()
    if not (directory / CONFIG).is_file():
        raise UsageError(f"no {CONFIG} in {directory}", hint="create a task with `evokit init NAME`")
    try:
        config = yaml.safe_load((directory / CONFIG).read_text()) or {}
    except yaml.YAMLError as exc:
        raise UsageError(f"{directory / CONFIG}: {exc}") from None
    if not isinstance(config, dict):
        raise UsageError(f"{CONFIG} must be a mapping with `task:` and `run:` sections")
    spec = config.get("task") or {}
    for key in ("genome", "evaluate"):
        if key not in spec:
            raise UsageError(f"{CONFIG} has no task.{key}", hint="e.g. genome: [main.py] and evaluate: python evaluate.py under task:")
    instructions = directory / INSTRUCTIONS
    if not instructions.is_file():
        raise UsageError(f"no {INSTRUCTIONS} in {directory}", hint="describe the goal, the rules and the scoring there")
    project = Project(
        directory,
        genome=spec["genome"],
        protected=PROTECTED,
        ignore=(*IGNORE, *spec.get("ignore", ()), *ignore),
        environment=environment,
    )
    return Task(
        directory=directory,
        project=project,
        eval_fn=ScriptEvaluator(project, spec["evaluate"], timeout=spec.get("timeout")),
        instructions=instructions.read_text().strip(),
        objective=spec.get("objective", "score"),
        minimize=bool(spec.get("minimize", False)),
        run=dict(config.get("run") or {}),
    )


def open_run(directory: str | Path, task: str | Path | None = None) -> Run:
    """The run in `directory`. A new directory needs `task`, and is seeded from it."""
    storage = DirectoryStorage(directory)
    if task is None:
        if not storage.exists(RUN_TASK):
            raise UsageError(f"no run in {Path(directory)}", hint="start one with --task TASK_DIR (create a task with `evokit init NAME`)")
        task = json.loads(storage.read(RUN_TASK))["task"]
    task_dir = Path(task).resolve()
    inside = _relative(storage.root, task_dir)
    loaded = load_task(task_dir, environment=storage.root / "env", ignore=(f"{inside}/**",) if inside else ())
    return loaded.open(storage)


def open_history(directory: str | Path) -> Run:
    """The run in `directory`, for reading only: no task needed, so it opens any run directory,
    including ones a Python recipe wrote."""
    if not (Path(directory) / LOG).is_file():
        raise UsageError(f"no run in {Path(directory)}", hint="point --dir (or $EVOKIT_DIR) at a run directory")
    storage = DirectoryStorage(directory)
    return Run(storage, History(storage))


def init_task(directory: str | Path) -> Path:
    """Write the task template into `directory` (new or empty): a toy task that runs as it is."""
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise UsageError(f"{directory} is not empty", hint="choose a new folder for the task")
    shutil.copytree(TEMPLATE, directory, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__"))
    return directory.resolve()


def load_factory(ref: str, cwd: Path) -> Callable:
    if ":" not in ref:
        raise UsageError(f"invalid factory {ref!r}", hint="use MODULE:NAME or FILE.py:NAME, e.g. evo:build")
    module_ref, _, attr = ref.rpartition(":")
    path = Path(module_ref) if module_ref.endswith(".py") else cwd / (module_ref.replace(".", "/") + ".py")
    if not path.is_absolute():
        path = cwd / path
    if path.is_file():
        name = f"_evokit_factory_{abs(hash(str(path.resolve())))}"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    elif module_ref.endswith(".py"):
        raise UsageError(f"factory file not found: {path}")
    else:
        module = importlib.import_module(module_ref)
    target = getattr(module, attr, None)
    if not callable(target) or len(inspect.signature(target).parameters) != 2:
        raise UsageError(f"{ref} must be a function build(task, storage)")
    return target


@contextlib.contextmanager
def _working_directory(path: Path) -> Iterator[None]:
    previous = os.getcwd()
    os.chdir(path)
    sys.path.insert(0, str(path))
    try:
        yield
    finally:
        os.chdir(previous)
        with contextlib.suppress(ValueError):
            sys.path.remove(str(path))


def _one(objects: list, kind: str, required: bool = True) -> Any:
    if len(objects) > 1 or (required and not objects):
        raise UsageError(f"the factory must return {'exactly' if required else 'at most'} one {kind} (got {len(objects)})")
    return objects[0] if objects else None


def _relative(path: Path, root: Path) -> str | None:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        return None
