"""Task: what evolves and how it is scored.

    task = evokit.Task(seed=1.0, evaluate=lambda individual: Evaluation({"score": -abs(individual.genome - 3)}))
    task = evokit.Task.from_directory("examples/circle_packing")
    experiment = task.create_experiment("experiments/one", max_cost=5)

A genome is any JSON value, or Files. A task folder holds Files:

    packing/
      TASK.md        what the agent is told: the goal, the rules, what "better" means
      config.yaml    which files evolve and how they are scored (`task:`), and optionally which
                     recipe `evokit run` uses (`recipe:`)
      evaluate.py    the evaluator: part of the fixed environment, never evolved
      main.py        the seed: the genome files as they are now

`evokit init NAME` writes a working template.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import yaml

from evokit.errors import UsageError
from evokit.files import DEFAULT_IGNORE, DEFAULT_PROTECTED, Files, Project, ScriptEvaluator
from evokit.evaluate import Evaluate

if TYPE_CHECKING:
    from evokit.archive import Archive
    from evokit.experiment import Experiment, StopRule

CONFIG, INSTRUCTIONS = "config.yaml", "TASK.md"
GENOME_JSON = "genome.json"
TEMPLATE = Path(__file__).parent / "templates" / "task"
IGNORE = (*DEFAULT_IGNORE, ".evokit/**", "runs/**", "experiments/**")
PROTECTED = (*DEFAULT_PROTECTED, CONFIG, INSTRUCTIONS)


@dataclass(frozen=True)
class Task:
    """`seed`: the first genome (any JSON value, or Files). `evaluate(individual) -> Evaluation`,
    sync or async: the one place scores come from. `instructions`: what agents are told.
    `objective` and `minimize`: what archives rank by. `files`: for a Files genome, which files
    of a folder evolve and which are the fixed environment around them (`Task.from_directory`)."""

    seed: Any
    evaluate: Evaluate
    instructions: str = ""
    objective: str = "score"
    minimize: bool = False
    files: Project | None = None
    directory: Path | None = None

    @classmethod
    def from_directory(cls, directory: str | Path, environment: str | Path | None = None, ignore: Sequence[str] = ()) -> "Task":
        """Read a task folder. `environment` is where the fixed files are snapshotted (a temporary
        folder by default), and `ignore` adds patterns that are neither genome nor environment."""
        directory = Path(directory).resolve()
        spec = read_config(directory).get("task") or {}
        for key in ("genome", "evaluate"):
            if key not in spec:
                raise UsageError(f"{CONFIG} has no task.{key}", hint="e.g. genome: [main.py] and evaluate: python evaluate.py under task:")
        instructions = directory / INSTRUCTIONS
        if not instructions.is_file():
            raise UsageError(f"no {INSTRUCTIONS} in {directory}", hint="describe the goal, the rules and the scoring there")
        files = Project(
            directory,
            genome=spec["genome"],
            protected=PROTECTED,
            ignore=(*IGNORE, *spec.get("ignore", ()), *ignore),
            environment=environment,
        )
        return cls(
            seed=files.seed_genome(),
            evaluate=ScriptEvaluator(files, spec["evaluate"], timeout=spec.get("timeout")),
            instructions=instructions.read_text().strip(),
            objective=spec.get("objective", "score"),
            minimize=bool(spec.get("minimize", False)),
            files=files,
            directory=directory,
        )

    @property
    def name(self) -> str:
        return self.directory.name if self.directory else "task"

    @property
    def editable(self) -> list[str]:
        """Which files in a checkout are the genome, described for agents."""
        return self.files.editable if self.files else [GENOME_JSON]

    def checkout(self, genome: Any, directory: str | Path) -> Path:
        """`directory`, emptied, then holding `genome`: with `files`, the whole folder (environment
        and genome files); otherwise the genome as genome.json. Where an agent works."""
        directory = Path(directory)
        if self.files is not None:
            self.files.checkout(genome if isinstance(genome, Files) else Files(), directory)
        else:
            _empty(directory)
            (directory / GENOME_JSON).write_text(json.dumps(genome, indent=2) + "\n")
        return directory

    def read(self, directory: str | Path) -> Any:
        """The genome in `directory`, as an agent left it. Raises EnvironmentModified if it edited
        fixed files, UsageError if genome.json is not JSON."""
        if self.files is not None:
            return self.files.read(directory)
        path = Path(directory) / GENOME_JSON
        try:
            return json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise UsageError(f"{GENOME_JSON} is not valid JSON: {exc}", hint="leave the genome as one JSON value") from None

    def create_experiment(
        self,
        dir: str | Path | None = None,
        *,
        archive: "Archive | None" = None,
        max_cost: float | None = None,
        max_evaluations: int | None = None,
        max_failures: int | None = 10,
        stop: Sequence["StopRule"] = (),
    ) -> "Experiment":
        """A search on this task, kept in `dir` (in memory without one), or the one there, continued.
        `archive` (default: BestArchive) is kept in the experiment; a new experiment scores the seed
        into it first (UsageError, and nothing recorded, if the seed fails). The limits hold over
        the whole experiment, restarts included."""
        from evokit.experiment import Experiment

        return Experiment.new(self, dir, archive, max_cost, max_evaluations, max_failures, stop)


def read_config(directory: str | Path) -> dict:
    """A task folder's config.yaml."""
    directory = Path(directory)
    if not (directory / CONFIG).is_file():
        raise UsageError(f"no {CONFIG} in {directory}", hint="create a task with `evokit init NAME`")
    try:
        config = yaml.safe_load((directory / CONFIG).read_text()) or {}
    except yaml.YAMLError as exc:
        raise UsageError(f"{directory / CONFIG}: {exc}") from None
    if not isinstance(config, dict):
        raise UsageError(f"{CONFIG} must be a mapping with a `task:` section")
    return config


def init_task(directory: str | Path) -> Path:
    """Write the task template into `directory` (new or empty): a toy task that runs as it is."""
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise UsageError(f"{directory} is not empty", hint="choose a new folder for the task")
    shutil.copytree(TEMPLATE, directory, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__"))
    return directory.resolve()


def _empty(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for child in directory.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()


__all__ = ["Task", "init_task", "read_config"]
