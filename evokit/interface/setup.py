"""Setup: the objects behind a run's commands, rebuilt from the run's task on every CLI call.

A run directory records its task folder (run/task.json, written when the run is created and
seeded), so later commands rebuild the same objects from any working directory.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from evokit.files.commands import FileCommands
from evokit.interface.run import RunCommands
from evokit.task import open_run

T = TypeVar("T")


@dataclass(frozen=True)
class Setup:
    """RunCommands first, then FileCommands over the task's project, then whatever else the task's factory returned."""

    objects: tuple[Any, ...]

    @property
    def run(self) -> RunCommands:
        return self.objects[0]

    @property
    def directory(self) -> Path:
        return self.run.storage.root

    def find(self, cls: type[T]) -> T | None:
        return next((o for o in self.objects if isinstance(o, cls)), None)


def open_setup(directory: str | Path, task: str | Path | None = None) -> Setup:
    """The run in `directory`; a new directory needs `task`, and is created and seeded from it."""
    run = open_run(directory, task)
    files = next((o for o in run.extras if isinstance(o, FileCommands)), None) or FileCommands(run.task.project)
    return Setup((run, files, *(o for o in run.extras if o is not files)))
