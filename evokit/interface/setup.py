"""Setup: the objects behind a run's commands, rebuilt from the run's task on every CLI call.

A run directory records its task folder (run/task.json, written when the run is created and
seeded), so later commands rebuild the same objects from any working directory. The read
commands (log, show, diff, grep) need only the run's storage, so `open_history` opens any run
directory, including ones a Python recipe wrote.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from evokit.history import LOG
from evokit.errors import UsageError
from evokit.files.commands import FileCommands
from evokit.interface.history import HistoryCommands
from evokit.interface.run import RunCommands
from evokit.storages.directory import DirectoryStorage
from evokit.task import open_run

T = TypeVar("T")


@dataclass(frozen=True)
class Setup:
    """RunCommands, FileCommands over the task's project, HistoryCommands, then whatever else the
    task's factory returned; or only HistoryCommands, for a run opened from its storage alone."""

    objects: tuple[Any, ...]

    @property
    def run(self) -> RunCommands | None:
        return self.find(RunCommands)

    @property
    def history(self) -> HistoryCommands:
        return self.find(HistoryCommands)

    @property
    def directory(self) -> Path:
        return self.history.storage.root

    def find(self, cls: type[T]) -> T | None:
        return next((o for o in self.objects if isinstance(o, cls)), None)


def open_setup(directory: str | Path, task: str | Path | None = None) -> Setup:
    """The run in `directory`; a new directory needs `task`, and is created and seeded from it."""
    run = open_run(directory, task)
    files = next((o for o in run.extras if isinstance(o, FileCommands)), None) or FileCommands(run.task.project)
    return Setup((run, files, run.history, *(o for o in run.extras if o is not files)))


def open_history(directory: str | Path) -> Setup:
    """The run in `directory`, for reading only: no task needed."""
    if not (Path(directory) / LOG).is_file():
        raise UsageError(f"no run in {Path(directory)}", hint="point --dir (or $EVOKIT_DIR) at a run directory")
    return Setup((HistoryCommands(DirectoryStorage(directory)),))
