"""Commands for working on Files genomes in directories: what an agent uses when it is the operator."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from evokit.errors import UsageError
from evokit.files.project import Project
from evokit.files.tree import Files
from evokit.interface.commands import command
from evokit.interface.run import RunCommands
from evokit.records import Individual


class FileCommands:
    """checkout-individual, evaluate-directory (and edit-individual for tools)."""

    def __init__(self, project: Project):
        self.project = project

    @command
    def checkout_individual(self, individual: Individual, directory: str) -> dict:
        """Write the full project (environment plus this individual's genome) into a new directory to edit.

        Edit the genome files there, then evaluate_directory.
        """
        path = Path(directory)
        if path.exists() and any(path.iterdir()):
            raise UsageError(f"directory is not empty: {path}", hint="check out into a new directory")
        self.project.write(_files(individual), path)
        return {
            "directory": str(path.resolve()),
            "individual": individual.id,
            "genome": self.project.editable,
            "hint": f"Edit only files matching {self.project.editable}, then "
            f"`evokit evaluate-directory {path} --parent {individual.id}`.",
        }

    @command
    def evaluate_directory(
        self, run: RunCommands, directory: str, parents: list[str] | None = None, name: str = "", description: str = ""
    ) -> Individual:
        """Make an individual from a directory's genome files, evaluate it, and offer it to the archive.

        `name` is one line saying what this version is, `description` what changed and why. The
        result has its evaluation and info.admitted. Edits to environment files (anything outside
        the genome) are rejected.
        """
        path = Path(directory)
        if not path.is_dir():
            raise UsageError(f"not a directory: {path}")
        return run.evaluate_genome(self.project.read(path), parents, name, description)

    @command(surfaces=("tools",))
    def edit_individual(
        self, run: RunCommands, individual: Individual, files: Mapping[str, str | None]
    ) -> Individual:
        """Evaluate a copy of an individual with files replaced (None deletes), and offer it to the archive."""
        outside = [p for p in files if not self.project.is_genome(p)]
        if outside:
            raise UsageError(f"not genome files: {', '.join(outside)}", hint=f"editable: {self.project.editable}")
        return run.evaluate_genome(_files(individual).with_files(files), [individual.id])


def _files(individual: Individual) -> Files:
    if not isinstance(individual.genome, Files):
        raise UsageError(f"individual {individual.id} does not have a Files genome")
    return individual.genome
