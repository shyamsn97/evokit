"""Project: which files of a seed directory evolve (the genome) and which stay fixed (the environment)."""

from __future__ import annotations

import atexit
import shutil
import tempfile
from pathlib import Path
from typing import Sequence

from evokit.errors import EnvironmentModified, UsageError
from evokit.files.tree import DEFAULT_IGNORE, Files, sha256, walk_files
from evokit.util import glob_match

DEFAULT_PROTECTED = ("evaluate*", "eval/**", ".git/**")


class Project:
    """A seed directory split into genome files and a read-only environment.

    The environment (everything that is not genome: the evaluator, data, ...) is copied once
    into `environment` (a temporary directory by default; pass a path to share it across
    processes), so later edits to the seed directory cannot change what is evaluated.
    """

    def __init__(
        self,
        seed: str | Path,
        genome: Sequence[str] = ("**",),
        protected: Sequence[str] = DEFAULT_PROTECTED,
        ignore: Sequence[str] = DEFAULT_IGNORE,
        environment: str | Path | None = None,
        scratch: str | Path | None = None,
    ):
        self.seed = Path(seed).resolve()
        if not self.seed.is_dir():
            raise UsageError(f"seed directory not found: {self.seed}")
        self.genome_patterns = tuple(genome)
        self.protected = tuple(protected)
        self.ignore = tuple(ignore)
        if environment is None:
            self.environment = Path(tempfile.mkdtemp(prefix="evokit-env-"))
            atexit.register(shutil.rmtree, self.environment, True)
        else:
            self.environment = Path(environment).resolve()
        self.scratch = Path(scratch).resolve() if scratch else self.environment.parent / f"{self.environment.name}-scratch"
        self.ignore += tuple(p for p in map(self._inside_seed, (self.environment, self.scratch)) if p)
        if not self.environment.exists() or not any(self.environment.iterdir()):
            self._snapshot()
        self._hashes = {rel: sha256(full.read_bytes()) for rel, full in walk_files(self.environment, ())}

    def _inside_seed(self, path: str | Path) -> str | None:
        try:
            return Path(path).resolve().relative_to(self.seed).as_posix() + "/**"
        except ValueError:
            return None

    def _snapshot(self) -> None:
        self.environment.mkdir(parents=True, exist_ok=True)
        for rel, full in walk_files(self.seed, self.ignore):
            if not self.is_genome(rel):
                target = self.environment / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(full, target)

    @property
    def editable(self) -> list[str]:
        """The genome patterns, described for people and agents."""
        desc = list(self.genome_patterns)
        if self.protected:
            desc.append("except " + ", ".join(self.protected))
        return desc

    def is_genome(self, path: str) -> bool:
        return glob_match(path, self.genome_patterns) and not glob_match(path, self.protected)

    def seed_genome(self) -> Files:
        """The genome files of the seed directory, as they are now."""
        return Files({rel: full.read_bytes() for rel, full in walk_files(self.seed, self.ignore) if self.is_genome(rel)})

    @property
    def environment_digest(self) -> str:
        return sha256("\n".join(f"{p} {h}" for p, h in sorted(self._hashes.items())).encode())

    def write(self, genome: Files, directory: str | Path) -> None:
        """The full project in `directory`: the environment plus the genome's files."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        shutil.copytree(self.environment, directory, dirs_exist_ok=True)
        genome.write_to(directory)

    def read(self, directory: str | Path) -> Files:
        """The genome files in `directory`. Raises EnvironmentModified if environment files were edited."""
        files, modified = {}, []
        for rel, full in walk_files(directory, self.ignore):
            if self.is_genome(rel):
                files[rel] = full.read_bytes()
            elif rel in self._hashes and sha256(full.read_bytes()) != self._hashes[rel]:
                modified.append(rel)
        if modified:
            raise EnvironmentModified(sorted(modified), self.editable)
        return Files(files)

    def __repr__(self) -> str:
        return f"Project({str(self.seed)!r}, genome={list(self.genome_patterns)})"
