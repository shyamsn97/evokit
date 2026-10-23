"""HistoryCommands: log, show, diff and grep, git-style reading of a run from its storage alone.

They need no task and no archive class: individuals come from the history and the archive's
log, so they work on any run directory, including ones a Python recipe wrote.
"""

from __future__ import annotations

import difflib
import json
import re
from typing import Any

from evokit import history
from evokit.archive import Archive, read_members
from evokit.errors import UsageError
from evokit.interface.commands import command
from evokit.records import Individual
from evokit.storage import Storage
from evokit.util import truncate


def genome_text(genome: Any) -> list[tuple[str, str]]:
    """(path, text) pieces of a genome: each file of a Files-like genome, else the whole genome
    (a string as it is, anything else as indented JSON) under the path "genome"."""
    if hasattr(genome, "paths") and hasattr(genome, "text"):
        return [(path, genome.text(path)) for path in genome.paths()]
    if isinstance(genome, str):
        return [("genome", genome)]
    return [("genome", json.dumps(genome, indent=2, sort_keys=True, default=str))]


def diff_genomes(a: Any, b: Any, label_a: str = "a", label_b: str = "b") -> str:
    """A unified diff from genome `a` to `b`: the genome type's own `diff` if it has one."""
    if hasattr(a, "diff") and type(a) is type(b):
        return a.diff(b, label_a, label_b)
    out = []
    old, new = dict(genome_text(a)), dict(genome_text(b))
    for path in sorted(set(old) | set(new)):
        before = old.get(path, "").splitlines(keepends=True)
        after = new.get(path, "").splitlines(keepends=True)
        out.extend(difflib.unified_diff(before, after, f"{label_a}/{path}", f"{label_b}/{path}"))
    return "".join(out)


class HistoryCommands:
    """log, show, diff, grep.

    `archive` is the run's live Archive, or the name of its log (default "archive") to read
    membership from.
    """

    def __init__(self, storage: Storage, archive: Archive | str = "archive"):
        self.storage, self._archive = storage, archive

    def members(self) -> dict[str, Individual]:
        if isinstance(self._archive, Archive):
            return {m.id: m for m in self._archive.members()}
        return read_members(self.storage, self._archive)

    def lookup(self, id: str) -> Individual:
        """An individual by id: anything in the history, else an archive member."""
        try:
            return history.individual(self.storage, id)
        except KeyError:
            pass
        found = self.members().get(id)
        if found is None:
            raise UsageError(f"no individual {id!r}", hint="list them with `evokit log --all`")
        return found

    def _entry(self, individual: Individual, members: dict, admitted: set) -> dict:
        out: dict[str, Any] = {"id": individual.id, "name": individual.name, "parents": list(individual.parents)}
        if individual.evaluation is not None:
            out["objectives"] = dict(individual.evaluation.objectives)
        out["admitted"] = individual.id in admitted
        out["member"] = individual.id in members
        if individual.info.get("operator"):
            out["operator"] = individual.info["operator"]
        return out

    def _ancestors(self, start: Individual) -> list[Individual]:
        seen, out, queue = set(), [], [start]
        while queue:
            current = queue.pop(0)
            if current.id in seen:
                continue
            seen.add(current.id)
            out.append(current)
            for parent in current.parents:
                try:
                    queue.append(self.lookup(parent))
                except UsageError:
                    pass
        return out

    @command
    def log(self, individual: Individual | None = None, all: bool = False, limit: int | None = None) -> list[dict]:
        """An individual and its ancestors, nearest first, like `git log`.

        Without an individual, starts from the last one the archive admitted. With `all`, every
        evaluated individual instead, newest first, rejected ones too. Each entry has its name,
        objectives, whether it was admitted, and whether it is still in the archive (`member`).
        """
        if all:
            chain = list(reversed(history.evaluated(self.storage)))
        else:
            if individual is None:
                admitted = history.admitted(self.storage)
                if not admitted:
                    return []
                individual = admitted[-1]
            chain = self._ancestors(individual)
        members, admitted_ids = self.members(), {i.id for i in history.admitted(self.storage)}
        return [self._entry(i, members, admitted_ids) for i in chain[:limit]]

    @command(positional=("path",))
    def show(self, individual: Individual, path: str | None = None) -> Any:
        """An individual's name, description, parents, evaluation and files; with `path`, that file's text."""
        if path is not None:
            files = dict(genome_text(individual.genome))
            if path not in files:
                raise UsageError(f"{path} is not in {individual.id}", hint=f"files: {', '.join(files)}")
            return files[path]
        out: dict[str, Any] = {
            "id": individual.id,
            "name": individual.name,
            "description": individual.description,
            "parents": list(individual.parents),
            "member": individual.id in self.members(),
            "files": [p for p, _ in genome_text(individual.genome)],
        }
        if individual.evaluation is not None:
            out["objectives"] = dict(individual.evaluation.objectives)
            out["feedback"] = individual.evaluation.feedback
        out["info"] = dict(individual.info)
        return out

    @command
    def diff(self, a: Individual, b: Individual | None = None) -> str:
        """What changed from A to B, as a unified diff; with only A, from its first parent to A (like `git show`)."""
        if b is None:
            if not a.parents:
                raise UsageError(f"{a.id} has no parent", hint="give two ids: `evokit diff A B`")
            a, b = self.lookup(a.parents[0]), a
        return diff_genomes(a.genome, b.genome, a.id, b.id)

    @command
    def grep(self, pattern: str, limit: int = 20) -> list[dict]:
        """Individuals whose name, description or genome matches a regex, rejected ones too.

        Newest first; each result has its name, objectives, and the matching lines.
        """
        try:
            regex = re.compile(pattern)
        except re.error as exc:
            raise UsageError(f"invalid pattern {pattern!r}: {exc}") from None
        members, admitted = self.members(), {i.id for i in history.admitted(self.storage)}
        candidates = {i.id: i for i in reversed(history.evaluated(self.storage))}
        for id, member in members.items():
            candidates.setdefault(id, member)
        out = []
        for individual in candidates.values():
            pieces = [("name", individual.name), ("description", individual.description), *genome_text(individual.genome)]
            matches = [
                {"path": path, "line": n, "text": truncate(line.strip(), 200)}
                for path, text in pieces
                for n, line in enumerate(text.splitlines(), 1)
                if regex.search(line)
            ]
            if matches:
                out.append({**self._entry(individual, members, admitted), "matches": matches[:20]})
                if len(out) >= limit:
                    break
        return out
