"""The history of a run: what was created, evaluated, discarded and noted, with what it cost.

    history = History(storage)

`ask` and `tell` write it; a recipe only writes notes and discards. Everything is one JSON line
per record in `history/records.jsonl`. Totals are kept in memory and brought up to date with
what other processes appended (CLI calls, agents) before every read and write, so checking a
budget every iteration reads only the new lines. Use one History per run in a process: which
pending children this process holds is kept in it.
"""

from __future__ import annotations

import difflib
import json
import logging
import re
import threading
import time
from collections import Counter
from typing import Any, Iterable, Mapping

from evokit.archive import read_members
from evokit.errors import UsageError
from evokit.records import Individual, merge_usage
from evokit.storage import Storage, decode, encode
from evokit.util import truncate

log = logging.getLogger("evokit")

LOG = "history/records.jsonl"


class History:
    """The run's record in `storage`. `archive` names the archive whose log says who is a member."""

    def __init__(self, storage: Storage, archive: str = "archive"):
        self.storage, self.archive = storage, archive
        self._usage: Counter = Counter()
        self._counts: Counter = Counter()
        self._individuals: dict[str, Individual] = {}
        self._evaluated: list[str] = []
        self._admitted: list[str] = []
        self._pending: dict[str, None] = {}  # insertion-ordered set
        self._claimed: set[str] = set()      # pending, but handed out by this History
        self._notes: dict[str, str] = {}     # the last text of each note kind
        self._failures_in_a_row = 0          # "failed" notes since the last creation
        self._offset = 0                     # bytes of the log applied so far
        self._lock = threading.RLock()

    def __repr__(self) -> str:
        return f"History({self.storage!r})"

    # keeping up with the log

    def refresh(self) -> None:
        """Apply what was appended to the log since the last read, by this process or another."""
        with self._lock:
            data = self.storage.read_from(LOG, self._offset) if self.storage.exists(LOG) else b""
            end = data.rfind(b"\n") + 1  # only whole lines: another writer may be mid-line
            for line in data[:end].splitlines():
                if line.strip():
                    self._apply(json.loads(line))
            self._offset += end

    def _apply(self, record: dict) -> None:
        kind = record["kind"]
        self._usage.update(merge_usage(record.get("usage")))
        if kind == "created":
            for data in record["children"]:
                child = decode(data, self.storage)
                self._individuals[child.id] = child
                self._pending[child.id] = None
            self._counts["created"] += len(record["children"])
            self._failures_in_a_row = 0
        elif kind == "evaluated":
            individual = decode(record["individual"], self.storage)
            self._individuals[individual.id] = individual
            self._pending.pop(individual.id, None)
            self._claimed.discard(individual.id)
            self._evaluated.append(individual.id)
            self._counts["evaluated"] += 1
            if record["added"]:
                self._admitted.append(individual.id)
                self._counts["added"] += 1
        elif kind == "discarded":
            self._pending.pop(record["id"], None)
            self._claimed.discard(record["id"])
            self._counts["discarded"] += 1
        elif kind == "note":
            self._notes[record["note"]] = record.get("text", "")
            self._counts[record["note"]] += 1
            if record["note"] == "failed":
                self._failures_in_a_row += 1

    def _write(self, record: dict) -> None:
        line = (json.dumps({**record, "time": time.time()}, default=str) + "\n").encode()
        with self._lock:
            self.storage.append(LOG, line)
            self.refresh()

    # writing: ask, tell, and recipes that replace them

    def record_created(self, children: Iterable[Individual], parents: Iterable[str] = (), usage=None) -> None:
        """Children an operator made. They stay pending until evaluated or discarded; this History holds them."""
        children = list(children)
        with self._lock:
            self._write({
                "kind": "created",
                "children": [encode(c, self.storage) for c in children],
                "parents": list(parents),
                "usage": merge_usage(usage),
            })
            self._claimed.update(c.id for c in children)

    def record_evaluated(self, individual: Individual, *, added: bool) -> None:
        """An evaluated individual, and whether the archive admitted it. Its cost is its evaluation's usage."""
        usage = individual.evaluation.usage if individual.evaluation is not None else {}
        self._write({"kind": "evaluated", "individual": encode(individual, self.storage), "added": added, "usage": usage})

    def claim_pending(self, n: int) -> list[Individual]:
        """Up to n children that were created but never evaluated (e.g. a crash), not held by this History."""
        with self._lock:
            self.refresh()
            free = [i for i in self._pending if i not in self._claimed][:n]
            self._claimed.update(free)
            return [self._individuals[i] for i in free]

    def note(self, kind: str, text: str = "", usage=None) -> None:
        """Anything else worth keeping: a failed operator call, a redirect, a summary, an LLM judge's cost."""
        if text:
            log.info("%s: %s", kind, truncate(text, 500))
        self._write({"kind": "note", "note": kind, "text": text, "usage": merge_usage(usage)})

    def discard(self, individual: Individual, reason: str) -> None:
        """A child that was created but will not be evaluated."""
        self._write({"kind": "discarded", "id": individual.id, "reason": reason})

    def write_logs(self, folder: str, logs: Mapping[str, bytes]) -> None:
        """Files kept with the run, e.g. an evaluation's or an agent's logs under artifacts/<id>/."""
        for name, data in logs.items():
            self.storage.write(f"{folder}/{name}", data)

    # reading

    def usage(self) -> Counter:
        """cost_usd, tokens, ...: the sum over everything recorded."""
        self.refresh()
        return Counter(self._usage)

    def counts(self) -> Counter:
        """"created", "evaluated", "added", "discarded", and each note kind."""
        self.refresh()
        return Counter(self._counts)

    def evaluated(self) -> list[Individual]:
        """Everything evaluated so far, in order, with their evaluations."""
        self.refresh()
        return [self._individuals[i] for i in self._evaluated]

    def admitted(self) -> list[Individual]:
        """The ones the archive admitted at the time."""
        self.refresh()
        return [self._individuals[i] for i in self._admitted]

    def pending(self) -> list[Individual]:
        """Created, but not yet evaluated or discarded."""
        self.refresh()
        return [self._individuals[i] for i in self._pending]

    def individual(self, id: str) -> Individual:
        """Raises KeyError if the history has no such individual."""
        self.refresh()
        return self._individuals[id]

    def failures_in_a_row(self) -> int:
        """Operator failures ("failed" notes) since the last child was created."""
        self.refresh()
        return self._failures_in_a_row

    def last_note(self, kind: str) -> str | None:
        self.refresh()
        return self._notes.get(kind)

    def records(self) -> list[dict]:
        """Every record, oldest first, as stored: for tools like compare.py."""
        if not self.storage.exists(LOG):
            return []
        return [json.loads(line) for line in self.storage.read(LOG).decode().splitlines() if line.strip()]

    # reading like git: what the CLI's log, show, diff and grep print

    def members(self) -> dict[str, Individual]:
        """The archive's current members by id, from its log."""
        return read_members(self.storage, self.archive)

    def lookup(self, id: str) -> Individual:
        """An individual by id: anything in the history, else an archive member."""
        try:
            return self.individual(id)
        except KeyError:
            pass
        found = self.members().get(id)
        if found is None:
            raise UsageError(f"no individual {id!r}", hint="list them with `evokit log --all`")
        return found

    def log(self, individual: Individual | None = None, all: bool = False, limit: int | None = None) -> list[dict]:
        """An individual and its ancestors, nearest first, like `git log`.

        Without an individual, starts from the last one the archive admitted. With `all`, every
        evaluated individual instead, newest first, rejected ones too. Each entry has its name,
        objectives, whether it was admitted, and whether it is still in the archive (`member`).
        """
        if all:
            chain = list(reversed(self.evaluated()))
        else:
            if individual is None:
                admitted = self.admitted()
                if not admitted:
                    return []
                individual = admitted[-1]
            chain = self._ancestors(individual)
        members, admitted_ids = self.members(), {i.id for i in self.admitted()}
        return [self._entry(i, members, admitted_ids) for i in chain[:limit]]

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

    def diff(self, a: Individual, b: Individual | None = None) -> str:
        """What changed from A to B, as a unified diff; with only A, from its first parent to A (like `git show`)."""
        if b is None:
            if not a.parents:
                raise UsageError(f"{a.id} has no parent", hint="give two ids: `evokit diff A B`")
            a, b = self.lookup(a.parents[0]), a
        return diff_genomes(a.genome, b.genome, a.id, b.id)

    def grep(self, pattern: str, limit: int = 20) -> list[dict]:
        """Individuals whose name, description or genome matches a regex, rejected ones too.

        Newest first; each result has its name, objectives, and the matching lines.
        """
        try:
            regex = re.compile(pattern)
        except re.error as exc:
            raise UsageError(f"invalid pattern {pattern!r}: {exc}") from None
        members, admitted = self.members(), {i.id for i in self.admitted()}
        candidates = {i.id: i for i in reversed(self.evaluated())}
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


__all__ = ["History", "diff_genomes", "genome_text"]
