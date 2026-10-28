"""Experiment: one search on a task. Its history (what was created, evaluated, discarded, with what
it cost), its archive, its limits, and what agents and recipes wrote down along the way (notes:
its memory).

    experiment = task.create_experiment("experiments/one", max_cost=5)   # or ek.open_experiment(dir)
    async with experiment.start(agent):                                   # agent.prompt works inside
        while not experiment.limits.reached():
            await agent.prompt("Improve the program, then submit it.")
    experiment.best()

Everything enters through the experiment: `create(make)` records a child a variation function
made (or why it failed, with what it cost), and `submit(individual)` scores it with the task's
evaluate function and offers it to the archive. Recipes and agents (the `remember` tool) write
notes. Everything is one JSON line per record in `history/records.jsonl`. Totals are kept in
memory and brought up to date with what other processes appended (CLI calls, agents) before
every read and write, so checking a budget every iteration reads only the new lines.

A process's first write to an experiment takes its driver lock and keeps it until the process
exits (or `close()`), so only one process at a time drives an experiment; reading never takes it.
"""

from __future__ import annotations

import asyncio
import atexit
import contextlib
import copy
import difflib
import functools
import inspect
import json
import logging
import pickle
import re
import shutil
import tempfile
import threading
import time
from collections import Counter
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Iterable, Mapping, Sequence

from evokit.archive import Archive, improves, read_members
from evokit.errors import OperatorError, UsageError
from evokit.evaluate import as_evaluate
from evokit.records import Evaluation, Individual, merge_usage
from evokit.storage import Storage, decode, encode
from evokit.util import NAME_LIMIT, background, new_id, title_and_body, truncate

if TYPE_CHECKING:
    from evokit.agents.harness import Agent, Tool
    from evokit.task import Task

logger = logging.getLogger("evokit")

LOG = "history/records.jsonl"
TASK_FILE = "task.json"
ARCHIVE_FILE = "archive.pickle"
FEEDBACK_LIMIT = 4000

_held: dict[str, contextlib.ExitStack] = {}  # storage key -> the driver lock this process holds
_held_lock = threading.Lock()


def _take(storage: Storage) -> None:
    """Hold `storage`'s driver lock for the rest of this process. RunBusy if another process has it."""
    with _held_lock:
        if storage.key not in _held:
            stack = contextlib.ExitStack()
            stack.enter_context(storage.lock("driver"))
            _held[storage.key] = stack


def _release_all() -> None:
    with _held_lock:
        for stack in _held.values():
            stack.close()
        _held.clear()


atexit.register(_release_all)


StopRule = Callable[["Experiment"], "str | None"]
"""`rule(experiment)` returns why the experiment should stop, or None."""


@dataclass(frozen=True)
class Limits:
    """When an experiment stops, counted over all of it (seeds included), so limits hold across restarts.

    `max_failures` stops an operator that fails every time: failed calls are neither evaluations
    nor (often) costly, so without it such a loop never reaches the other limits. `stop` rules add
    any condition, e.g. `lambda e: "plateau" if e.since_improvement() >= 30 else None`.
    """

    max_evaluations: int | None = None
    max_cost: float | None = None
    max_failures: int | None = 10
    stop: Sequence[StopRule] = ()
    experiment: "Experiment | None" = field(default=None, repr=False, compare=False)

    def reached(self) -> str | None:
        """Why the experiment should stop, or None."""
        experiment = self.experiment
        if experiment is None:
            return None
        if self.max_evaluations is not None and experiment.evaluations() >= self.max_evaluations:
            return f"{self.max_evaluations} evaluations"
        if self.max_cost is not None and experiment.cost() >= self.max_cost:
            return f"${self.max_cost:g} spent"
        if self.max_failures is not None and experiment.failures_in_a_row() >= self.max_failures:
            last = truncate(experiment.last_note("failed") or "", 1000)
            return f"{self.max_failures} failures in a row; the last: {last}"
        for rule in self.stop:
            if reason := rule(experiment):
                return reason
        return None

    def remaining_cost(self) -> float | None:
        """US dollars left under `max_cost`, or None if there is no cost limit."""
        if self.max_cost is None or self.experiment is None:
            return None
        return max(0.0, self.max_cost - self.experiment.cost())


class Experiment:
    """One search, kept in `storage`: its `task` (None when opened only to be read), its `archive`
    (the survivors), its `limits`, and its history. Made by `task.create_experiment` or
    `open_experiment`."""

    def __init__(self, storage: Storage, task: "Task | None" = None, archive: Archive | None = None, limits: Limits | None = None):
        self.storage, self.task, self.archive = storage, task, archive
        self.limits = replace(limits or Limits(max_failures=None), experiment=self)
        if archive is not None:
            archive.attach(storage, task.objective if task else "score", task.minimize if task else False)
        self._usage: Counter = Counter()
        self._counts: Counter = Counter()
        self._individuals: dict[str, Individual] = {}
        self._evaluated: list[str] = []
        self._admitted: list[str] = []
        self._pending: dict[str, None] = {}  # insertion-ordered set
        self._notes: dict[str, list[dict]] = {}  # each note kind's notes, oldest first: {"text", "time", ...}
        self._failures_in_a_row = 0          # "failed" notes since the last creation
        self._start: float | None = None     # the time of the first record
        self._attempts = 0                   # evaluated records plus draft notes
        self._evaluations_at_improvement = 0
        self._attempts_at_improvement = 0
        self._attempts_at_note: dict[str, int] = {}
        self._best: str | None = None        # the last individual that improved on the best
        self._best_over_time: list[tuple[int, str]] = []
        self._drafts: list[dict] = []
        self._offset = 0                     # bytes of the log applied so far
        self._lock = threading.RLock()

    @classmethod
    def new(
        cls,
        task: "Task",
        dir: str | Path | None = None,
        archive: Archive | None = None,
        max_cost: float | None = None,
        max_evaluations: int | None = None,
        max_failures: int | None = 10,
        stop: Sequence[StopRule] = (),
    ) -> "Experiment":
        """See `Task.create_experiment`."""
        from evokit.archives import BestArchive
        from evokit.storages import DirectoryStorage, MemoryStorage

        storage = DirectoryStorage(dir) if dir is not None else MemoryStorage()
        _check_task(storage, task)
        experiment = cls(storage, task, archive if archive is not None else BestArchive(), Limits(max_evaluations, max_cost, max_failures, tuple(stop)))
        if not len(experiment.archive):
            experiment.seed()
        for child in experiment.pending():  # created, then a crash before it was scored
            background.run(experiment.submit(child))
        if not storage.exists(TASK_FILE):
            storage.write(TASK_FILE, json.dumps({"task": str(task.directory) if task.directory else None}).encode())
        if not storage.exists(ARCHIVE_FILE):
            with contextlib.suppress(Exception):  # an archive with a lambda in it cannot be reopened
                storage.write(ARCHIVE_FILE, pickle.dumps(_template(experiment.archive)))
        return experiment

    def seed(self, genome: Any = None) -> Individual:
        """Score `genome` (default: the task's seed) and add it to the archive. Raises UsageError,
        recording nothing, if it fails its evaluation."""
        task = self._task()
        individual = Individual.new(task.seed if genome is None else genome, name="seed")
        evaluation = background.run(self.evaluate(individual))
        if not evaluation.objectives:
            raise UsageError(
                f"the seed failed its evaluation: {truncate(evaluation.feedback, 2000)}",
                hint="fix the task's files or its evaluator, then try again",
            )
        return background.run(self._tell(individual, evaluation))

    def _task(self) -> "Task":
        if self.task is None:
            raise UsageError("this experiment was opened without its task", hint="open it with its task (--task TASK_DIR)")
        return self.task

    async def evaluate(self, individual: Individual) -> Evaluation:
        """Score `individual` with the task's evaluate function, recording nothing."""
        return await as_evaluate(self._task().evaluate)(individual)

    def best(self) -> Individual | None:
        """The archive's best; without an archive, the last individual that improved on the best."""
        if self.archive is not None and hasattr(self.archive, "best"):
            return self.archive.best()
        self.refresh()
        return self._individuals[self._best] if self._best else None

    def __repr__(self) -> str:
        return f"Experiment({self.storage!r})"

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
        self._start = self._start or record.get("time")
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
            self._evaluated.append(individual.id)
            self._counts["evaluated"] += 1
            if record["added"]:
                self._admitted.append(individual.id)
                self._counts["added"] += 1
            self._attempts += 1
            if record.get("improved") or (self._best is None and record["added"]):
                self._best = individual.id
                self._best_over_time.append((self._counts["evaluated"], individual.id))
                self._evaluations_at_improvement = self._counts["evaluated"]
                self._attempts_at_improvement = self._attempts
        elif kind == "discarded":
            self._pending.pop(record["id"], None)
            self._counts["discarded"] += 1
        elif kind == "note":
            data = {k: v for k, v in record.items() if k not in ("kind", "note", "usage")}
            self._notes.setdefault(record["note"], []).append({"text": "", **data})
            self._counts[record["note"]] += 1
            if record["note"] == "failed":
                self._failures_in_a_row += 1
            if record["note"] == "draft":
                self._attempts += 1
                self._drafts.append({"objectives": record.get("objectives", {}), "improved": record.get("improved", False)})
                if record.get("improved"):
                    self._attempts_at_improvement = self._attempts
            self._attempts_at_note[record["note"]] = self._attempts

    def _write(self, record: dict) -> None:
        _take(self.storage)
        line = (json.dumps({**record, "time": time.time()}, default=str) + "\n").encode()
        with self._lock:
            self.storage.append(LOG, line)
            self.refresh()

    # writing: what goes in

    async def create(self, make: Awaitable[tuple[Individual, Mapping[str, float]]]) -> Individual | None:
        """Await `make`, a variation function's `(child, usage)`, and record the child as created,
        with its cost; it is pending until submitted or discarded. If it raises OperatorError,
        record why it failed (a "failed" note, with its cost and its logs under failed/<id>/) and
        return None."""
        try:
            child, usage = await make
        except OperatorError as exc:
            reason = truncate(exc.reason, FEEDBACK_LIMIT)
            if exc.logs:
                folder = f"failed/{new_id()}"
                self.write_logs(folder, exc.logs)
                reason += f"\n\nlogs: {folder}/"
            self.note("failed", reason, exc.usage)
            return None
        self.record_created([child], usage)
        logger.info("created %s from %s", child.id, ", ".join(child.parents) or "nothing")
        return child

    async def submit(self, individual: Individual, usage: Mapping[str, float] | None = None) -> Individual:
        """Score `individual` with the task's evaluate function and offer it to the archive; the
        score never comes from the caller. Records it as created first (with `usage`, what making
        it cost) unless `create` did. Returns it with its evaluation and info["admitted"]."""
        self.refresh()
        if individual.id not in self._individuals:
            self.record_created([individual], usage)
        return await self._tell(individual, await self.evaluate(individual))

    async def _tell(self, individual: Individual, evaluation: Evaluation) -> Individual:
        self.write_logs(f"artifacts/{individual.id}", evaluation.logs)
        individual = individual.with_evaluation(replace(evaluation, logs={}))
        archive = self.archive
        best = archive.best() if hasattr(archive, "best") else None
        added = bool(archive.add(individual))
        self.record_evaluated(individual, added=added, improved=added and improves(archive, individual, best))
        logger.info("evaluated %s: %s%s", individual.id, _scores(evaluation), " (added)" if added else "")
        return individual.with_info(admitted=added)

    def record_created(self, children: Iterable[Individual], usage=None) -> None:
        """Children made, with what making them cost. They stay pending until evaluated or discarded."""
        self._write({
            "kind": "created",
            "children": [encode(c, self.storage) for c in children],
            "usage": merge_usage(usage),
        })

    def record_evaluated(self, individual: Individual, *, added: bool, improved: bool = False) -> None:
        """An evaluated individual, whether the archive admitted it, and whether it strictly beat the
        best so far. Its cost is its evaluation's usage."""
        usage = individual.evaluation.usage if individual.evaluation is not None else {}
        self._write({
            "kind": "evaluated", "individual": encode(individual, self.storage),
            "added": added, "improved": improved, "usage": usage,
        })

    def note(self, kind: str, text: str = "", usage=None, **data) -> None:
        """Anything else worth keeping: a failed operator call, a redirect, a summary, an LLM judge's
        cost. `data` is stored with it, e.g. a draft's objectives and whether it improved."""
        if text:
            logger.info("%s: %s", kind, truncate(text, 500))
        self._write({"kind": "note", "note": kind, "text": text, "usage": merge_usage(usage), **data})

    # agents

    @contextlib.asynccontextmanager
    async def start(
        self,
        agent: "Agent",
        start: Individual | None = None,
        submit: bool = True,
        *,
        tools: Sequence["Tool"] = (),
        resume: str | None = None,
    ):
        """Start `agent` on this experiment: inside the block, `agent.prompt(text)` and
        `agent.send(text)` talk to one conversation, in a directory holding `start` (default: the
        best) that the block yields and deletes at its end. Its system prompt (`SYSTEM`) tells it the
        setting: the search, the editable files, its tools, and the task's instructions.

        The agent can read the experiment (stats, log, show, diff, grep, notes), remember what it
        learned, score its files, and (with `submit`) submit them to the archive; `tools` adds your
        own. Each prompt may spend what the limits leave, the agent is stopped once they are
        reached, and every turn is recorded. Another agent prompted inside the block answers in a
        conversation of its own that can only read. `resume` continues an earlier conversation.
        """
        task = self._task()
        parent = start if start is not None else self.best()
        directory = Path(tempfile.mkdtemp(prefix="evokit-work-"))
        session = None
        try:
            await asyncio.to_thread(task.checkout, parent.genome if parent is not None else task.seed, directory)
            parents = [parent.id] if parent is not None else []
            granted = [self._stopping(t, lambda: session) for t in (*agent_tools(self, directory, parents, submit), *tools)]
            from evokit.agents.harness import bind

            system = SYSTEM.format(
                editable=", ".join(task.editable),
                submit=SUBMIT if submit else "",
                instructions=f"\n\nThe task:\n{task.instructions}" if task.instructions else "",
            )
            async with agent.session(directory, granted, resume, system=system) as session:
                with bind(agent, _Conversation(self, session), self._read_only):
                    yield directory
        finally:
            await asyncio.to_thread(shutil.rmtree, directory, True)

    def _stopping(self, tool: "Tool", session: Callable[[], Any]) -> "Tool":
        from evokit.agents.tools import call, tool_doc, tool_name

        @functools.wraps(tool)
        async def checked(*args, **kwargs):
            try:
                return await call(tool, inspect.signature(tool).bind(*args, **kwargs).arguments)
            finally:
                if (reason := self.limits.reached()) and session() is not None:
                    session().stop(reason)

        checked.__name__, checked.__doc__ = tool_name(tool), tool_doc(tool)
        return checked

    @contextlib.asynccontextmanager
    async def _read_only(self, agent: "Agent"):
        with tempfile.TemporaryDirectory(prefix="evokit-agent-") as directory:
            async with agent.session(directory, agent_tools(self)) as session:
                yield _Conversation(self, session)

    def record(self, turn) -> str | None:
        """An agent's turn: its transcript under sessions/<session>/, and a "turn" note with its
        reply, cost, and the tools it called. Returns where the transcript is, if it has one."""
        path = None
        if turn.transcript:
            path = f"sessions/{turn.session or 'unnamed'}/{new_id()}.jsonl"
            self.storage.write(path, turn.transcript.encode())
        self.note(
            "turn", truncate(turn.text, 2000) or "", usage=turn.usage, session=turn.session,
            transcript=path, calls=list(turn.calls), error=turn.error, stopped=turn.stopped,
            **({"output": asdict(turn.output)} if turn.output is not None else {}),
        )
        return path

    def close(self) -> None:
        """Give the experiment back before the process exits, so another process can drive it."""
        with _held_lock:
            if (stack := _held.pop(self.storage.key, None)) is not None:
                stack.close()

    def discard(self, individual: Individual, reason: str) -> None:
        """A child that was created but will not be evaluated."""
        self._write({"kind": "discarded", "id": individual.id, "reason": reason})

    def write_logs(self, folder: str, logs: Mapping[str, bytes]) -> None:
        """Files kept with the experiment, e.g. an evaluation's or an agent's logs under artifacts/<id>/."""
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
        """Failed variations ("failed" notes) since the last child was created."""
        self.refresh()
        return self._failures_in_a_row

    def last_note(self, kind: str) -> str | None:
        """The text of the last note of `kind`, or None."""
        self.refresh()
        notes = self._notes.get(kind)
        return notes[-1]["text"] if notes else None

    def notes(self, kind: str, limit: int | None = None) -> list[dict]:
        """The notes of `kind`, newest first: {"text", "time", and what was stored with it}. The
        experiment's memory: what agents and recipes wrote down (e.g. "insight", "review", "turn")."""
        self.refresh()
        return list(reversed(self._notes.get(kind, [])))[:limit]

    # the run's numbers: for stop rules, supervisors, plots and `evokit status`

    def evaluations(self) -> int:
        """Versions scored and recorded, seeds included."""
        self.refresh()
        return self._counts["evaluated"]

    def drafts(self) -> list[dict]:
        """Every draft the score tool scored, oldest first: {"objectives": ..., "improved": ...}."""
        self.refresh()
        return list(self._drafts)

    def cost(self) -> float:
        """US dollars spent, everything included."""
        return self.usage()["cost_usd"]

    def elapsed(self) -> float:
        """Seconds since the run's first record."""
        self.refresh()
        return time.time() - self._start if self._start else 0.0

    def since_improvement(self, drafts: bool = True) -> int:
        """Attempts since the best last strictly improved: evaluations, and drafts unless drafts=False."""
        self.refresh()
        if drafts:
            return self._attempts - self._attempts_at_improvement
        return self._counts["evaluated"] - self._evaluations_at_improvement

    def since_note(self, kind: str) -> int:
        """Attempts (evaluations and drafts) since the last note of `kind`; all of them if there is none."""
        self.refresh()
        return self._attempts - self._attempts_at_note.get(kind, 0)

    def best_over_time(self) -> list[tuple[int, Individual]]:
        """(evaluations so far, individual) at each improvement: the run's progress curve."""
        self.refresh()
        return [(n, self._individuals[i]) for n, i in self._best_over_time]

    def records(self) -> list[dict]:
        """Every record, oldest first, as stored: for tools like compare.py."""
        if not self.storage.exists(LOG):
            return []
        return [json.loads(line) for line in self.storage.read(LOG).decode().splitlines() if line.strip()]

    # reading like git: what the CLI's log, show, diff and grep print

    def members(self) -> dict[str, Individual]:
        """The archive's current members by id, from its log."""
        return read_members(self.storage, self.archive.name if self.archive is not None else "archive")

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



def _template(archive: Archive) -> Archive:
    """`archive` with no members and no storage: what `open_experiment` attaches again."""
    template = copy.copy(archive)
    template.storage, template._members, template.metadata, template.state = None, {}, {}, {}
    return template


SYSTEM = """You are working on one version of a program in an evolutionary search run by EvoKit: \
versions are made, scored on the task's objective, and the best are kept.

Your working directory holds the version. Editable files (the genome): {editable}
Only changes to these files count. Do not modify any other file; edits to them are rejected.

Every earlier version, kept or rejected, with its score, can be read with the log, show, diff and \
grep tools: check what was already tried before repeating an idea. Earlier agents' notes are in \
notes; leave your own with remember when you learn something the code and its score do not show. \
The score tool scores your files as they are, keeping nothing.{submit}

Only the task's evaluator, run by the score tool, measures a version, with its own time limit and \
resources. Do not run the evaluator yourself, or run the program under other limits (a longer time \
budget, more starts), to measure it. Do not paste results computed outside it (precomputed answers, \
coordinates, tuned constants found by a long side run) into the files: a version must earn its score \
when the evaluator runs it.

When a tool result has "stop", the run is over: stop at once.{instructions}"""

SUBMIT = " The submit tool scores them and offers them to the archive, which keeps them if they are good enough."


class _Conversation:
    """An agent's session in an experiment: each prompt may spend what the limits leave, and its
    turn is recorded."""

    def __init__(self, experiment: Experiment, session):
        self.experiment, self.session = experiment, session

    async def prompt(self, text: str, output: type | None = None):
        from evokit.agents.harness import Turn

        events = self.session.prompt(text, self.experiment.limits.remaining_cost(), output=output)
        async with contextlib.aclosing(aiter(events)) as events:
            async for event in events:
                if isinstance(event, Turn):
                    self.experiment.record(event)
                yield event

    def send(self, text: str) -> None:
        self.session.send(text)


def open_experiment(dir: str | Path, task: "Task | None" = None) -> Experiment:
    """The experiment in `dir`, to read or to continue, with its archive. Its task is read again
    from the folder it was made from (or pass `task`); one without a task can only be read."""
    from evokit.storages import DirectoryStorage

    if not (Path(dir) / LOG).is_file():
        raise UsageError(f"no experiment in {Path(dir)}", hint="point --dir (or $EVOKIT_DIR) at an experiment directory")
    storage = DirectoryStorage(dir)
    if task is not None:
        _check_task(storage, task)
    elif storage.exists(TASK_FILE):
        recorded = json.loads(storage.read(TASK_FILE)).get("task")
        if recorded and Path(recorded).is_dir():
            task = task_for(recorded, dir)
    archive = pickle.loads(storage.read(ARCHIVE_FILE)) if storage.exists(ARCHIVE_FILE) else None
    return Experiment(storage, task, archive)


def _check_task(storage, task: "Task") -> None:
    if task.directory is None or not storage.exists(TASK_FILE):
        return
    recorded = json.loads(storage.read(TASK_FILE)).get("task")
    if recorded and Path(recorded) != task.directory:
        raise UsageError(f"{storage.root} is an experiment on the task {recorded}, not {task.directory}", hint="use another --dir")


def task_for(task_dir: str | Path, dir: str | Path) -> "Task":
    """The task in `task_dir`, for the experiment in `dir`: its fixed files are snapshotted once,
    into `dir`/env, so every process on the experiment evaluates against the same ones."""
    from evokit.task import Task

    task_dir, dir = Path(task_dir).resolve(), Path(dir).resolve()
    try:
        inside = (f"{dir.relative_to(task_dir).as_posix()}/**",)
    except ValueError:
        inside = ()
    return Task.from_directory(task_dir, environment=dir / "env", ignore=inside)


# the tools an agent gets: each takes the experiment first


def stats(experiment: Experiment) -> dict:
    """The experiment so far: evaluations, drafts, cost, attempts since the best last improved, and the best."""
    found = experiment.best()
    return {
        "evaluations": experiment.evaluations(),
        "drafts": len(experiment.drafts()),
        "cost_usd": round(experiment.cost(), 4),
        "since_improvement": experiment.since_improvement(),
        "best": _summary(found) if found else None,
    }


def log(experiment: Experiment, id: str | None = None, all: bool = False, limit: int | None = 20) -> list[dict]:
    """ID and its ancestors, nearest first, with names and scores, like `git log`; without ID, from
    the last individual the archive admitted. With --all, every evaluated individual, newest first,
    rejected ones too; `member` says whether it is still in the archive."""
    return experiment.log(experiment.lookup(id) if id else None, all=all, limit=limit)


def show(experiment: Experiment, id: str, path: str | None = None) -> Any:
    """An individual's name, description, parents, evaluation and files; with PATH, that file."""
    return experiment.show(experiment.lookup(id), path)


def diff(experiment: Experiment, a: str, b: str | None = None) -> str:
    """What changed from A to B, as a unified diff; with only A, from its parent to A."""
    return experiment.diff(experiment.lookup(a), experiment.lookup(b) if b else None)


def grep(experiment: Experiment, pattern: str, limit: int = 20) -> list[dict]:
    """Individuals whose name, description or files match a regex, rejected ones too, so you can
    see what was already tried and how it scored."""
    return experiment.grep(pattern, limit)


def notes(experiment: Experiment, kind: str = "insight", limit: int = 20) -> list[dict]:
    """Notes left by earlier agents and by the experiment, newest first: what was learned, what to avoid."""
    return experiment.notes(kind, limit)


def remember(experiment: Experiment, text: str) -> str:
    """Leave a note for later agents: something you learned that is not in the code or its score."""
    experiment.note("insight", text)
    return "noted"


READ_TOOLS = (stats, log, show, diff, grep, notes)
"""The tools that read an experiment."""


def agent_tools(experiment: Experiment, directory: Path | None = None, parents: Sequence[str] = (), submitting: bool = True) -> list["Tool"]:
    """What an agent can do: read the experiment and remember; with its working `directory`, score
    the files there, keeping nothing; with `submitting`, submit them, scored and offered to the
    archive. A version descends from `parents`, then from the last version the archive kept."""
    tools: list = [functools.partial(tool, experiment) for tool in (*READ_TOOLS, remember)]
    if directory is None:
        return tools
    task, archive, lineage = experiment._task(), experiment.archive, list(parents)

    async def score() -> dict:
        """Score the files in your working directory as they are now, without keeping them."""
        draft = Individual.new(await asyncio.to_thread(task.read, directory), lineage, {"operator": "draft"})
        evaluation = await experiment.evaluate(draft)
        improved = archive is not None and improves(archive, draft.with_evaluation(evaluation), experiment.best())
        experiment.note(
            "draft", _scores(evaluation), {**evaluation.usage, "draft_evals": 1},
            objectives=dict(evaluation.objectives), improved=improved,
        )
        return _evaluation(evaluation)

    tools.append(score)
    if not submitting or archive is None:
        return tools

    async def submit(name: str, description: str = "") -> dict:
        """Score the files in your working directory and offer them to the archive: `kept` says
        whether they were kept. `name` is one line saying what this version is, `description` what
        changed and why."""
        genome = await asyncio.to_thread(task.read, directory)
        best = experiment.best()
        if best is not None and genome == best.genome:
            raise UsageError("nothing changed since the best version", hint="edit the files first")
        if len(name) > NAME_LIMIT or "\n" in name:
            name, rest = title_and_body(name)
            description = "\n\n".join(p for p in (rest, description) if p)
        child = Individual.new(genome, lineage, {"operator": "agent"}, name=name, description=truncate(description, 4000))
        result = await experiment.submit(child)
        if result.info["admitted"]:
            lineage[:] = [result.id]
        return {"id": result.id, "kept": result.info["admitted"], **_evaluation(result.evaluation)}

    tools.append(submit)
    return tools


def _summary(individual: Individual) -> dict:
    out: dict[str, Any] = {"id": individual.id, "name": individual.name, "parents": list(individual.parents)}
    if individual.evaluation is not None:
        out["objectives"] = dict(individual.evaluation.objectives)
    return out


def _evaluation(evaluation: Evaluation) -> dict:
    return {"objectives": dict(evaluation.objectives), "feedback": truncate(evaluation.feedback, FEEDBACK_LIMIT)}


def _scores(evaluation: Evaluation) -> str:
    return ", ".join(f"{k}={v:.6g}" for k, v in evaluation.objectives.items()) or "failed"

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


__all__ = ["Experiment", "Limits", "StopRule", "agent_tools", "diff_genomes", "genome_text", "open_experiment", "task_for"]
