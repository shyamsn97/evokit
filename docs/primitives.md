# Primitives

EvoKit provides the pieces of an evolutionary algorithm, as functions and components. A
recipe is a plain loop that stitches them together ([recipes](recipes.md)). There is no
`Experiment` class, no base class to inherit from, no event bus, and no hooks. To change what
a recipe does, you edit its loop or copy it. The code below is condensed from the
implementation; the files it names are the full versions.

Every recipe has the same core:

```python
import evokit

@evokit.eval_fn(timeout="10m")
def score(individual) -> evokit.Evaluation: ...

asked = await evokit.ask(storage, archive, strategy, operator)
evaluations = await asyncio.gather(*map(score, asked.individuals))
admitted = await evokit.tell(storage, archive, asked.individuals, evaluations)
```

`ask` and `tell` take the storage first and are plural. Evaluating needs no storage: the eval
function scores one individual and returns its files as `Evaluation.logs`, which `tell` stores.
What goes into the archive first is the caller's input.

## Types

```python
@dataclass(frozen=True)
class Individual:
    id: str
    genome: Genome
    parents: tuple[str, ...]
    info: dict
    evaluation: Evaluation | None      # set with individual.with_evaluation(...), which returns a copy
    name: str                          # one line saying what this version is, set by its operator
    description: str                   # what changed and why


@dataclass(frozen=True)
class Evaluation:
    objectives: dict[str, float]       # empty if the program crashed, timed out or was invalid
    feedback: str = ""                 # what the next attempt should know, e.g. a traceback or failing tests
    usage: dict[str, float] = field(default_factory=dict)    # what evaluating cost


@dataclass(frozen=True)
class Selection:                       # what a strategy picks for one child-to-be
    parents: tuple[Individual, ...]
    inspirations: tuple[Individual, ...] = ()


@dataclass(frozen=True)
class Offspring:                       # what an operator returns
    children: tuple[Individual, ...]
    usage: dict[str, float]            # what creating them cost
    logs: Mapping[str, bytes]          # e.g. an agent's transcript; ask stores them in each child's artifacts


@dataclass(frozen=True)
class Failure:                         # an operator call that produced nothing
    selection: Selection
    reason: str
    usage: dict[str, float]


@dataclass(frozen=True)
class Asked:
    individuals: tuple[Individual, ...]
    failures: tuple[Failure, ...] = ()
    exhausted: bool = False            # nothing to select right now
```

## Components

The components are the actual algorithms:

| Interface | Implementations |
| --- | --- |
| `Archive(storage=None)`, `add(individual) -> bool`, `members()`, `metadata`, `state`, `best()`, `get(id)` | `BestArchive`, `SimplerBestArchive`, `PopulationArchive`, `RankedArchive`, `UnboundedArchive`, `LatestArchive`, `NoveltyArchive`, `IslandArchive` (with `niches()` and `migrate(top)`) |
| `Strategy.select(archive, n) -> list[Selection]` | `Greedy`, `Tournament`, `ProportionalSelection`, `IslandSampler` |
| `Operator.create(selection, archive) -> Offspring`; raises `OperatorError(reason, usage)` | `DiffOperator`, `FileRewriteOperator`, `CodingAgentOperator`, `SelfModify`, `MetaAgentModify`, `Choice` |
| `eval_fn(individual) -> Evaluation`, sync or async; `@ek.eval_fn(timeout=, executor=)` makes it async and turns crashes and timeouts into failed evaluations | `ScriptEvaluator`, `with_lines`, your function |

Strategies read everything they need from the archive. For example, `ProportionalSelection`
counts a member's children from the parents of the other members. So strategies keep no
state that has to be saved.

### Archives save themselves

Every archive has three things, and nothing else is assumed:

- **members**, by id;
- **metadata** per member: any JSON the archive wants to keep about it, such as an island and a
  cell, a behaviour descriptor, a novelty score, or a crowding distance;
- **state** of its own: any JSON, such as a novelty threshold or a counter.

A subclass writes `add`, its rule for who gets in. It changes the archive only through
`insert`, `remove` and `set_state`. The base class appends each call to its own log in the
storage, `archive/changes`, and when an archive is built on an existing run, it replays that
log. Nobody writing an archive or a recipe handles saving.

```python
class Archive:
    def __init__(self, storage=None, name="archive"):
        self.storage, self.log = storage, f"{name}/changes.jsonl"
        self._members: dict[str, Individual] = {}
        self.metadata: dict[str, dict] = {}
        self.state: dict = {}
        if storage is not None and storage.exists(self.log):    # an existing run: replay the log, in order
            for line in storage.read(self.log).splitlines():
                change = json.loads(line)
                if "state" in change:
                    self.state = change["state"]
                elif "removed" in change:
                    self._members.pop(change["removed"], None)
                    self.metadata.pop(change["removed"], None)
                else:
                    individual = decode(change["inserted"], storage)
                    self._members[individual.id], self.metadata[individual.id] = individual, change["metadata"]

    def insert(self, individual, **metadata):               # inserting a member again replaces its metadata
        self._members[individual.id], self.metadata[individual.id] = individual, metadata
        self._append({"inserted": encode(individual, self.storage), "metadata": metadata})

    def remove(self, id):
        del self._members[id], self.metadata[id]
        self._append({"removed": id})

    def set_state(self, **state):
        self.state = {**self.state, **state}
        self._append({"state": self.state})

    def _append(self, change):
        if self.storage is not None:
            self.storage.append(self.log, (json.dumps({**change, "time": time.time()}) + "\n").encode())

    def members(self) -> list[Individual]:
        return list(self._members.values())

    def add(self, individual) -> bool:                      # the archive's rule: the one method a subclass writes
        raise NotImplementedError
```

Three archives with nothing in common, written the same way:

```python
class BestArchive(Archive):                                 # one member, no metadata
    def add(self, individual):
        best = self.best()
        new = score(individual)                             # None if it has no score, e.g. it crashed
        if new is None or (best and score(best) >= new):
            return False
        if best:
            self.remove(best.id)
        self.insert(individual)
        return True


class NoveltyArchive(Archive):                              # k-nearest neighbours, no structure, an adaptive threshold
    def add(self, individual):
        behaviour = individual.evaluation.objectives["behaviour"]
        others = [m["behaviour"] for m in self.metadata.values()]
        novelty = mean(sorted(distance(behaviour, o) for o in others)[: self.k]) if others else math.inf
        threshold = self.state.get("threshold", self.initial_threshold)
        if novelty < threshold:
            self.set_state(threshold=threshold * 0.95)
            return False
        self.insert(individual, behaviour=behaviour, novelty=novelty)
        self.set_state(threshold=threshold * 1.05)
        return True


class IslandArchive(Archive):                               # islands of MAP-Elites grids
    def add(self, individual):
        island = self.metadata[individual.parents[0]]["island"] if individual.parents else 0
        cell = self.cell(individual, island)
        ...                                                 # keep the cell's best cell_size members
        self.insert(individual, island=island, cell=cell)
        return True

    def niches(self):                                       # only archives with niches have niches()
        grouped = defaultdict(list)
        for id, m in self.metadata.items():
            grouped[(m["island"], tuple(m["cell"]))].append(self._members[id])
        return dict(grouped)
```

A reloaded archive is the exact archive from before, including the copies
`IslandArchive.migrate` makes. The log doesn't depend on the run's history: each insert carries
the encoded individual, and genome contents go to the shared blobs, so no genome is stored
twice. The log is also the history of the archive, so `compare.py` can show what the archive
held at any point in the run. A run with two archives gives each a `name=`, and each gets its
own folder.

## Storage

Storage reads and writes bytes at paths. It knows nothing about individuals, archives, logs or
costs, so a backend only implements these methods:

```python
class Storage:
    def write(self, path: str, data: bytes) -> None: ...
    def read(self, path: str) -> bytes: ...
    def exists(self, path: str) -> bool: ...
    def list(self, prefix: str) -> list[str]: ...                  # paths under prefix, sorted
    def append(self, path: str, data: bytes) -> None: ...         # add to the end of the file at path

    @contextmanager
    def local(self, path: str) -> Iterator[Path]: ...             # a real folder for path, for code that needs one
```

`local` exists because evaluators and agents run programs that need a real folder. A
`DirectoryStorage` returns the folder itself. A remote storage downloads what's there into a
temporary folder, and uploads it again when the block ends.

| Backend | `append` | `local` |
| --- | --- | --- |
| `DirectoryStorage("runs/mar5")` | appends to the file, under a file lock | the folder itself |
| `MemoryStorage()` | appends to bytes in a dict | a temporary folder, read back at the end |
| `S3Storage("s3://bucket/runs/mar5")` (later) | one new object per call, under `path/`; `read` joins them in order | download, then upload |
| `ModalStorage(volume)` (later) | appends to the file on the mounted volume | the folder on the volume |

Everything above `Storage` picks its own path and format:

```
runs/mar5/
  history/records.jsonl      evokit.history: one JSON record per line
  archive/changes.jsonl      the Archive base class: one change per line
  blobs/ab/cdef...           genome encoding: content-addressed, so equal content is stored once
  artifacts/<id>/            tell: the evaluation's logs (Evaluation.logs);
                             ask: the operator's logs for the child (agent.jsonl: the agent's session)
  failed/<id>/               ask: the logs of an operator call that failed (its note names the folder)
```

Individuals are saved with `encode(individual, storage)` and `decode(data, storage)`. A genome
is any JSON value, or any type with `to_json(put_blob)` and `from_json(data, get_blob)`, where
`put_blob` writes to `blobs/` under its content hash. That covers a string, a whole repository
as `Files`, a model checkpoint. The storage never knows which.

## History

The history of a run is written inside the functions that make it, so a recipe never records
anything by hand:

- `ask` records the children it creates and the operator calls that fail, with their cost.
- `tell` records each evaluated individual, and whether the archive admitted it.

It is read through functions in `evokit.history`:

```python
from evokit import history

history.usage(storage)              # Counter: cost_usd, tokens, ...: the sum over everything recorded
history.counts(storage)             # Counter: "created", "evaluated", "added", and each note kind
history.evaluated(storage)          # everything evaluated so far, in order, with their evaluations
history.admitted(storage)           # the ones the archive admitted at the time
history.pending(storage)            # created, but not yet evaluated or discarded
history.individual(storage, id)
history.last_note(storage, kind)
history.records(storage)            # everything, for tools like compare.py
```

A recipe writes to the history in only two cases:

```python
history.note(storage, kind, text="", usage=None)     # anything else worth keeping: a redirect, a summary, an LLM judge's cost
history.discard(storage, individual, reason)         # created, but will not be evaluated
```

The functions keep running totals in memory, so a recipe can check its budget every iteration
without reading the whole log again.

**Every cost is in the history.** Creating a child, evaluating it, a failed operator call and
a note all record their own cost. So `history.usage(storage)["cost_usd"]` is always the cost of
the whole run, without a counter in the recipe.

**Children are recorded when they are created, before they are evaluated.** After a crash,
`ask` hands them back first, so their creation cost isn't paid twice.

## The functions

```python
async def ask(storage, archive, strategy, operator, n=1) -> Asked:
    """Up to n individuals to evaluate. Children a crash left unevaluated come first; otherwise select n
    times and create from each selection. Every creation and failure is recorded with its cost."""
    if pending := history.claim_pending(storage, n):
        return Asked(tuple(pending))
    selections = await strategy.select(archive, n)
    if not selections:
        return Asked((), exhausted=True)
    individuals, failures = [], []
    for selection in selections:
        try:
            offspring = await operator.create(selection, archive)
        except OperatorError as exc:
            failures.append(Failure(selection, exc.reason, exc.usage))
            history.note(storage, "failed", exc.reason, exc.usage)
            continue
        history.record_created(storage, offspring.children, tuple(p.id for p in selection.parents), offspring.usage)
        individuals += offspring.children
    return Asked(tuple(individuals), tuple(failures))


def eval_fn(fn=None, *, timeout=None, executor=None):
    """Make fn(individual) -> Evaluation, sync or async, async and never raising."""
    def wrap(fn):
        async def wrapped(individual):
            try:
                call = fn(individual) if is_async(fn) else loop.run_in_executor(executor, fn, individual)
                evaluation = await asyncio.wait_for(call, timeout)
            except asyncio.TimeoutError:
                return Evaluation(feedback=f"evaluation timed out after {timeout}")
            except Exception:
                return Evaluation(feedback=traceback.format_exc())
            return evaluation
        return wrapped
    return wrap if fn is None else wrap(fn)


async def tell(storage, archive, individuals, evaluations) -> list[Individual]:
    """Give each individual its evaluation, store its logs, add it to the archive and record it.
    Returns the ones the archive admitted."""
    admitted = []
    for individual, evaluation in zip(individuals, evaluations, strict=True):
        write_logs(storage, f"artifacts/{individual.id}", evaluation.logs)
        individual = individual.with_evaluation(replace(evaluation, logs={}))
        added = archive.add(individual)
        history.record_evaluated(storage, individual, added=added)
        if added:
            admitted.append(individual)
    return admitted

```

`history.record_created`, `history.record_evaluated` and `history.claim_pending` are how these
functions write the history. A recipe only calls them if it replaces `ask` or `tell`.

A recipe takes its input and returns its output. Three things are not the algorithm's job:

- **The starting individuals.** The caller puts them in the archive, with the same functions:

  ```python
  archive = BestArchive(storage=storage)
  if not archive.members():                                   # a fresh run; a continued one restores itself
      seed = Individual.new(project.seed_genome())
      await evokit.tell(storage, archive, [seed], [await score(seed)])
  ```

  `tell` only needs an evaluation, not where it came from. A seed whose score is already known,
  such as a published baseline, is never evaluated:

  ```python
  await evokit.tell(storage, archive, [Individual.new(genome)], [Evaluation({"score": 0.52})])
  ```

- **Continuing a stopped run.** An archive built with `storage=` restores itself. The recipe
  never knows whether it is continuing.
- **Running several attempts at once.** That is `asyncio.gather` over copies of the loop. The
  archive already has members when the recipe starts, so no worker starts on an empty archive.

A recipe that needs something unusual between selecting and creating calls `strategy.select`
and `operator.create` itself, then `history.record_created`. `ask` is only those calls plus the
recording.

## What goes away

| Today | Here |
| --- | --- |
| `Experiment` with `ask` / `tell` / `run(workers, max_*)` | `evokit.ask`, an `@evokit.eval_fn`, `evokit.tell`, and a loop (`asyncio.gather` for several workers) |
| `ask() -> Individual \| None`, `tell(individual) -> Individual` | `ask` returns `Asked`; `tell` returns the admitted individuals |
| `transaction`, `_encode`, `keep=`, `state_dict` on components | the history log; archives built with `storage=` save and restore themselves |
| `Fitness(status, objectives, constraints, feedback, usage)` and `.ok` | `Evaluation(objectives, feedback, usage)`: no objectives means it failed |
| `_charge`, `_used`, the discard streak, `in_flight` | `history.usage(storage)` and `history.counts(storage)` |
| the queue of extra children, the seed counter, `_seeds_in_flight`, the select lock | `ask` hands back unevaluated children first; the caller seeds the archive before the recipe starts |
| `create_individuals(selected, archive) -> list`, with cost on `children[0].info` | `create(selection, archive) -> Offspring` |
| `select_individuals(archive, n) -> list[tuple]` | `select(archive, n) -> list[Selection]`, with parents and inspirations kept apart |
| `strategy.tell` / `operator.tell` | none: strategies read the archive; recipes derive any other state from the storage |
| migration inside `IslandArchive.add` | `IslandArchive.migrate(top)`, called by the recipe |
| recipes as configuration of `Experiment` | recipes as loops: copy one to change it |

## Open questions

- **The CLI rebuilds a run's components from its task folder.** The first command on a run
  directory (with `--task`) saves the folder at `run/task.json`, and every later command calls
  `load_task(folder).open(storage)` again (see [tasks.md](tasks.md)). So the CLI can run the
  generic loop (`evokit run`) but not a recipe's own loop. A `run: {recipe: avo}` key could
  name one.
- **Several processes on one run.**
  - Each backend's `append` must be safe when two processes append at once: a file lock for
    `DirectoryStorage`, one object per append for S3.
  - `claim_pending` must claim in storage, not in memory.
  - A process must read new records before it selects.
- **A sandbox.** Self-modifying agents run code they wrote themselves. The same sandbox should
  wrap both `eval_fn` and the agents inside operators, for example
  `CommandAgent(..., sandbox=Docker(image))`.
- **Re-evaluation.** Two cases need it: noisy benchmarks, and re-running archived agents on a
  new domain, as Hyperagents does to measure transfer. `tell` should accept an
  already-archived individual with a new evaluation, so `archive.add` replaces it and the storage
  records a second `Evaluated` for the same id.
