# Primitives

EvoKit gives the pieces of an evolutionary algorithm as plain objects and functions. A recipe is
a loop that stitches them together ([recipes](recipes.md)). There is no base class to inherit
from, no event bus and no hooks: to change what a recipe does, edit its loop or copy it.

The code below is condensed from the implementation; the files it names are the full versions.

!!! warning "Work in progress"
    These interfaces are still changing. Names and signatures here match the code at the time
    of writing; the docstrings in `evokit/` are the source of truth.

Every recipe has the same core:

```python
import evokit as ek

task = ek.Task(seed=genome, evaluate=score)                        # what evolves, how it is scored
experiment = task.create_experiment("experiments/0", max_cost=5)   # one search; seeded or continued
while not experiment.limits.reached():
    parent, *others = experiment.archive.sample(3)                 # selection
    if child := await experiment.create(vary(parent, others)):     # variation, recorded with its cost
        await experiment.submit(child)                             # evaluation and survival, recorded
```

`create` and `submit` are the only two ways in. Everything they do is written to the
experiment's history, so a recipe never records anything by hand.

## Types

`evokit/records.py` holds plain data, with no behaviour:

```python
@dataclass(frozen=True)
class Individual:
    id: str
    genome: Any                        # any JSON value, or a type with to_json/from_json (e.g. Files)
    parents: tuple[str, ...] = ()
    info: Mapping[str, Any] = {}       # e.g. {"operator": "diff", "admitted": True}
    evaluation: Evaluation | None = None
    name: str = ""                     # one line saying what this version is, like a commit subject
    description: str = ""              # what changed and why, like a commit body

    @classmethod
    def new(cls, genome, parents=(), info=None, *, name="", description="") -> Individual: ...
    def with_evaluation(self, evaluation) -> Individual: ...   # copies; records are frozen
    def with_info(self, **info) -> Individual: ...
    def objective(self, name) -> float | None: ...


@dataclass(frozen=True)
class Evaluation:
    objectives: dict[str, float] = {}  # empty if the program crashed, timed out or was invalid
    feedback: str = ""                 # what the next attempt should know: a traceback, failing tests
    usage: dict[str, float] = {}       # what evaluating cost: cost_usd, tokens, seconds, ...
    logs: Mapping[str, bytes] = {}     # files kept under artifacts/<id>/, not in the history
```

An evaluation with no objectives is a failure. There is no separate status field.

## Task

`evokit/task.py`. A task is what evolves and how it is scored:

```python
task = ek.Task(
    seed=1.0,                          # the first genome
    evaluate=score,                    # score(individual) -> Evaluation, sync or async
    instructions="Make it close to 3", # what agents and models are told
    objective="score",                 # what ranked archives rank by
    minimize=False,
)
task = ek.Task.from_directory("examples/circle_packing")   # a task folder: see tasks.md
```

`evaluate` is the one place scores come from. `@ek.evaluator(timeout="10m")` makes a function
async and never raising: a crash, a timeout or a wrong return type becomes an evaluation with no
objectives. The experiment wraps `evaluate` this way if you did not.

A task folder's genome is a `Files` value (a set of files), and the rest of the folder is a
read-only environment: `task.checkout(genome, dir)` writes both into a directory, and
`task.read(dir)` reads the genome back, raising `EnvironmentModified` if the fixed files were
edited. Agents work in such a checkout. A non-file genome is checked out as `genome.json`.

## Experiment

`evokit/experiment.py`. One search on a task: its history, its archive and its limits.

```python
experiment = task.create_experiment(
    "experiments/0",                   # a directory (DirectoryStorage); None keeps it in memory
    archive=ek.BestArchive(),          # the default
    max_cost=5.0,                      # US dollars, everything included
    max_evaluations=None,
    max_failures=10,                   # failed variations in a row
    stop=[lambda e: "plateau" if e.since_improvement() >= 30 else None],
)
experiment = ek.open_experiment("experiments/0")   # read one back (and continue it, given its task)
```

A new experiment scores the seed into the archive. If the seed fails its evaluation, it raises
`UsageError` and records nothing. On an existing directory, `create_experiment` continues: the
archive replays its own log, and children that a crash left unevaluated are submitted again. A
recipe never knows whether it is continuing.

### What goes in

```python
child = await experiment.create(make)        # make: an awaitable of (child, usage)
child = await experiment.submit(individual)  # scored by task.evaluate, offered to the archive
experiment.note("review", text, usage)       # anything else worth keeping
experiment.discard(individual, "not novel")  # created, but will not be evaluated
```

- `create` awaits a variation function and records the child as created, with its cost. If the
  variation raises `OperatorError(reason, usage, logs)`, it records a `failed` note with the cost
  (its logs go under `failed/<id>/`) and returns `None`.
- `submit` scores the individual with the task's `evaluate`, never with a score the caller
  passes, offers it to the archive, and records both. It records the individual as created first
  if `create` did not. It returns the individual with its evaluation and `info["admitted"]`.

**Every cost is in the experiment.** Creating, evaluating, failed variations, agent turns and
notes each record their own `usage`, so `experiment.cost()` is the cost of the whole search and
the limits hold across restarts.

### Limits

`experiment.limits.reached()` returns why the experiment should stop, or `None`. It checks
`max_evaluations` (seeds included), `max_cost`, `max_failures` and every `stop` rule. A stop
rule is `rule(experiment) -> str | None`. `limits.remaining_cost()` is what is left of
`max_cost`, which agents get as their per-turn budget.

### What can be read

```python
experiment.best()                    # the archive's best
experiment.cost(), experiment.usage()
experiment.counts()                  # "created", "evaluated", "added", "discarded", each note kind
experiment.evaluated()               # everything evaluated, in order
experiment.admitted()                # what the archive admitted at the time
experiment.pending()                 # created, not yet evaluated or discarded
experiment.individual(id)
experiment.notes("insight")          # newest first; also last_note(kind)
experiment.since_improvement()       # attempts (evaluations and drafts) since the best improved
experiment.since_note("migrate")     # attempts since the last note of a kind: for schedules
experiment.best_over_time()          # the progress curve
experiment.log(), .show(i), .diff(a, b), .grep(pattern)   # git-style reading, see reading-runs.md
```

Totals are kept in memory and brought up to date with what other processes appended before
every read, so checking a budget every iteration reads only the new lines.

Only one process drives an experiment at a time: the first write takes a driver lock, kept until
the process exits or calls `experiment.close()`. Reading never takes it.

## Archives

`evokit/archive.py` and `evokit/archives/`. The archive is the population: it decides who
survives (`add`), and offers its members for selection (`sample`, `top`).

| Archive | Keeps | Default `sample` |
| --- | --- | --- |
| `BestArchive(ties="reject")` | the single best; `ties="accept"` lets equal scores replace it | `best` |
| `SimplerBestArchive(size="lines")` | the single best; a tie wins only if it is smaller by `size` | `best` |
| `PopulationArchive(size)` | the top `size` (truncation) | `uniform` |
| `UnboundedArchive()` | everyone with a score (DGM's open-ended archive) | `uniform` |
| `LatestArchive()` | only the newest with a score | `best` |
| `NoveltyArchive(behaviour, k=15)` | individuals far from their `k` nearest members, with an adaptive threshold | `uniform` |
| `IslandArchive(islands=4, features=("lines", "diversity"))` | islands of MAP-Elites grids; `migrate(top)` copies each island's best to the next | `uniform`, one island per call |

Ranked archives take `objective=` and `minimize=`; without them they use the task's.

```python
parent, *others = archive.sample(3)                         # the archive's own strategy
parents = archive.sample(10, strategy=tournament(3))        # or any other
(best,) = archive.top(1)
```

### Archives save themselves

Every archive has **members** by id, **metadata** per member (any JSON: an island and a cell, a
behaviour descriptor, a novelty score), and **state** of its own (any JSON: a threshold, a
counter). A subclass writes `add`, its rule for who gets in, and changes the archive only
through `insert`, `remove` and `set_state`:

```python
class BestArchive(RankedArchive):
    def add(self, individual):
        if self.score(individual) is None:          # failed, or no such objective
            return False
        best = self.best()
        if best is not None and not self.admits(individual, best):
            return False
        if best is not None:
            self.remove(best.id)
        self.insert(individual)
        return True
```

The base class appends each change to `<name>/changes.jsonl` in the experiment's storage. An
archive attached to an existing experiment replays that log, so it comes back exactly as it was,
including what `IslandArchive.migrate` copied. Nobody writing an archive or a recipe handles
saving. The log is also the archive's history, so a tool can show what it held at any point.

## Strategies

`evokit/strategies.py`. A strategy is a plain function:

```python
def strategy(candidates: Sequence[Individual], n: int, archive: Archive) -> list[Individual]: ...
```

The archive offers its candidates, best first if it ranks (an `IslandArchive` one island at a
time), and the strategy draws from `archive.rng`. EvoKit ships `best`, `uniform` and
`tournament(size)`. Algorithm-specific strategies live in their recipes: DGM's `proportional`,
OpenEvolve's `islands`, Shinka's `parents`. A strategy keeps no state; it reads what it needs
from the archive (for example, child counts from the members' parents).

## Variation functions

`evokit/operators/`. A variation function makes one child and returns `(child, usage)`, or
raises `OperatorError(reason, usage)`. It is what `experiment.create` awaits:

| Function | For |
| --- | --- |
| `rewrite_text(model, parent, others, instructions=)` | `str` genomes: a full rewrite in a fenced block |
| `diff(model, parent, others, task=)` | `Files` genomes: SEARCH/REPLACE diffs, respecting EVOLVE-BLOCK markers and the editable files |
| `rewrite_files(model, parent, others, task=)` | `Files` genomes: complete new contents of the changed files |
| `self_modify(parent, experiment)` | DGM: the parent agent diagnoses its own logs, then rewrites itself |
| `meta_agent_modify(parent, experiment)` | Hyperagents: the parent's meta agent edits the whole repository |
| `generate(model, prompt, parents, parse, operator)` | the building block for your own |

Any async function returning `(child, usage)` works, including a plain mutation with no model
(see `recipes.ga`).

The models are `AnthropicModel(model, price=)`, `OpenAIModel(model, base_url=, price=)` (any
OpenAI-compatible endpoint) and `FunctionModel(fn)` (for tests and custom clients). `price` is
US dollars per million input and output tokens, so their usage carries `cost_usd`.

## Agents

`evokit/agents/`. An `Agent` is a coding harness EvoKit can drive: `ClaudeCode`, `CommandAgent`
(any command run in the agent's directory), `ReplAgent` (Python functions in a REPL), or your
own. The agent layer knows nothing about evolution: it opens sessions in a directory with tools
and a system prompt, and `session.prompt(text, output=Dataclass)` streams `ToolCall` events and
ends with a `Turn`, whose `output` is what the agent passed to its `finish` tool.

`experiment.start(agent)` puts an agent to work on an experiment:

```python
async with experiment.start(agent, start=parent, submit=True) as directory:
    async for event in agent.prompt("Improve it; submit what is better."):
        if experiment.since_improvement() > 5:
            agent.send("Try a different data structure.")   # reaches it at its next step
```

Inside the block:

- `directory` is a checkout of `start` (default: the best), deleted when the block ends.
- The agent's tools are `stats`, `log`, `show`, `diff`, `grep`, `notes` and `remember` over the
  experiment, `score` (scores its files as they are and records a `draft` note, keeping
  nothing), and, with `submit=True`, `submit` (scores them and offers them to the archive).
  `tools=` adds your own. Shell agents reach them as an `evokit` command on their `PATH`.
- `agent.prompt` and `agent.send` talk to one conversation. Each prompt may spend what the
  limits leave, the agent is stopped once a limit is reached, and every turn is recorded as a
  `turn` note with its transcript under `sessions/`.
- `agent.fresh()` is the same agent in a conversation of its own that can only read the
  experiment, for example a reviewer.

## Storage

`evokit/storage.py` and `evokit/storages/`. Storage is bytes at paths. It knows nothing about
individuals, archives or costs, so a backend implements only these:

```python
class Storage:
    def write(self, path, data: bytes) -> None: ...
    def read(self, path) -> bytes: ...
    def exists(self, path) -> bool: ...
    def list(self, prefix) -> list[str]: ...
    def append(self, path, data: bytes) -> None: ...        # safe for concurrent writers
    def local(self, path) -> ContextManager[Path]: ...       # a real folder, for programs that need one
    def lock(self, name) -> ContextManager[None]: ...        # across processes; RunBusy if held
```

`DirectoryStorage(dir)` and `MemoryStorage()` ship today; remote storages are on the README's
list. Everything above `Storage` picks its own paths:

```text
experiments/0/
  history/records.jsonl    the history: one record per line (created, evaluated, discarded, note)
  archive/changes.jsonl    the archive's own change log
  archive.pickle           the archive with no members, so open_experiment can rebuild it
  task.json                which task folder the experiment belongs to
  blobs/ab/cdef...         genome contents, content-addressed, so equal content is stored once
  artifacts/<id>/          an evaluation's logs (Evaluation.logs, files written to $EVOKIT_WORKDIR)
  failed/<id>/             the logs of a variation that failed (its note names the folder)
  sessions/<session>/      agents' transcripts, one file per turn
  env/                     the task's fixed files, snapshotted once (experiments made from the CLI)
```

A genome is any JSON value, or any type with `to_json(put_blob)` and `from_json(data, get_blob)`,
where `put_blob` writes to `blobs/` under the content's hash. That covers a string, a vector, a
whole repository as `Files`. The storage never knows which.

## Open questions

The README's TODO list has the open design work: several processes on one experiment, a sandbox
for agents and evaluators, re-evaluation, evaluation caching, sampling that sees what is in
flight, remote storages and multi-objective archives.
