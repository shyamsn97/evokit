# Primitives, and recipes that own the loop

**Status: implemented.** This replaces `Experiment` and the earlier events proposal. The code
below is condensed from the implementation; the files it names are the full versions.

EvoKit provides the pieces of an evolutionary algorithm, as functions and components. A
recipe is a plain loop that stitches them together. There is no `Experiment` class, no base
class to inherit from, no event bus, and no hooks. To change what a recipe does, you edit its
loop or copy it.

Every recipe has the same core:

```python
import evokit

asked = await evokit.ask(storage, archive, strategy, operator)
evaluated = await evokit.evaluate(storage, asked.individuals, eval_fn=eval_fn)
admitted = await evokit.tell(storage, archive, evaluated)
```

Every function takes the storage first, and `ask`, `evaluate` and `tell` are all plural. The
library provides only these three; what goes into the archive first is the caller's input.

## Types

```python
@dataclass(frozen=True)
class Individual:
    id: str
    genome: Genome
    parents: tuple[str, ...]
    info: dict
    evaluation: Evaluation | None      # set with individual.with_evaluation(...), which returns a copy


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
| `eval_fn(individual, workdir) -> Evaluation`, sync or async | `ScriptEvaluator`, `with_lines`, your function |

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
  artifacts/<id>/            evaluate: each individual's evaluation folder, through storage.local
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


async def evaluate(storage, individuals, *, eval_fn, timeout=None, executor=None) -> list[Individual]:
    """Evaluate each individual concurrently, in its artifact folder. A crash or a timeout becomes an evaluation with no objectives."""
    async def one(individual):
        with storage.local(f"artifacts/{individual.id}") as workdir:
            try:
                evaluation = await call_eval_fn(eval_fn, individual, workdir, timeout=timeout, executor=executor)
            except Exception as exc:
                evaluation = Evaluation({}, feedback=f"{type(exc).__name__}: {exc}")
        return individual.with_evaluation(evaluation)

    return list(await asyncio.gather(*(one(i) for i in individuals)))


async def tell(storage, archive, individuals) -> list[Individual]:
    """Add evaluated individuals to the archive and record them. Returns the ones the archive admitted."""
    admitted = []
    for individual in individuals:
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
      seeds = [Individual.new(project.seed_genome())]
      await evokit.tell(storage, archive, await evokit.evaluate(storage, seeds, eval_fn=eval_fn))
  ```

  `tell` only needs individuals that carry an evaluation, not where it came from. A seed whose
  score is already known, such as a published baseline, skips `evaluate`:

  ```python
  seed = Individual.new(genome).with_evaluation(Evaluation({"score": 0.52}))
  await evokit.tell(storage, archive, [seed])
  ```

- **Continuing a stopped run.** An archive built with `storage=` restores itself. The recipe
  never knows whether it is continuing.
- **Running several attempts at once.** That is `asyncio.gather` over copies of the loop. The
  archive already has members when the recipe starts, so no worker starts on an empty archive.

A recipe that needs something unusual between selecting and creating calls `strategy.select`
and `operator.create` itself, then `history.record_created`. `ask` is only those calls plus the
recording.

## Recipes

Each recipe is one file, and each is a complete loop.

### autoresearch

[autoresearch](https://github.com/karpathy/autoresearch) is one lineage:

- edit `train.py`;
- train for a fixed time;
- keep the change if `val_bpb` went down, else revert;
- log every attempt to `results.tsv`;
- never stop.

```python
# evokit/recipes/autoresearch.py
async def autoresearch(project, eval_fn, agent, *, storage, archive, instructions, objective="val_bpb",
                       results="results.tsv"):
    strategy = Greedy()
    operator = CodingAgentOperator(agent, project, instructions=instructions)

    with open(results, "a") as tsv:
        while True:
            asked = await evokit.ask(storage, archive, strategy, operator)
            evaluated = await evokit.evaluate(storage, asked.individuals, eval_fn=eval_fn)
            admitted = await evokit.tell(storage, archive, evaluated)
            for child in evaluated:
                status = "keep" if child in admitted else "discard" if child.evaluation.objectives else "crash"
                tsv.write(row(child, status, objective))
            tsv.flush()
```

```python
project = Project(".", genome=["train.py"])
eval_fn = with_lines(ScriptEvaluator(project, "uv run train.py", timeout="10m"))
storage = DirectoryStorage("runs/mar5")
archive = SimplerBestArchive(objective="val_bpb", minimize=True, size="lines", storage=storage)
if not archive.members():
    seeds = [Individual.new(project.seed_genome())]                # the current train.py
    await evokit.tell(storage, archive, await evokit.evaluate(storage, seeds, eval_fn=eval_fn))

await recipes.autoresearch(project, eval_fn, ClaudeCode(), storage=storage, archive=archive,
                           instructions=open("program.md").read())
```

With an outside agent there is no loop at all. The task is a folder ([tasks.md](tasks.md)), and
every CLI command loads it with `load_task(folder).open(storage)`, so the agent calls the same
functions:

- `evokit select-individuals` rebuilds the archive from the storage, and prints `strategy.select(archive, 1)`.
- `evokit evaluate-directory DIR --parent ID` records the agent's program with `history.record_created(...)`, then
  runs `evokit.evaluate` and `evokit.tell`.

### AVO

AVO is a single coding-agent session. A supervisor redirects the agent after a run of
attempts that do not improve.

```python
# evokit/recipes/avo.py
REDIRECT = "The last attempts did not improve the score. Step back and try a different direction."


async def avo(project, eval_fn, agent, *, storage, archive, max_cost=5.0, redirect_after=5):
    strategy = Greedy()
    operator = CodingAgentOperator(agent, project, persistent=True)          # one session for the whole run

    rejected_in_a_row = 0
    while history.usage(storage)["cost_usd"] < max_cost:
        asked = await evokit.ask(storage, archive, strategy, operator)
        evaluated = await evokit.evaluate(storage, asked.individuals, eval_fn=eval_fn)
        admitted = await evokit.tell(storage, archive, evaluated)
        rejected_in_a_row = 0 if admitted else rejected_in_a_row + len(evaluated)
        if rejected_in_a_row >= redirect_after:
            operator.redirect(REDIRECT)
            history.note(storage, "redirect", REDIRECT)
            rejected_in_a_row = 0
    return archive.best()
```

### OpenEvolve

[OpenEvolve](https://github.com/codelion/openevolve) is an open AlphaEvolve. It has three parts:

- **Islands of MAP-Elites grids.** The default features are code length and diversity.
- **Parents, inspirations and LLMs.** A parent is drawn by exploration, exploitation or elite
  sampling. Inspirations come from the same island. An ensemble of LLMs writes diffs.
- **Ring migration** every few generations.

```python
# evokit/recipes/openevolve.py
async def openevolve(project, eval_fn, models, weights=None, *, storage, archive, inspirations=4, migrate_every=20,
                     migrate_top=2, workers=4, max_evaluations=1000, max_cost=None, seed=0):
    strategy = IslandSampler(inspirations=inspirations, seed=seed)
    operator = Choice({name: DiffOperator(model, project) for name, model in models.items()}, weights=weights, seed=seed)

    def done():
        return history.counts(storage)["evaluated"] >= max_evaluations or (max_cost is not None and history.usage(storage)["cost_usd"] >= max_cost)

    async def worker():
        while not done():
            asked = await evokit.ask(storage, archive, strategy, operator)
            if asked.exhausted:
                return
            evaluated = await evokit.evaluate(storage, asked.individuals, eval_fn=eval_fn)
            before = history.counts(storage)["evaluated"]
            await evokit.tell(storage, archive, evaluated)
            if before // migrate_every < history.counts(storage)["evaluated"] // migrate_every:
                archive.migrate(top=migrate_top)

    await asyncio.gather(*(worker() for _ in range(workers)))
    return archive.best()
```

```python
storage = DirectoryStorage("runs/packing")
archive = IslandArchive(islands=4, features=("lines", "diversity"), bins=10, storage=storage)
await recipes.openevolve(project, eval_fn, models, storage=storage, archive=archive)
```

OpenEvolve's cascade evaluation (a quick test, then the full one for programs that pass) is
part of the `eval_fn` the caller passes, like DGM's `staged_eval` below.
Migration is counted from the storage, so a continued run keeps the same schedule.

### ShinkaEvolve

[ShinkaEvolve](https://github.com/SakanaAI/ShinkaEvolve) (Sakana AI, 2025) adds four ideas to
an AlphaEvolve-style loop:

- islands, with migration every 10 admissions;
- parents weighted by fitness and by how few children they have, plus inspirations;
- an ensemble of LLMs chosen by a UCB1 bandit, and diff, full-rewrite or crossover edits;
- novelty rejection before evaluation, and a meta-scratchpad of what has worked so far.

```python
# evokit/recipes/shinka.py
KINDS = {"diff": 0.6, "full": 0.3, "cross": 0.1}


def make_operator(kind, model, project, instructions):
    if kind == "diff":
        return DiffOperator(model, project, instructions=instructions)
    if kind == "full":
        return FileRewriteOperator(model, project, instructions=instructions)
    return FileRewriteOperator(model, project, instructions=instructions + COMBINE)


async def shinka(project, eval_fn, models, *, instructions, embed, judge, summarizer, storage, archive,
                 workers=4, threshold=0.95, every=10, max_cost=10.0, max_evaluated=150, seed=0):
    strategy = ShinkaParents(seed=seed)
    bandit = ModelBandit(models).fit(storage)                                 # rewards from every child so far; none on a fresh run
    rng = random.Random(seed)
    notes = history.last_note(storage, "meta") or ""

    async def worker():
        nonlocal notes
        while history.usage(storage)["cost_usd"] < max_cost and history.counts(storage)["evaluated"] < max_evaluated:
            kind = rng.choices(list(KINDS), list(KINDS.values()))[0]
            name = rng.choices(*zip(*bandit.probabilities().items()))[0]
            prompt = instructions + (f"\n\nWhat has worked so far:\n{notes}" if notes else "")
            operator = make_operator(kind, models[name], project, prompt)

            asked = await evokit.ask(storage, archive, strategy, operator)
            if asked.exhausted:
                return
            novel = []
            for child in asked.individuals:
                if await is_novel(child, archive, embed, judge, storage, threshold):
                    novel.append(child.with_info(operator=f"{kind}/{name}"))
                else:
                    history.discard(storage, child, "not novel")
            evaluated = await evokit.evaluate(storage, novel, eval_fn=eval_fn)
            before = history.counts(storage)["added"]
            await evokit.tell(storage, archive, evaluated)
            for child in evaluated:
                bandit.reward(child, history.individual(storage, child.parents[0]))
            if before // every < history.counts(storage)["added"] // every:
                archive.migrate(top=1)

            if history.counts(storage)["evaluated"] % every == 0:
                reply = await summarizer.generate(META_PROMPT + describe(archive.best_k(20)))
                notes = reply.text
                history.note(storage, "meta", notes, reply.usage)

    await asyncio.gather(*(worker() for _ in range(workers)))
    return archive.best()
```

The caller passes `IslandArchive(islands=2, features=(), cell_size=50, storage=storage)`. Everything
else a continued run needs is read from the storage: the bandit refits from past children, the
migration schedule from `history.counts(storage)`, and the meta notes from the last `meta` note.

The recipe-specific pieces live next to the recipe:

- **`ShinkaParents`** is a strategy. A parent's weight is sigmoid(normalized score) / (1 + children),
  and the island's top members and some archive elites come back as `Selection.inspirations`.
- **`ModelBandit`** is UCB1 over models. Its reward is how much a child improved on its parent.
  `fit(storage)` rewards it for every child already in the history.
- **`is_novel(child, archive, embed, judge, storage, threshold)`** compares the child's embedding with
  the archive. It asks the LLM judge when the closest member is within `threshold`, and notes the
  cost of the check in the history.

### Darwin Gödel Machine

In the [Darwin Gödel Machine](https://arxiv.org/abs/2505.22954), the genome is a coding
agent's repository:

- Every agent that still works is kept, so the archive is open-ended.
- Parents are sampled by score, discounted by how many children they already have.
- The parent agent reads the logs of its own evaluation, proposes one improvement to itself,
  and implements it on a copy of its code. That is `SelfModify`.
- Evaluation is staged: a few tasks first, and the full benchmark only for agents that do well
  on those.

```python
# evokit/recipes/dgm.py
async def dgm(eval_fn, *, storage, archive, operator=None, fixed_agent=None, instructions=None, workers=4,
              max_children=80, sharpness=10.0, midpoint=0.5, open_ended=True):
    strategy = ProportionalSelection(sharpness, midpoint) if open_ended else Greedy()
    operator = operator or SelfModify(storage, fixed_agent=fixed_agent, instructions=instructions)

    async def worker():
        while history.counts(storage)["created"] <= max_children:
            asked = await evokit.ask(storage, archive, strategy, operator)
            if asked.exhausted:
                return
            await evokit.tell(storage, archive, await evokit.evaluate(storage, asked.individuals, eval_fn=eval_fn))

    await asyncio.gather(*(worker() for _ in range(workers)))
    return archive.best()


def staged_eval(quick, full, promote):
    """DGM's staged evaluation: run `quick`; only if promote(its evaluation), run `full` and return that."""
    async def eval_fn(individual, workdir):
        evaluation = await maybe_await(quick(individual, workdir))
        return await maybe_await(full(individual, workdir)) if promote(evaluation) else evaluation
    return eval_fn
```

The caller brings the seed agent and the evaluation:

```python
storage = DirectoryStorage("runs/dgm")
eval_fn = staged_eval(run_10_tasks, run_50_tasks, promote=lambda e: e.objectives.get("score", 0) >= 0.4)
archive = UnboundedArchive("score", storage=storage)
if not archive.members():
    seeds = [Individual.new(Files.from_directory("seed_agent"))]
    await evokit.tell(storage, archive, await evokit.evaluate(storage, seeds, eval_fn=eval_fn))
await recipes.dgm(eval_fn, storage=storage, archive=archive)
```

The paper's ablations:

- **No self-improvement:** pass `fixed_agent=seed_agent`, so the seed agent makes every
  modification.
- **Not open-ended:** pass `open_ended=False` and a `LatestArchive("score", storage=storage)`,
  so the loop builds greedily on the newest working agent.

### DGM-Hyperagents

[Hyperagents](https://arxiv.org/abs/2603.19461) (Zhang et al., 2026) removes DGM's assumption
that evaluation and self-modification are the same kind of task. In DGM:

- the improvement procedure is fixed: diagnose, then implement;
- the procedure only gets better because the agent gets better at coding;
- that only helps if the task is coding too.

A **hyperagent** puts the task agent and a **meta agent** in one editable repository. The
parent's meta agent edits a copy of the whole repository, its own code included. So the way
agents get improved evolves too, on any task. In the paper, the meta agent invented persistent
memory and performance tracking for itself, and those improvements transferred across domains.

In EvoKit this is DGM's loop with one different component: the operator.

```python
# evokit/recipes/hyperagents.py
async def hyperagents(eval_fn, *, storage, archive, fixed_meta_agent=None, **dgm_options):
    """DGM-H: DGM, where the improvement procedure is part of the genome."""
    operator = MetaAgentModify(storage, fixed_meta_agent=fixed_meta_agent)
    return await dgm(eval_fn, storage=storage, archive=archive, operator=operator, **dgm_options)
```

```python
# evokit/files/meta_agent.py
class MetaAgentModify(Operator):
    """The genome is one repository: task_agent.py solves the task, meta_agent.py improves the repository.
    The parent's meta agent edits a copy of the whole repository, itself included. With
    `fixed_meta_agent`, that meta agent edits every parent instead (the ablation without
    metacognitive self-modification)."""

    name = "meta-agent"

    def __init__(self, storage, *, entrypoint="meta_agent.py", logs="logs", fixed_meta_agent=None, timeout="30m"):
        self.storage, self.entrypoint, self.logs = storage, entrypoint, logs
        self.fixed_meta_agent, self.timeout = fixed_meta_agent, timeout

    async def create(self, selection, archive) -> Offspring:
        parent = selection.parents[0]
        with tempfile.TemporaryDirectory(prefix="evokit-hyperagent-") as tmp, \
                self.storage.local(f"artifacts/{parent.id}") as parent_artifacts:
            root = Path(tmp)
            parent.genome.write_to(root / "repo")
            write_history(root / "history", archive, parent_artifacts / self.logs)
            meta = root / "repo"
            if self.fixed_meta_agent is not None:
                meta = root / "meta"
                self.fixed_meta_agent.write_to(meta)
            argv = [sys.executable, str(meta / self.entrypoint), "--repo", str(root / "repo"), "--history", str(root / "history")]
            result = await CommandAgent(argv, self.timeout).run(root / "repo", "")
            if result.error:
                raise OperatorError(f"the meta agent failed: {result.error}", usage=result.usage)
            child = Files.from_directory(root / "repo")
        if child == parent.genome:
            raise OperatorError("the meta agent changed nothing", usage=result.usage)
        return Offspring((Individual.new(child, [parent.id], {"operator": self.name, "summary": result.summary}),), result.usage)


def write_history(directory, archive, parent_logs):
    """What the meta agent can read: every archived agent's lineage and score, and the parent's logs."""
    directory.mkdir()
    with open(directory / "archive.jsonl", "w") as f:
        for member in archive.members():
            f.write(json.dumps({"id": member.id, "parents": member.parents,
                                "score": member.evaluation.objectives.get("score"),
                                "summary": member.info.get("summary", "")}) + "\n")
    if parent_logs.is_dir():
        shutil.copytree(parent_logs, directory / "parent_logs")
```

The pieces fit together like this:

- **The seed** is a repository with `task_agent.py` and a minimal `meta_agent.py`. In the
  paper, the seed meta agent is an LLM with tools, told to modify any part of the codebase.
- **`eval_fn`** runs `task_agent.py` on the domain's tasks, through
  `script_agent(..., entrypoint="task_agent.py")`. It writes logs to `workdir/logs`, where the
  next meta agent will read them.
- **Memory comes for free.** A meta agent can only remember across generations by writing files
  into the repository, for example notes or a score tracker. The repository is the genome, so
  that memory is inherited and selected like everything else.
- **The paper's ablations are arguments:**
  - `fixed_meta_agent=seed_repository`: the seed's meta agent edits every parent;
  - `open_ended=False`: no archive, build greedily on the latest agent.

### A generational GA

```python
# evokit/recipes/ga.py
async def ga(eval_fn, mutate, *, storage, archive, generations=100):    # archive: PopulationArchive(size), seeded by the caller
    strategy = Tournament(3)
    for _ in range(generations):
        asked = await evokit.ask(storage, archive, strategy, mutate, n=archive.size)
        await evokit.tell(storage, archive, await evokit.evaluate(storage, asked.individuals, eval_fn=eval_fn))
    return archive.best()
```

### Across the recipes

| Recipe | Archive the caller passes | Strategy | Who creates the child | Loop |
| --- | --- | --- | --- | --- |
| autoresearch | `SimplerBestArchive` | `Greedy` | a coding agent | sequential, forever |
| AVO | `BestArchive` | `Greedy` | one persistent agent session, redirected | sequential, cost limit |
| OpenEvolve | `IslandArchive` (MAP-Elites) | `IslandSampler` | an ensemble of LLMs writing diffs | 4 workers |
| ShinkaEvolve | `IslandArchive` | `ShinkaParents` | LLMs chosen by a bandit, three edit kinds, novelty-filtered | 4 workers |
| DGM | `UnboundedArchive` | `ProportionalSelection` | the parent agent (`SelfModify`) | 4 workers |
| DGM-Hyperagents | `UnboundedArchive` | `ProportionalSelection` | the parent's own meta agent (`MetaAgentModify`) | 4 workers |
| GA | `PopulationArchive` | `Tournament` | a mutation operator | generational |

## What goes away

| Today | Here |
| --- | --- |
| `Experiment` with `ask` / `tell` / `run(workers, max_*)` | `evokit.ask`, `evokit.evaluate`, `evokit.tell`, and a loop (`asyncio.gather` for several workers) |
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
