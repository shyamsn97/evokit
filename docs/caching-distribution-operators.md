# Caching, distributed execution, and classic operators

A proposal for the three gaps in [design-evaluation.md](design-evaluation.md) that matter:
identical genomes are evaluated again, the loop only runs one generation at a time inside one
process, and there are no classic mutation or crossover operators. The third turns out not
to be a gap in the design; it needs only an example.

The constraints are the same as for the rest of EvoKit: standard EC vocabulary, one job per
component, implementations in the folder named after their base class, and no filesystem
assumptions in the core. None of the three needs a new base class.

## 1. Evaluation caching

### The problem

`Experiment.evaluate_individual_async` always calls the evaluator. A greedy run with a
low-rate mutation, or an agent that reverts its edit, proposes genomes that have already been
scored. With a `ScriptEvaluator` that takes minutes, that time is wasted.

### Who decides whether results can be reused

The evaluator, because only it knows whether it is deterministic and what its result depends
on. Add one method to `Evaluator`:

```python
class Evaluator(ABC):
    @abstractmethod
    async def evaluate(self, genome: Any) -> Fitness: ...

    def cache_key(self) -> str | None:
        """Identifies this evaluation: the same key and the same genome give the same fitness.

        None (the default) means results are never reused.
        """
        return None
```

- `FunctionEvaluator(fn, cache=False)`. With `cache=True` the key is the function's
  qualified name. With `cache="v2"` (any string), that string is added to the key, so
  bumping it invalidates old results after `fn` changes.
- `ScriptEvaluator(project, command, cache=False)`. With `cache=True` the key is a hash of
  the command, `env`, and `project.environment_digest()`. Editing `evaluate.py` or any
  protected file changes the digest, so stale results are never reused.

### Who does the caching

The `Experiment`, not an evaluator wrapper. A cache hit has to change what the experiment
records (the event, the budget, a link to the original individual), and only the experiment
does that. An evaluator wrapper could skip the call, but could not report that it did.

```python
async def evaluate_individual_async(self, individual):
    self.ensure_budget()
    key = self._cache_key(individual.genome)      # None if evaluator.cache_key() is None
    original = key and self.store.get_cached(key)
    if original:
        fitness = self.store.load_individual(original).fitness
        evaluated = individual.with_fitness(fitness).with_info(cached_from=original)
    else:
        ...                                       # evaluate as today
        if key and fitness.status in ("ok", "failed"):
            self.store.put_cached(key, evaluated.id)
```

- **Genome key.** `genome_digest(genome)` goes in `store.py`, next to `encode_genome`. It is
  the sha256 of the canonical JSON of `encode_genome(genome, put_blob=sha256)`. This hashes
  blob contents without writing anything, so it needs no store. For `Files`, like `Files.digest`, it
  depends only on the file contents. Any genome type that implements `to_json` gets a digest
  for free. The
  cache key is `sha256(evaluator.cache_key() + genome_digest(genome))`.
- **Store.** Two methods, `get_cached(key) -> str | None` and
  `put_cached(key, individual_id)`. They are not abstract: the base `Store` implements them
  with `load_state`/`save_state("cache")`, and `DirectoryStore` overrides them with one file
  per key under `cache/`, so the map is never rewritten as a whole.
- **What gets cached.** `ok` and `failed` results, which are properties of the genome.
  `timeout` and `error` are not cached, because they are usually infrastructure problems
  and worth retrying.
- **Duplicates in flight.** When two concurrent workers create the same genome, the second
  awaits the first's evaluation instead of starting its own. The experiment keeps a
  `dict[key, asyncio.Future]` of evaluations in progress.
- **Records.** Every hit is still a new `Individual`, with its own id, parents, operator, and
  usage. Lineage stays accurate, and the archive sees an ordinary evaluated individual. The
  `evaluate` event gets `cached=<original id>`.

### Budget

A cache hit counts toward `Budget.evaluations`. It is recorded separately as `cached_evals`
in the usage, next to `final_evals` and `draft_evals`. There are two reasons:

- **Fairness.** The budget measures proposals. An operator that keeps proposing duplicates
  should not get extra tries over one that doesn't; comparing operators at equal cost is
  what the library is for.
- **Termination.** A zero-cost operator under `Greedy` that keeps producing the same genome
  would otherwise loop forever without spending any budget.

Caching saves wall time and compute, not evaluation budget.

### Not included

Draft evaluations in `CodingAgentOperator` call the evaluator directly and are not cached.
They could share the experiment's cache later. Agents seldom submit the exact same draft
twice, so it isn't worth doing now.

## 2. Distributed execution

There are three levels, each built on the one below.

### 2a. Steady-state concurrency in one process

Today `Engine.step` selects `batch_size` individuals, creates and evaluates them together,
and waits for the slowest before adding any. Agent runs vary from seconds to many minutes,
so most slots sit idle.

Replace `batch_size` with `workers`:

```python
Engine(experiment, strategy, operator, workers=8)
```

- `evolve()` becomes steady-state. It keeps `workers` individuals in flight. When one
  finishes, it is added and both `tell` calls run, then a new selection is made for the free
  slot. `steps` counts individuals, not generations.
- `step()` keeps its current meaning: one generation of `workers` individuals, created
  together. This matches what the tools surface expects.

The heavy work (agent sessions, evaluation scripts) already runs in subprocesses, so one
asyncio process with `workers=N` uses N cores on one machine. This level alone covers most
real runs.

One rule needs to be stated in the code: **never `await` inside a transaction.**
`Store.lock()` is reentrant per thread, and every coroutine on the background loop shares
one thread. Two coroutines that both entered a transaction would corrupt the nesting depth.
So the steady-state loop loads state, releases the lock, awaits the strategy or operator,
and takes the lock again to commit. `_select` gets an async counterpart that follows this
pattern.

### 2b. Several processes on one store

Every state change already goes through `Experiment.transaction`, which reloads the archive,
budget, and component state from the store. So N `evokit evolve --dir runs/x` processes
sharing a `DirectoryStore` mostly work today. Three things need fixing before this can be
supported:

1. **`start()` race.** Two processes that start together can both see
   `started == False` and both evaluate the seeds. Fix: under the lock, the first process
   records `meta["seeding"] = <worker id>`. Other processes wait until `started` is true.
2. **Operator state overwritten.** `step` saves the operator's in-memory state over the
   stored one (`reload=["strategy"]`), so with N processes the last writer wins, and for
   example `Choice` loses the other workers' statistics. Fix: save operator state per
   worker, as `operator.<worker id>`. The strategy is shared and reloaded on every
   transaction, as now.
3. **Budget overshoot.** Each process checks the budget before evaluating, so no evaluation
   starts after the budget is used up. Individuals already being created when it runs out
   still spend their creation cost. The overshoot is at most the total number of workers.
   Document it rather than add reservations, which would leak when a process dies.

The worker id is `<hostname>:<pid>` by default, and `--worker NAME` overrides it.

### 2c. Several machines

This needs a store every machine can reach. A shared filesystem doesn't work: `fcntl` locks
over NFS are unreliable. Add one networked store, served over HTTP:

```
evokit serve-store --dir runs/x --port 8700      # on one machine: owns a DirectoryStore
evokit evolve --store http://host:8700 ...       # on every worker machine
```

- **`stores/http.py`** contains `HTTPStore(url)`, a client that implements `Store`, and
  `serve_store(store, host, port)`, which exposes any `Store`. It uses only the standard
  library (`http.server`, `urllib`), so it adds no dependency or infrastructure.
- **Lock.** `lock()` is a lease: acquiring returns a token with a TTL (30s by default), and
  releasing it, or the holder dying, frees it. Leases are safe because of the rule in 2a:
  transactions never await, so they last milliseconds.
- **Blobs.** Content-addressed, so a client checks with `HEAD /blobs/<key>` before
  uploading. `Files` genomes upload only the files that changed.

### Remote evaluation and mutation: left to the user

(Extended by [sandboxes.md](sandboxes.md): this holds for Python functions; evokit's file-based
components get a `runtime=` parameter.)

There is no `Pool` or executor abstraction. `Evaluator.evaluate` and
`Operator.create_individual` are async, so where they run is their own business, and
`FunctionEvaluator` already accepts async functions:

```python
@app.function()                                   # Modal
def rastrigin_remote(x): ...

async def rastrigin(x):
    return await rastrigin_remote.remote.aio(x)   # Ray: await f.remote(x)

ek.Experiment(rastrigin, BestArchive(), seeds=[seed])
```

The same pattern covers a local `ProcessPoolExecutor` (`loop.run_in_executor`) for
CPU-bound Python, Slurm, or a GPU box. What gets shipped (a function, files and a command,
an agent workspace) depends on the problem, which is why no single interface would fit.
The engine's `workers` sets how many calls are in flight at once.

### Not included

Islands, migration, any built-in task queue, and integrations with Modal, Ray, or similar.
Several engines on one store already give a shared archive. Islands are an archive and strategy topology question for later, not an
execution one.

## 3. Classic operators

This is not a design gap, and nothing needs to be added to the library. `Operator` assumes
nothing about models or agents: a classic mutation is a subclass with one
`create_individual`, and it runs through the same loop, budget, and store unchanged.

```python
class GaussianMutation(ek.Operator):
    name = "gaussian"

    def __init__(self, sigma: float, seed: int | None = None):
        self.sigma = sigma
        self.rng = random.Random(seed)

    async def create_individual(self, individuals):
        parent = individuals[0]
        genome = [v + self.rng.gauss(0, self.sigma) for v in parent.genome]
        return ek.Individual.new(genome, parents=[parent.id])
```

Which mutations make sense depends on the genome, so they belong with the problem, not in
`evokit/operators/`. [`examples/fixed_mutation`](../examples/fixed_mutation/main.py) shows
the pattern. It minimizes Rastrigin in 10 dimensions with two fixed mutations (Gaussian
steps and resetting one coordinate), mixed by the existing `Choice(adapt=True)` under
`Greedy`. That is 5000 evaluations in about two seconds, with no cost in `info["usage"]`,
and the run doubles as a zero-cost baseline next to an agent run.

Population methods (crossover, tournament selection) work the same way: a crossover is an
`Operator` that reads two individuals, and tournament selection is a `Strategy`. They can be
added as examples if someone needs them.

## Order of work

Each step is self-contained and comes with tests.

1. **Caching.** `Evaluator.cache_key`, `genome_digest`, `Store.get_cached`/`put_cached`,
   and the `Experiment` path for hits. Test: a deterministic `FunctionEvaluator` called once
   for two identical genomes; hits counted in `cached_evals`; `timeout` not cached.
2. **Steady-state `workers`.** Replace `batch_size`, make `evolve` async inside, and add the
   no-await-in-transaction rule. Test: a slow individual doesn't block the others' admission.
3. **Multi-process fixes.** The `start()` claim and per-worker operator state. Test: four
   processes on one `DirectoryStore` seed once and finish within the budget plus the number
   of workers.
4. **`HTTPStore` and `serve-store`.** Test: the existing core test suite parametrized over
   `MemoryStore`, `DirectoryStore`, and `HTTPStore`.

## Open questions

- Should `ScriptEvaluator` cache by default? It is usually deterministic, but timing
  benchmarks are not, and silently reusing a timing result would be wrong. The proposal is
  opt-in.
- Should strategy state also be per worker? `Greedy` is stateless, so it
  doesn't matter yet. An adaptive strategy would need to merge state from all workers.
