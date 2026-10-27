# EvoKit

A small library for evolutionary search where the variation operator can be a one-shot
LLM call, a headless coding agent, or the coding agent you are talking to.

EvoKit provides the pieces of an evolutionary algorithm, and a recipe is a plain loop over
`ask`, an eval function, and `tell`:

```python
asked = await evokit.ask(storage, archive, strategy, operator)              # select, then create
evaluations = await asyncio.gather(*map(score, asked.individuals))          # score = @evokit.eval_fn
admitted = await evokit.tell(storage, archive, asked.individuals, evaluations)  # the archive keeps them or not
```

- `Storage` is where a run lives (`DirectoryStorage`, `MemoryStorage`).
- `Archive` is the population; built with `storage=`, it saves and restores itself.
- `Strategy.select(archive, n)` picks parents and inspirations.
- `Operator.create(selection, archive)` makes children.
- `eval_fn(individual)` returns an `Evaluation(objectives, feedback, usage, logs)`; no
  objectives means it failed. Decorated with `@evokit.eval_fn(timeout=..., executor=...)` it is
  async and never raises: a crash or a timeout becomes a failed evaluation. `tell` stores
  `logs` (file name to contents) in the individual's artifact folder.
- `evokit.history` records what was created, evaluated and spent, inside `ask` and `tell`.

A genome is any value: a number, a string, a dict, or a `Files` tree from the optional
`files` package. See [docs/primitives.md](docs/primitives.md) for the design.

```bash
pip install -e ".[mcp]"        # or ".[dev]" for the tests
```

## How the design works

There is no `Experiment` or `Engine` class, no hooks and no event bus. The library is a few
components with one job each, three functions that call them, and a log of everything that
happened. An algorithm is a loop you can read top to bottom, and changing one means editing
or copying that loop.

- **Components decide; functions record.** The archive decides who survives, the strategy who
  becomes a parent, the operator what the child is, and `eval_fn` how good it is. `ask` and
  `tell` call them and write the history, so a recipe never records anything by hand.
- **Every cost is in the history.** Creating, evaluating, failed operator calls and notes each
  record their own `usage`, so `history.usage(storage)["cost_usd"]` is the cost of the whole
  run, and limits (`Limits(max_evaluations, max_cost, max_failures)`) hold across restarts.
- **Restarting is free.** An archive built with `storage=` replays its own change log, and
  `ask` hands back children a crash left unevaluated before creating new ones. A recipe never
  knows whether it is continuing a run.
- **Strategies keep no state.** They read everything from the archive, so any strategy works
  with any archive and nothing about them has to be saved.
- **Storage is bytes at paths.** `write`, `read`, `exists`, `list`, `append`, `local` (a real
  folder, for programs that need one) and `lock`. Individuals, archive logs and genome blobs are
  formats on top of it, so a new backend is those seven methods.
- **The evaluator cannot be gamed.** With `files`, a `Project` splits a folder into the
  genome (the files that evolve) and a read-only environment (the evaluator, the data).
  `Project.read` raises if an operator edited the environment.
- **One registry, four interfaces.** Methods marked `@command` become CLI subcommands, MCP
  tools, the shipped `SKILL.md`, and plain callables (`ek.tools(...)`) for RLMs and scripts.

## Example: any genome

```python
import evokit as ek
from evokit import history
from evokit.archives import BestArchive
from evokit.operators import AnthropicModel, TextRewriteOperator
from evokit.storages import DirectoryStorage
from evokit.strategies import Greedy

@ek.eval_fn(timeout="10m")
def score(individual):
    return ek.Evaluation({"score": run_my_benchmark(individual.genome)})

storage = DirectoryStorage("runs/prompt")
archive = BestArchive(storage=storage)
strategy, operator = Greedy(), TextRewriteOperator(AnthropicModel("claude-sonnet-4-5", price=(3, 15)))

if not archive.members():                         # a fresh run; a continued one restored itself
    seed = ek.Individual.new("You are a helpful assistant.")
    await ek.tell(storage, archive, [seed], [await score(seed)])

while history.usage(storage)["cost_usd"] < 5:
    asked = await ek.ask(storage, archive, strategy, operator)
    await ek.tell(storage, archive, asked.individuals, await asyncio.gather(*map(score, asked.individuals)))
archive.best().genome
```

Several workers are `asyncio.gather` over copies of the loop.

## Example: evolving files, with a recipe

```python
from evokit import recipes
from evokit.archives import IslandArchive
from evokit.files import Project, ScriptEvaluator, with_lines

project = Project(".", genome=["main.py"])         # everything else is the read-only environment
eval_fn = with_lines(ScriptEvaluator(project, "python evaluate.py"))
storage = DirectoryStorage("runs/packing")
archive = IslandArchive(islands=4, features=("lines", "diversity"), storage=storage)
# ... seed the archive as above, with project.seed_genome()
await recipes.openevolve(project, eval_fn, {"fast": fast_model, "strong": strong_model}, storage=storage, archive=archive)
```

The recipes are `autoresearch`, `avo`, `openevolve`, `shinka`, `dgm` (with `staged_eval`),
`hyperagents` and `ga`, one file each in `evokit/recipes/`. To change one, copy its loop.

## From the shell

A task is a folder: `TASK.md` (the goal, the rules, the scoring), `config.yaml` (which files
evolve, how they are scored, which components to use), `evaluate.py`, and the files that evolve.
`evokit init NAME` writes a template that runs as it is. See [docs/tasks.md](docs/tasks.md).

```bash
evokit init packing                                   # the task template
evokit status --task packing --dir runs/0             # creates the run, seeded with the task's files
evokit run --dir runs/0 --workers 4 --max-cost 5      # when config.yaml names an operator
evokit best-individual --dir runs/0
evokit log --dir runs/0                               # the lineage; also show, diff, grep
```

In Python, `ek.load_task("packing")` gives the same objects: `task.project`, `task.eval_fn`,
`task.instructions`, and `task.open(storage)` for the run.

## Who is the operator

- **EvoKit:** a run with an operator runs the loop (`evokit run`, or a recipe).
- **The coding agent you are talking to:** a task with `operator = "none"`. The agent calls
  `evokit select-individuals`, `evokit checkout-individual ID DIR`, edits, then
  `evokit evaluate-directory DIR --parent ID`, which evaluates it and offers it to the archive.
  `evokit skill export claude` (or `cursor`, `codex`) installs the instructions.
- **An RLM as the whole algorithm:** `ek.tools(*open_setup("runs/0").objects)` (from `evokit.interface.setup`) gives it plain
  callables (`evaluate_genome`, `select_individuals`, `list_individuals`, ...) and it decides
  everything else.

Classic operators need no model: `ek.Mutate(fn)` turns a function of the parents' genomes into
an operator. [`examples/fixed_mutation`](examples/fixed_mutation/main.py) evolves a vector with two.

## Examples

[`examples/circle_packing`](examples/circle_packing) packs 26 circles in the unit square
(the seed's sum of radii is 1.6026; the best known is about 2.635) with four algorithms:
autoresearch, AVO, AlphaEvolve / OpenEvolve, and the Darwin Gödel Machine, each with its
ablations as one-swap variants.

```bash
cd examples/circle_packing
python -m algorithms.openevolve --dir runs/openevolve-full-0
python compare.py run autoresearch avo openevolve:full openevolve:one-island --repeats 3
python compare.py report
```

[`examples/co_bench`](examples/co_bench) runs a recipe on one
[CO-Bench](https://github.com/sunnweiwei/CO-Bench) combinatorial optimization problem, then scores
the best `solve.py` on the held-out test set. CO-Bench's protocol is 64 evaluations per problem.

```bash
cd examples/co_bench
python setup.py                                       # once: downloads five problems into tasks/
python run.py bin-packing-one-dimensional --algorithm autoresearch --dir runs/bin-packing-autoresearch-0
python run.py aircraft-landing --algorithm avo --agent claude --max-cost 5 --dir runs/aircraft-avo-0
```

## Layout

| Package | What is in it |
| --- | --- |
| `evokit/` (core) | `loop.py` (`ask`, `evaluate`, `tell`, `Limits`), `records.py` (`Individual`, `Evaluation`, `Selection`, `Offspring`, `Asked`), `history.py`, the `Archive`, `Strategy`, `Operator` and `Storage` bases, `task.py` (`load_task`) |
| `archives/` | `BestArchive`, `SimplerBestArchive`, `PopulationArchive`, `RankedArchive`, `UnboundedArchive`, `LatestArchive`, `NoveltyArchive`, `IslandArchive` |
| `strategies/` | `Greedy`, `Tournament`, `ProportionalSelection`, `IslandSampler` |
| `operators/` | `Choice`, `Mutate`, `LLMOperator`, `TextRewriteOperator`, and the models they call (`AnthropicModel`, `OpenAIModel`, `FunctionModel`) |
| `storages/` | `DirectoryStorage`, `MemoryStorage` |
| `files/` | `Files` (a genome that is a set of files), `Project`, `ScriptEvaluator`, `with_lines`, `DiffOperator`, `FileRewriteOperator` |
| `agents/` | `CodingAgent` and its implementations (`ClaudeCode`, `CommandAgent`, `ReplAgent`), `CodingAgentOperator`, and the self-modifying operators `SelfModify` (DGM) and `MetaAgentModify` (Hyperagents) |
| `recipes/` | one loop per published algorithm |
| `interface/` | the `@command` registry, the CLI, the MCP server, the skill, and the read commands (`log`, `show`, `diff`, `grep`) |

`import evokit as ek` gives the core and the implementation packages; `files`, `agents`,
`recipes` and `interface` are imported explicitly.

## Docs

| Doc | What it covers |
| --- | --- |
| [primitives.md](docs/primitives.md) | Types, components, storage, history, and `ask`, `evaluate`, `tell` |
| [recipes.md](docs/recipes.md) | One loop per published algorithm: autoresearch, AVO, OpenEvolve, ShinkaEvolve, DGM, Hyperagents, a GA |
| [tasks.md](docs/tasks.md) | Task folders, `config.yaml`, the evaluator contract, the CLI |
| [reading-runs.md](docs/reading-runs.md) | `log`, `show`, `diff`, `grep` over a run |

## TODO

Design work that is open, roughly in order:

- [ ] **Run a recipe from the CLI.** `evokit run` runs the generic ask/evaluate/tell loop, not a
  recipe's own loop; a `run: {recipe: avo}` key in `config.yaml` could name one.
- [ ] **Several processes on one run.** `history.claim_pending` claims in memory, not in storage, and
  a process does not read other processes' new records before it selects. Today a second driver
  exits with `run_busy`.
- [ ] **A sandbox** that wraps both `eval_fn` and the agents inside operators, since
  self-modifying agents run code they wrote.
- [ ] **Re-evaluation.** For noisy benchmarks, and for re-running archived agents on a new domain
  (Hyperagents' transfer): `tell` should accept an archived individual with a new evaluation.
- [ ] **Evaluation caching.** Identical genomes are evaluated again.
- [ ] **Selection that sees what is in flight.** With 8 workers, `Greedy` hands the same parent to
  all 8 (Optuna's `constant_liar`).
- [ ] **Remote storages:** `S3Storage`, `ModalStorage`.
- [ ] **Multi-objective archives** (a Pareto front).
- [ ] **The core imports `interface/`.** `evokit/__init__.py` imports `command` and `tools` from
  `interface/commands.py`, so the core depends on the optional layer above it.
- [ ] **The agent's `evokit` command needs evokit installed.** `CodingAgentOperator` puts a
  shim on the agent's `PATH` that imports `evokit`, which fails when the package is only on
  `sys.path` from the working directory (and fails
  `test_coding_agent_gets_the_skill_and_can_read_the_run` without `pip install -e .`).

Docs and packaging:

- [ ] `mkdocs.yml` still has template placeholders and no `nav`.
- [ ] `Makefile` is an unrendered cookiecutter template.

## Development

```bash
pip install -e ".[dev]"
pytest                      # offline: scripted agents and models stand in for LLMs
```

## CLI contract

When stdout is not a TTY, output is compact JSON (indented when it is a TTY). Errors are JSON on stderr.
Exit codes: 0 ok, 1 internal error, 2 usage error.
