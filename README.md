# EvoKit

A small library for evolutionary search where the variation operator can be a one-shot
LLM call, a headless coding agent, or the coding agent you are talking to.

A `Task` says what evolves and how it is scored; an `Experiment` is one search on it, kept in a
directory; a recipe is a plain async function that takes the task and every component, creates
the experiment, and loops: pick individuals from the archive, make a child, submit it.

```python
import evokit as ek
from evokit.operators import rewrite_text

task = ek.Task(seed="You are a helpful assistant.", evaluate=score)    # score(individual) -> Evaluation
experiment = task.create_experiment("experiments/prompt", max_cost=5)  # seeded; continued if it exists
while not experiment.limits.reached():
    parent, *others = experiment.archive.sample(3)                     # the archive's own strategy
    if child := await experiment.create(rewrite_text(model, parent, others)):
        await experiment.submit(child)                                 # scored, offered to the archive
```

- `Task(seed, evaluate, instructions=, objective=, minimize=, files=)`: a genome is any JSON
  value, or a `Files` tree; `Task.from_directory("packing")` reads a task folder.
- `evaluate(individual)` returns an `Evaluation(objectives, feedback, usage, logs)`; no
  objectives means it failed. It can be sync or async; `@evokit.evaluator(timeout=...)` makes
  it never raise, so a crash or a timeout becomes a failed evaluation.
- `Experiment` holds the storage (`DirectoryStorage`, or in memory), the task, the archive and
  the `Limits` (`max_cost`, `max_evaluations`, `max_failures`, `stop` rules), and records
  everything: what was created, evaluated, spent and noted. `ek.open_experiment(dir)` reads one back.
- `Archive` is the population. `archive.sample(n, strategy=None)` picks `n` individuals with its
  default strategy or any other (`best`, `uniform`, `tournament(size)` in `evokit.strategies`, or
  a function of your own); `archive.top(n)` gives the best.
- A child is made by an agent (`experiment.start`, below) or by a variation function returning
  `(child, usage)`: `rewrite_text`, `diff`, `rewrite_files`, `self_modify`, `meta_agent_modify`
  in `evokit.operators`, or any async function of yours.
- `experiment.create(make)` records the child it makes, with its cost (a failure, OperatorError,
  is a "failed" note); `experiment.submit(individual)` scores it with the task's `evaluate`,
  offers it to the archive, and records both.

```bash
pip install -e ".[mcp]"        # or ".[dev]" for the tests
```

## How the design works

There are few layers: components with one job each, a few functions that call them, and an
experiment that records everything that happened. An algorithm is a function you can read top
to bottom, and changing one means editing or copying it.

- **Components decide; the experiment records.** The archive decides who survives, the strategy
  who is sampled, the agent or variation function what the child is, and the task's `evaluate`
  how good it is. `create` and `submit` write to the experiment, so a recipe never records
  anything by hand.
- **Every cost is in the experiment.** Creating, evaluating, failed variations, agent turns
  and notes each record their own `usage`, so `experiment.cost()` is the cost of the whole
  search, and the limits hold across restarts.
- **Restarting is free.** `create_experiment` on an existing directory continues it: the archive
  replays its own change log, and children a crash left unevaluated are submitted. A recipe
  never knows whether it is continuing.
- **Strategies are plain functions.** `strategy(candidates, n, archive)`: the archive offers its
  candidates, best first (an `IslandArchive` one island at a time), and the strategy draws from
  `archive.rng`. Any strategy works with any archive, and algorithm-specific ones (DGM's
  `proportional`, OpenEvolve's `islands`, Shinka's `parents`) live in their recipes.
- **Storage is bytes at paths.** `write`, `read`, `exists`, `list`, `append`, `local` (a real
  folder, for programs that need one) and `lock`. Individuals, archive logs and genome blobs are
  formats on top of it, so a new backend is those seven methods.
- **The evaluator cannot be gamed.** A task folder splits into the genome (the files that
  evolve) and a read-only environment (the evaluator, the data); `task.read(dir)` raises if an
  operator edited the environment.
- **The agent layer knows nothing about evolution.** An `Agent` opens sessions in a directory
  with tools and a system prompt; `session.prompt(text, output=Dataclass)` streams `ToolCall`
  events and ends with a `Turn`, whose `output` is what the agent passed to its `finish` tool.
  EvoKit's tools (read the experiment, `score`, `submit`) and its system prompt (the search, the
  editable files, the task) are ordinary arguments it passes in.

## Agents

An agent can make one child at a time, editing its parent and naming the change:

```python
@dataclass
class Change:
    title: str
    description: str = ""

(parent,) = experiment.archive.top(1)
async with experiment.start(agent, start=parent, submit=False) as directory:
    turn = await agent.prompt("Make one change that improves the score.", output=Change)
    child = ek.Individual.new(task.read(directory), [parent.id], name=turn.output.title)
await experiment.submit(child)
```

(the autoresearch recipe), or work in one conversation for the whole search, scoring drafts and
submitting versions itself:

```python
async with experiment.start(agent):                  # a work directory holding the best version
    while not experiment.limits.reached():
        async for event in agent.prompt("Improve it; submit what is better."):
            if experiment.since_improvement() > 5:
                agent.send("Try a different data structure.")  # reaches it at its next step
```

Inside `start`, `agent.prompt` and `agent.send` talk to that one conversation, every turn is
recorded, each prompt may spend what the limits leave, and the agent is stopped once they are
reached. `agent.fresh()` is the same agent in a conversation of its own that can only read the
experiment, e.g. a reviewer. This is the AVO recipe.

## Recipes

```python
from evokit import recipes

task = ek.Task.from_directory("examples/circle_packing")
best = await recipes.avo(task, ek.ClaudeCode(), dir="experiments/avo-0", max_cost=5)
best = await recipes.openevolve(task, {"fast": fast_model, "strong": strong_model}, dir="experiments/oe-0")
```

The recipes are `autoresearch`, `avo`, `openevolve`, `shinka`, `dgm` (with `staged_eval`),
`hyperagents` and `ga`, one file each in `evokit/recipes/`. Each takes the task first and its
components and options as keyword arguments. To change one, copy it.

## From the shell

A task is a folder: `TASK.md` (the goal, the rules, the scoring), `config.yaml` (which files
evolve, how they are scored, and optionally the recipe `evokit run` uses), `evaluate.py`, and
the files that evolve. `evokit init NAME` writes a template that runs as it is. See
[docs/tasks.md](docs/tasks.md).

```bash
evokit init packing                                     # the task template
evokit run avo packing --dir experiments/0 --max-cost 5   # a recipe; its options come from its signature
evokit run packing --dir experiments/0                  # the recipe named in config.yaml's recipe: section
evokit stats --dir experiments/0                        # progress: evaluations, cost, the best
evokit log --dir experiments/0                          # the lineage; also show, diff, grep, notes
```

`--agent claude` (the default) is Claude Code; `--agent "COMMAND"` runs any command in the
agent's directory.

## Who is the operator

- **EvoKit:** a recipe runs the loop (`evokit run`, or the recipe in Python).
- **The coding agent you are talking to:** `evokit status --task packing --dir experiments/0`
  creates the experiment. The agent calls `evokit list-individuals`, `evokit checkout-individual ID DIR`,
  edits, then `evokit evaluate-directory DIR --parent ID`, which evaluates it and offers it to the
  archive. `evokit skill export claude` (or `cursor`, `codex`) installs the instructions.
- **An RLM as the whole algorithm:** `experiment_tools(experiment)` (from
  `evokit.interface.commands`) gives it the same commands as plain callables, and it decides
  everything else.

Classic mutations need no model: `recipes.ga(task, mutate)` takes a plain function from a genome
to a new one. [`examples/fixed_mutation`](examples/fixed_mutation/main.py) evolves a vector with two.

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
| `evokit/` (core) | `task.py` (`Task`), `experiment.py` (`Experiment`, `Limits`, `open_experiment`, `create`/`submit`, the agent tools and system prompt), `evaluate.py` (`evaluator`, `as_evaluate`), `records.py` (`Individual`, `Evaluation`), `strategies.py` (`best`, `uniform`, `tournament`), the `Archive` and `Storage` bases |
| `archives/` | `BestArchive`, `SimplerBestArchive`, `PopulationArchive`, `RankedArchive`, `UnboundedArchive`, `LatestArchive`, `NoveltyArchive`, `IslandArchive` |
| `operators/` | variation functions (`rewrite_text`, `diff`, `rewrite_files`, `self_modify` for DGM, `meta_agent_modify` for Hyperagents) and the models they call (`AnthropicModel`, `OpenAIModel`, `FunctionModel`) |
| `storages/` | `DirectoryStorage`, `MemoryStorage` |
| `files/` | `Files` (a genome that is a set of files), `Project`, `ScriptEvaluator`, `with_lines` |
| `agents/` | `Agent`, `Session` and their events (`ToolCall`, `Turn`), the harnesses (`ClaudeCode`, `CommandAgent`, `ReplAgent`), and the `evokit` command that carries tool calls |
| `recipes/` | one function per published algorithm |
| `interface/` | the commands over an experiment, the CLI (`evokit run`), the MCP server, and the skill |

`import evokit as ek` gives the core and the implementation packages; `files`, `agents`,
`recipes` and `interface` are imported explicitly.

## Docs

| Doc | What it covers |
| --- | --- |
| [primitives.md](docs/primitives.md) | Types, components, storage and history (still describes `ask`/`tell`) |
| [recipes.md](docs/recipes.md) | One loop per published algorithm: autoresearch, AVO, OpenEvolve, ShinkaEvolve, DGM, Hyperagents, a GA |
| [tasks.md](docs/tasks.md) | Task folders, `config.yaml`, the evaluator contract, the CLI |
| [reading-runs.md](docs/reading-runs.md) | `log`, `show`, `diff`, `grep` over an experiment |

## TODO

Design work that is open, roughly in order:

- [ ] **Several processes on one experiment.** Today a second driver exits with `run_busy`, and
  pending children are resumed only by the process that opens the experiment.
- [ ] **A sandbox** that wraps both `evaluate` and the agents inside operators, since
  self-modifying agents run code they wrote.
- [ ] **Re-evaluation.** For noisy benchmarks, and for re-running archived agents on a new domain
  (Hyperagents' transfer): `submit` should accept an archived individual for a new evaluation.
- [ ] **Evaluation caching.** Identical genomes are evaluated again.
- [ ] **Sampling that sees what is in flight.** With 8 workers, `best` hands the same parent to
  all 8 (Optuna's `constant_liar`).
- [ ] **Remote storages:** `S3Storage`, `ModalStorage`.
- [ ] **Multi-objective archives** (a Pareto front).
- [ ] **The agent's `evokit` command needs evokit installed.** A session puts a shim on the
  agent's `PATH` that imports `evokit`, which fails when the package is only on `sys.path` from
  the working directory.
- [ ] **The rest of the docs** (`docs/primitives.md`, `docs/tasks.md`, `docs/recipes/`) still
  describe the API before `Task` and `Experiment`.

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
