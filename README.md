# EvoKit

A small library for evolutionary search where the variation operator can be a one-shot
LLM call, a headless coding agent, or the coding agent you are talking to.

EvoKit provides the pieces of an evolutionary algorithm, and a recipe is a plain loop over
three functions:

```python
asked = await evokit.ask(storage, archive, strategy, operator)              # select, then create
evaluated = await evokit.evaluate(storage, asked.individuals, eval_fn=eval_fn)
admitted = await evokit.tell(storage, archive, evaluated)                    # the archive keeps them or not
```

- `Storage` is where a run lives (`DirectoryStorage`, `MemoryStorage`).
- `Archive` is the population; built with `storage=`, it saves and restores itself.
- `Strategy.select(archive, n)` picks parents and inspirations.
- `Operator.create(selection, archive)` makes children.
- `eval_fn(individual, workdir)` returns an `Evaluation(objectives, feedback, usage)`; no
  objectives means it failed.
- `evokit.history` records what was created, evaluated and spent, inside `ask` and `tell`.

A genome is any value: a number, a string, a dict, or a `Files` tree from the optional
`files` package. See [docs/primitives.md](docs/primitives.md) for the design.

```bash
pip install -e ".[mcp]"
```

## Example: any genome

```python
import evokit as ek
from evokit import history
from evokit.archives import BestArchive
from evokit.operators import AnthropicModel, TextRewriteOperator
from evokit.storages import DirectoryStorage
from evokit.strategies import Greedy

def eval_fn(individual, workdir):
    return ek.Evaluation({"score": run_my_benchmark(individual.genome)})

storage = DirectoryStorage("runs/prompt")
archive = BestArchive(storage=storage)
strategy, operator = Greedy(), TextRewriteOperator(AnthropicModel("claude-sonnet-4-5", price=(3, 15)))

if not archive.members():                         # a fresh run; a continued one restored itself
    seeds = [ek.Individual.new("You are a helpful assistant.")]
    await ek.tell(storage, archive, await ek.evaluate(storage, seeds, eval_fn=eval_fn))

while history.usage(storage)["cost_usd"] < 5:
    asked = await ek.ask(storage, archive, strategy, operator)
    await ek.tell(storage, archive, await ek.evaluate(storage, asked.individuals, eval_fn=eval_fn))
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

## Algorithms on circle packing

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

## CLI contract

When stdout is not a TTY, output is compact JSON (indented when it is a TTY). Errors are JSON on stderr.
Exit codes: 0 ok, 1 internal error, 2 usage error.
