# Tasks

A task is a folder that defines one problem. It is the only way to define a task: the CLI
and Python both load the same folder into the same objects.

```text
packing/
  TASK.md        what the operator is told: the goal, the rules, what "better" means
  config.yaml    which files evolve, how they are scored, which components a CLI run uses
  evaluate.py    the evaluator: part of the fixed environment, never evolved
  main.py        the seed: the genome files as they are now
```

`evokit init packing` writes this folder from the template in `evokit/templates/task/`. It is a
toy task that works unedited: `solve()` in `main.py` should return 3. Replace the four files
with your problem.

## From the shell

Every run command takes `--dir` (the run directory). The first command on a new directory also
needs `--task`: it creates the run, evaluates the task's genome files as the seed, and adds the
seed to the archive. Later commands continue the run. The run records its task folder in
`run/task.json`, so `--task` is not needed again.

```bash
evokit init packing
evokit status --task packing --dir runs/0      # creates and seeds runs/0
evokit select-individuals --dir runs/0
evokit run --dir runs/0 --max-cost 5           # when config.yaml names an operator
```

If the seed fails its evaluation, the command exits with a usage error that includes the
evaluator's feedback, and nothing is recorded. Fix the files and run the command again. A run
belongs to one task: passing a different `--task` for an existing run directory is an error.

## From Python

```python
import evokit as ek
from evokit.storages import DirectoryStorage

task = ek.load_task("packing", environment="runs/0/env")
task.project        # Project: the genome patterns from config.yaml; TASK.md, config.yaml and evaluate* protected
task.eval_fn        # ScriptEvaluator(task.project, config.yaml's evaluate command and timeout)
task.instructions   # the text of TASK.md
task.seed()         # Individual.new(task.project.seed_genome(), info={"note": "seed"})

archive, strategy, operator = task.components(storage)   # what config.yaml's run section names
run = task.open(DirectoryStorage("runs/0"))               # RunCommands: seeds a new storage, continues an old one
```

`task.open(storage)` is what every CLI command does. The recipes skip the `run` section and take the parts
they need, for example `recipes.avo(task.project, task.eval_fn, agent, instructions=task.instructions, ...)`.
[`examples/circle_packing`](../examples/circle_packing) is a task folder, and its
`algorithms/` scripts load it this way.

## Folder to objects

| In the folder | Becomes |
| --- | --- |
| the folder, `task.genome`, `task.ignore` | `Project(folder, genome=..., protected=(evaluate*, eval/**, .git/**, config.yaml, TASK.md), ignore=...)` |
| everything that is not genome | the environment: copied once into `<run>/env`, so later edits to the folder do not change a run |
| `task.evaluate`, `task.timeout` | `ScriptEvaluator(project, evaluate, timeout)`, the eval function |
| `task.objective`, `task.minimize` | the archive's `objective` and `minimize` |
| `TASK.md` | `task.instructions`: the operator's instructions, and the task section of the exported skill |
| the genome files as they are | the seed individual |
| `run`: `archive`, `strategy`, `operator` | the components, by name; `*_options` mappings are keyword arguments |
| `run.factory: evo:build` | `build(task, storage)` in `evo.py` returns the components instead |

`runs/**` and `.evokit/**` inside the folder are ignored, and so is a run directory placed
anywhere inside it.

## config.yaml

```yaml
task:
  genome: [main.py]             # glob patterns of the files that evolve
  evaluate: python evaluate.py  # run in a copy of the folder
  timeout: 2m
  objective: score
  minimize: false
  # ignore: [data/cache/**]     # neither genome nor environment

run:                            # every key is optional
  archive: best                 # best | simpler | population | islands | unbounded | latest
  strategy: greedy              # greedy | tournament | proportional | islands
  operator: none                # none | diff | rewrite | coding-agent
  # model: claude-sonnet-4-5    # diff and rewrite: AnthropicModel(model, price=price)
  # price: [3.0, 15.0]          # USD per million input and output tokens
  # agent: claude               # coding-agent: ClaudeCode(), or a command for CommandAgent
  # archive_options: {size: 20}
  # strategy_options: {size: 3}
  # operator_options: {max_draft_evals: 5}   # coding-agent: drafts scored by task.evaluate
  # factory: evo:build
```

With `operator: none`, whoever drives the CLI is the operator: the coding agent you are
talking to (see `evokit skill export`), or a script. With an operator, `evokit run` runs ask,
evaluate and tell until a limit is reached.

The factory is the escape hatch for anything the names cannot express. It replaces the
`archive`, `strategy` and `operator` keys:

```python
# evo.py, next to config.yaml
def build(task, storage):
    archive = SimplerBestArchive("score", size="lines", storage=storage)   # storage=storage: it restores itself
    return archive, Greedy(), with_lines(task.eval_fn)
```

It returns one archive, one strategy, at most one operator, at most one eval function (the
default is `task.eval_fn`), and any other objects whose `@command` methods become commands.

## The evaluator contract

`task.evaluate` runs in a fresh directory: the environment plus the individual's genome files.

- **Metrics.** The last stdout line that is a JSON object is the result. Its numbers become
  objectives, `"feedback"` is text the operator reads next time, and `"usage"` (for example
  `{"cost_usd": 0.02}`) is added to the run's spending. Without a JSON line, `name: number` lines
  are read instead.
- **Rejection.** A non-zero exit is a failed evaluation, with no objectives. stderr (or stdout) is
  the feedback. A failed individual is recorded but never admitted.
- **Artifacts.** `$EVOKIT_WORKDIR` is the individual's artifact folder; files written there are
  kept at `artifacts/<id>/` in the run.
- **Time.** `task.timeout` bounds it. A timeout is a failed evaluation.

## An individual

`evokit get-individual ID` (and every command that returns individuals) prints:

```json
{
  "id": "c31fdf401d",
  "genome": {"files": ["main.py"], "digest": "db3cfd22..."},
  "parents": ["82f2e18f16"],
  "evaluation": {"objectives": {"score": -0.1}, "feedback": "solve() returned 2.9; the target is 3.0", "usage": {}},
  "info": {"note": "X to 2.9", "admitted": true}
}
```

`genome` lists the files, not their contents: read them with `evokit read-file ID PATH`, write
them out with `evokit checkout-individual ID DIR`, and compare two with `evokit diff-individuals A B`.
`evaluation` is `null` until the individual is evaluated, and has no objectives if it failed.
`info.admitted` says whether the archive kept it.
