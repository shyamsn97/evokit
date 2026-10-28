# Tasks

A task folder defines one problem. The CLI and Python load the same folder into the same `Task`.
(In Python, `ek.Task(seed, evaluate)` also works without a folder, for genomes that are not
files.)

```text
packing/
  TASK.md        what agents and models are told: the goal, the rules, what "better" means
  config.yaml    which files evolve and how they are scored (task:), and optionally a recipe (recipe:)
  evaluate.py    the evaluator: part of the fixed environment, never evolved
  main.py        the seed: the genome files as they are now
```

`evokit init packing` writes this folder from the template in `evokit/templates/task/`. It is a
toy task that works unedited: `solve()` in `main.py` should return 3. Replace the four files
with your problem.

## From the shell

Every command takes `--dir`, the experiment's directory (default `.evokit`, or `$EVOKIT_DIR`).

```bash
evokit init packing
evokit run avo packing --dir experiments/0 --max-cost 5    # a recipe on the task
evokit run packing --dir experiments/0                     # the recipe named in config.yaml
evokit stats --dir experiments/0                           # evaluations, cost, the best
evokit log --dir experiments/0                             # the lineage; also show, diff, grep, notes
```

`evokit run RECIPE TASK_DIR` takes the recipe's own options, parsed from its signature
(`--max-cost 5`, `--max-evaluations 50`, `--stall-after 5`). A recipe that takes an agent gets
`--agent claude` (the default, Claude Code) or `--agent "COMMAND"` (any command, run in the
agent's directory, with `{prompt}` in it replaced by the prompt, or the prompt on stdin). `RECIPE` can also be `FILE.py:NAME`, a recipe of
your own. `evokit run` on an existing directory continues the experiment.

When an outside coding agent is the operator, it drives the experiment one command at a time:

```bash
evokit status --task packing --dir experiments/0             # creates and seeds experiments/0
evokit list-individuals --dir experiments/0
evokit checkout-individual ID work/ --dir experiments/0
# ... edit work/main.py ...
evokit evaluate-directory work/ --parent ID --name "one line" --dir experiments/0
```

The first command on a new directory needs `--task`: it creates the experiment, evaluates the
task's genome files as the seed, and adds the seed to a `BestArchive`. Later commands continue
it; the experiment records its task folder in `task.json`, so `--task` is not needed again. An
experiment belongs to one task: a different `--task` for an existing directory is an error.
`evokit skill export claude` (or `cursor`, `codex`) installs instructions that teach an agent
these commands.

If the seed fails its evaluation, the command exits with a usage error that includes the
evaluator's feedback, and nothing is recorded. Fix the files and run the command again. One
process drives an experiment at a time; a second one on the same directory exits with
`run_busy`. Read commands (`stats`, `log`, `show`, `diff`, `grep`, `notes`) never take the lock.

## From Python

```python
import evokit as ek
from evokit import recipes

task = ek.Task.from_directory("packing")
task.seed            # Files: the genome files as they are now
task.evaluate        # ScriptEvaluator: config.yaml's evaluate command and timeout
task.instructions    # the text of TASK.md
task.files           # Project: which files are genome, which are environment

best = await recipes.avo(task, ek.ClaudeCode(), dir="experiments/0", max_cost=5)
```

[`examples/circle_packing`](https://github.com/shyamsn97/evokit/tree/main/examples/circle_packing)
is a task folder, and its `algorithms/` scripts load it this way.

## Folder to objects

| In the folder | Becomes |
| --- | --- |
| `task.genome` | the genome patterns: the files that evolve, read as a `Files` value |
| everything else | the environment, read-only: `evaluate*`, `eval/**`, `config.yaml` and `TASK.md` are protected, and an agent that edits them is rejected |
| `task.evaluate`, `task.timeout` | `ScriptEvaluator(files, evaluate, timeout)`, the task's `evaluate` |
| `task.objective`, `task.minimize` | what ranked archives rank by |
| `TASK.md` | `task.instructions`, and the task section of an agent's system prompt |
| the genome files as they are | `task.seed` |
| `recipe:` | what `evokit run TASK_DIR` runs: `name`, and default options for it |

`runs/**`, `experiments/**` and `.evokit/**` inside the folder are ignored, and so is an
experiment directory placed anywhere inside it. From the CLI, the environment is snapshotted once
into `<experiment>/env`, so later edits to the folder do not change a running experiment.

## config.yaml

```yaml
task:
  genome: [main.py]             # glob patterns of the files that evolve
  evaluate: python evaluate.py  # run in a fresh copy of the folder
  timeout: 2m
  objective: score
  minimize: false
  # ignore: [data/cache/**]     # neither genome nor environment

recipe:                         # optional: what `evokit run THIS_FOLDER` runs
  name: avo                     # avo | autoresearch | dgm | ... | FILE.py:NAME
  agent: claude                 # claude (Claude Code), or a command run in the agent's folder
  max_cost: 5                   # any other option of the recipe
```

## The evaluator contract

`task.evaluate` runs in a fresh directory: the environment plus the individual's genome files.

- **Metrics.** The last stdout line that is a JSON object is the result. Its numbers become
  objectives, `"feedback"` is text the operator reads next time, and `"usage"` (for example
  `{"cost_usd": 0.02}`) is added to the experiment's spending. Without a JSON line,
  `name: number` lines are read instead.
- **Rejection.** A non-zero exit is a failed evaluation, with no objectives. stderr (or stdout) is
  the feedback. A failed individual is recorded but never admitted.
- **Artifacts.** `$EVOKIT_WORKDIR` is the individual's artifact folder; files written there are
  kept at `artifacts/<id>/` in the experiment.
- **Time.** `task.timeout` bounds it. A timeout is a failed evaluation.

## An individual

Commands that return individuals (`list-individuals`, `evaluate-directory`) print:

```json
{
  "id": "c31fdf401d",
  "name": "X to 2.9",
  "description": "solve() now returns 2.9, closer to the target",
  "genome": {"files": ["main.py"], "digest": "db3cfd22..."},
  "parents": ["82f2e18f16"],
  "evaluation": {"objectives": {"score": -0.1}, "feedback": "solve() returned 2.9; the target is 3.0", "usage": {}},
  "info": {"admitted": true}
}
```

`genome` lists the files, not their contents. Read an experiment like a git repository (see
[reading runs](reading-runs.md)): `evokit log [ID] [--all]` for the lineage, `evokit show ID
[PATH]` for one individual or one of its files, `evokit diff A [B]` for what changed, and
`evokit grep PATTERN` to find what was tried. `evokit checkout-individual ID DIR` writes one out.
`evaluation` is `null` until the individual is evaluated, and has no objectives if it failed.
`info.admitted` says whether the archive kept it.
