# Tasks

A task folder defines one problem:

```text
packing/
  TASK.md        the goal, the rules, the scoring (what agents are told)
  config.yaml    which files evolve and how they are scored
  evaluate.py    the evaluator (fixed, never evolved)
  main.py        the seed
```

`evokit init packing` writes a toy version that runs as it is.

## config.yaml

```yaml
task:
  genome: [main.py]             # files that evolve; the rest is read-only
  evaluate: python evaluate.py  # run in a fresh copy of the folder
  timeout: 2m
  objective: score
  minimize: false

recipe:                         # optional: what `evokit run THIS_FOLDER` runs
  name: avo
  agent: claude
  max_cost: 5
```

## The evaluator

- Print metrics as a JSON object on the last stdout line. Numbers become objectives,
  `"feedback"` is shown next time, and `"usage"` adds to the cost.
- Exit non-zero to reject a program.
- Files written to `$EVOKIT_WORKDIR` are kept under `artifacts/<id>/`.
- `timeout` bounds it.

## From the shell

```bash
evokit run avo packing --dir experiments/0 --max-cost 5   # recipe options from its signature
evokit run packing --dir experiments/0                    # the recipe in config.yaml
```

`--agent claude` (the default) or `--agent "COMMAND"`. A recipe can also be `FILE.py:NAME`.

With an outside agent as the operator:

```bash
evokit status --task packing --dir experiments/0          # creates and seeds it
evokit list-individuals --dir experiments/0
evokit checkout-individual ID work/ --dir experiments/0
evokit evaluate-directory work/ --parent ID --name "what changed" --dir experiments/0
```

`--dir` defaults to `.evokit` (or `$EVOKIT_DIR`). Only the first command needs `--task`.

## From Python

```python
task = ek.Task.from_directory("packing")
best = await recipes.avo(task, ek.ClaudeCode(), dir="experiments/0", max_cost=5)
```
