# autoresearch

!!! note "An implementation, not the official code"
    See [how it differs](#how-it-differs-from-the-official-code).

[autoresearch](https://github.com/karpathy/autoresearch) (Karpathy, 2026) is one lineage. In
one session, an agent edits `train.py`, runs it, and keeps the commit only if `val_bpb` improved.

## The recipe

```python
async def autoresearch(task, agent, *, dir=None, archive=None, max_cost=None, max_evaluations=None):
    archive = archive or SimplerBestArchive(size="lines")      # ties go to shorter code
    experiment = task.create_experiment(dir, archive=archive, ...)
    while not experiment.limits.reached():
        parent = archive.top(1)[0]
        async with experiment.start(agent, start=parent, submit=False) as directory:
            turn = await agent.prompt(CHANGE, output=Change)   # "make one change; score it"
            genome = task.read(directory)
        child = Individual.new(genome, [parent.id], name=turn.output.title)
        await experiment.submit(child)                         # + a results.tsv row
    return experiment.best()
```

```python
task = ek.Task.from_directory("nanochat")   # genome train.py, objective val_bpb, minimize true
best = await recipes.autoresearch(task, ek.ClaudeCode(), dir="experiments/mar5", max_cost=20)
```

## How it differs from the official code

Compared with commit `228791f`:

- **A fresh session per change**, not one conversation. Earlier attempts are available through
  the read tools.
- **EvoKit evaluates the result.** The agent can score drafts but does not decide what is kept.
- **The keep rule is mechanical:** better, or tied within 1e-6 and fewer lines. In the original,
  simplicity is the agent's call.
- **No git.** Every version is in the experiment's history.
- **It stops** at `max_cost`, `max_evaluations`, or 10 sessions in a row with no change.
- **`results.tsv` columns differ:** `id, objective, status, name`, with no memory column.
- **No setup ritual** and no peak-VRAM tracking.

The same: one lineage, a fixed evaluator, and keep/discard/crash logging. [AVO](avo.md) is
closer to the original's single conversation.
