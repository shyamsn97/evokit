# autoresearch

!!! note "An implementation, not the official code"
    This is EvoKit's version of autoresearch, not a port of `program.md`. See
    [how it differs from the official code](#how-it-differs-from-the-official-code).

[autoresearch](https://github.com/karpathy/autoresearch) (Karpathy, 2026) is one lineage. All of
the original is `program.md`, a prompt telling one agent to run, in one session:

```text
LOOP FOREVER:
1. Look at the git state: the current branch/commit we're on
2. Tune train.py with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: uv run train.py > run.log 2>&1
5. Read out the results: grep "^val_bpb:\|^peak_vram_mb:" run.log
...
8. If val_bpb improved (lower), you "advance" the branch, keeping the git commit
9. If val_bpb is equal or worse, you git reset back to where you started
```

- One conversation for the whole run: "do NOT pause to ask the human if you should continue."
- Every run of the experiment is a recorded attempt (`results.tsv`: commit, val_bpb, memory,
  keep/discard/crash, description). There are no drafts.
- Keep only if strictly better. Equal or worse is reset.
- The simplicity rule is the agent's judgment: "A 0.001 val_bpb improvement that adds 20 lines
  of hacky code? Probably not worth it. ... An improvement of ~0 but much simpler code? Keep."
- No supervisor, no knowledge base beyond the repo, and no lineage beyond the git branch.

## The recipe

`evokit/recipes/autoresearch.py` is the same single lineage with EvoKit doing the bookkeeping. A
fresh agent session makes each change from the best version, EvoKit evaluates it, and the
archive decides what is kept.

```python
async def autoresearch(task, agent, *, dir=None, archive=None, max_cost=None, max_evaluations=None):
    if task.files is not None:
        task = replace(task, evaluate=with_lines(task.evaluate))     # adds a "lines" objective
    archive = archive or SimplerBestArchive(size="lines")
    experiment = task.create_experiment(dir, archive=archive, max_cost=max_cost, max_evaluations=max_evaluations)
    while not experiment.limits.reached():
        parent = archive.top(1)[0]
        async with experiment.start(agent, start=parent, submit=False) as directory:
            turn = await agent.prompt(CHANGE, output=Change)         # "make one change; score it"
            genome = task.read(directory)
        if genome == parent.genome:
            experiment.note("failed", "the agent made no changes")
            continue
        child = Individual.new(genome, [parent.id], name=turn.output.title, ...)
        child = await experiment.submit(child)
        experiment.storage.append("results.tsv", row(child, status, task.objective))   # keep / discard / crash
    return experiment.best()
```

The task is a folder whose `TASK.md` plays the part of `program.md`:

```python
task = ek.Task.from_directory("nanochat")     # config.yaml: genome [train.py], evaluate "uv run train.py",
                                              # objective val_bpb, minimize true
best = await recipes.autoresearch(task, ek.ClaudeCode(), dir="experiments/mar5", max_cost=20)
```

## How it differs from the official code

Compared with [karpathy/autoresearch](https://github.com/karpathy/autoresearch) at commit
`228791f`, whose whole method is `program.md`:

- **One conversation versus a fresh session per change.** The original agent runs every
  experiment in one session and remembers what it tried and why it failed. This recipe starts a
  new session for each change, in a checkout of the current best. The agent can read the earlier
  attempts with the `log`, `show`, `diff`, `grep` and `notes` tools, but only if it chooses to.
- **The agent scores drafts; EvoKit records the result.** In the original, the agent runs
  `uv run train.py > run.log`, reads `val_bpb` out of the log, and decides what to keep. Here
  the agent scores its edits with the `score` tool (the task's evaluator, recorded as a draft and
  never kept), and is told to put back the best version it measured. After the session ends,
  EvoKit evaluates what the agent left and the archive decides.
- **Fix-and-retry happens only inside a session.** The original agent reads `tail -n 50 run.log`,
  fixes typos and simple bugs in place, and gives up after a few tries. Here the agent can do the
  same with drafts, but once its session ends, a version that crashes is logged as `crash`, and
  the next change starts a new session from the best version.
- **The keep rule is mechanical.** The original keeps a commit only if `val_bpb` went down, and
  otherwise `git reset`s. Simplicity is left to the agent's judgment ("an improvement of ~0 but
  much simpler code? Keep"). Here `SimplerBestArchive` keeps a change if it is better, or tied
  within 1e-6 with fewer non-blank, non-comment lines. Nothing is kept or reverted by the agent.
- **No git.** The original works on an `autoresearch/<tag>` branch and resets rejected commits.
  Here every version, kept or not, is in the run's history, and the next parent is the archive's
  best.
- **It stops.** The original says never stop until the human interrupts. This recipe stops at
  `max_evaluations`, `max_cost`, or 10 sessions in a row that change nothing. Limits are checked
  between changes, so the last change can go over.
- **`results.tsv` has different columns.** The original's agent writes
  `commit, val_bpb, memory_gb, status, description` with a header and keeps it out of git. This
  recipe appends `id, objective, status, name` with no header and no memory column.
- **No setup ritual.** The original has the agent pick a run tag, create the branch, read the
  files, check the data cache, write the TSV header, confirm with the human, and run the
  unmodified `train.py` first. Here `create_experiment` evaluates the task's files as the seed,
  and the loop starts.
- **Peak VRAM is not tracked.** The original reports it and treats it as a soft constraint.

The same: one lineage, one incumbent, a fixed evaluator the agent cannot touch, every attempt
logged with keep, discard or crash, and the agent working unattended.

[AVO](avo.md) is closer to the original's memory: one conversation, in which the agent evaluates
and commits itself.

With an outside agent there is no loop at all. The task is a folder ([tasks](../tasks.md)), and
the agent calls the same functions from the shell: `evokit list-individuals` and
`evokit checkout-individual ID DIR` give it the parent, and
`evokit evaluate-directory DIR --parent ID` evaluates its program and offers it to the archive.
