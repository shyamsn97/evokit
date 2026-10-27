# autoresearch

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

`evokit/recipes/autoresearch.py` is the same single lineage with evokit doing the bookkeeping. A
fresh agent session makes each change, evokit evaluates it, and the archive decides what is kept.

```python
async def autoresearch(project, eval_fn, agent, *, storage, archive, instructions=None,
                       objective="val_bpb", results="results.tsv", max_evaluations=None,
                       max_cost=None, max_failures=10):
    strategy = Greedy()
    operator = CodingAgentOperator(agent, project, instructions=instructions)
    while not stop_reason():                              # max_evaluations, max_cost, or max_failures in a row
        asked = await evokit.ask(storage, archive, strategy, operator)
        evaluations = await asyncio.gather(*map(score, asked.individuals))
        admitted = await evokit.tell(storage, archive, asked.individuals, evaluations)
        for child in evaluated:                           # one TSV row: id, objective, keep/discard/crash, name
            tsv.write(row(child, status, objective))
    return archive.best()
```

```python
project = Project(".", genome=["train.py"])
eval_fn = with_lines(ScriptEvaluator(project, "uv run train.py", timeout="10m"))
storage = DirectoryStorage("runs/mar5")
archive = SimplerBestArchive(objective="val_bpb", minimize=True, size="lines", storage=storage)
if not archive.members():
    seed = Individual.new(project.seed_genome())
    await evokit.tell(storage, archive, [seed], [await evokit.as_eval_fn(eval_fn)(seed)])

await recipes.autoresearch(project, eval_fn, ClaudeCode(), storage=storage, archive=archive,
                           instructions=open("program.md").read())
```

## How it differs from the official code

Compared with [karpathy/autoresearch](https://github.com/karpathy/autoresearch) at commit
`228791f`, whose whole method is `program.md`:

- **One conversation versus a fresh session per change.** The original agent runs every
  experiment in one session and remembers what it tried and why it failed. This recipe starts a
  new session for each change (`CodingAgentOperator` with `persistent=False`), told only the
  current best's score and feedback. With a `DirectoryStorage`, the agent can read the earlier
  attempts with `evokit log`, `show`, `diff` and `grep`, but only if it chooses to.
- **The agent does not run the experiment.** In the original, the agent runs
  `uv run train.py > run.log`, reads `val_bpb` out of the log, and decides. Here the agent only
  edits; evokit runs `eval_fn` after the session ends. The recipe gives the agent no draft
  evaluation, so it cannot test a change before handing it back.
- **No fix-and-retry on a crash.** The original agent reads `tail -n 50 run.log`, fixes typos
  and simple bugs in place, and gives up after a few tries. Here a crash is logged as `crash`, and
  the next change starts a new session from the best version.
- **The keep rule is mechanical.** The original keeps a commit only if `val_bpb` went down, and
  otherwise `git reset`s. Simplicity is left to the agent's judgment ("an improvement of ~0 but
  much simpler code? Keep"). Here `SimplerBestArchive` keeps a change if it is better, or tied
  within 1e-6 with fewer non-blank, non-comment lines. Nothing is kept or reverted by the agent.
- **No git.** The original works on an `autoresearch/<tag>` branch and resets rejected commits.
  Here every version, kept or not, is in the run's history, and the next parent is the archive's
  best.
- **It stops.** The original says never stop until the human interrupts. This recipe stops at
  `max_evaluations`, `max_cost`, or `max_failures` (default 10) sessions in a row that change
  nothing.
- **`results.tsv` has different columns.** The original's agent writes
  `commit, val_bpb, memory_gb, status, description` with a header and keeps it out of git. This
  recipe appends `id, objective, status, name` with no header and no memory column.
- **No setup ritual.** The original has the agent pick a run tag, create the branch, read the
  files, check the data cache, write the TSV header, confirm with the human, and run the
  unmodified `train.py` first. Here the caller seeds the archive by evaluating the seed program,
  and the loop starts.
- **Peak VRAM is not tracked.** The original reports it and treats it as a soft constraint.

The same: one lineage, one incumbent, a fixed evaluator the agent cannot touch, every attempt
logged with keep, discard or crash, and the agent working unattended.

[AVO](avo.md) is closer to the original's memory: one conversation, in which the agent evaluates
and commits itself.

With an outside agent there is no loop at all. The task is a folder ([tasks](../tasks.md)), and
the agent calls the same functions: `evokit select-individuals` prints the parent's selection,
and `evokit evaluate-directory DIR --parent ID` records, evaluates and tells the agent's program.
