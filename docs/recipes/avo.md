# AVO

!!! note "An implementation, not the official code"
    NVIDIA released no code. See [how it differs](#how-it-differs-from-the-paper-and-the-open-reproduction).

*AVO: Agentic Variation Operators* (NVIDIA, 2026, [arXiv:2603.24517](https://arxiv.org/abs/2603.24517)).
The variation step is one agent run, `Vary(P_t) = Agent(P_t, K, f)`: the agent reads the
lineage, calls the scorer itself, and commits a version when it is correct and at least as
good as the best. One conversation lasts the whole run, and a supervisor steps in on stalls.

## The recipe

| The paper | Here |
| --- | --- |
| lineage P_t | the `log`, `show`, `diff`, `grep` tools |
| knowledge base K | not modeled; `TASK.md` and the task's files |
| scorer f | the `score` tool (drafts, unlimited) |
| commit | the `submit` tool, into `BestArchive(ties="accept")` |
| supervisor | after `stall_after=5` attempts with no improvement, a read-only reviewer suggests 3 to 5 directions |

```python
async with experiment.start(agent, resume=resume):
    while not experiment.limits.reached():
        async for event in agent.prompt(prompt):
            if stalled(experiment, stall_after):
                experiment.note("review")
                if directions := await review(experiment, reviewer or agent.fresh()):
                    agent.send(directions)
        prompt = CONTINUE.format(best=brief(experiment.best()))
```

Limits are checked on every tool call and before every turn, and each turn's budget is what
remains of `max_cost`. A stall counts drafts as well as submissions: the agent submits only what
its drafts showed was better, so a count of submissions alone would miss plateaus.

## How it differs from the paper and the open reproduction

Versus the paper:

- No knowledge base.
- The lineage is EvoKit's history, not git.
- Rejected submissions are recorded, and the workspace is not reset.
- Correctness is up to the task's evaluator; there is no separate flag.
- The stall rule and the reviewer's setup are EvoKit's choices.
- It stops on cost, evaluations or failures, not after 7 days.

Versus [gatordevin/avo](https://github.com/gatordevin/avo) (`f6dad9e`):

- That repo runs a fresh agent per step; here one session submits many times.
- Its default mode is your own Claude Code/Codex session acting as the operator.
- It ships a knowledge base and puts a lineage table in every prompt.
- It reverts on reject, and drafts in place.
- Its supervisor fires after 3 steps and can use another model.
- It stops on steps and time.

The same in all three: the agent chooses what to read, edit and score; scoring is uncapped;
commits must be correct and no worse; and there is a single lineage.
