# DGM-Hyperagents

!!! note "An implementation, not the official code"
    It ships no domains and no seed meta agent. See [how it differs](#how-it-differs-from-the-official-code).

[Hyperagents](https://arxiv.org/abs/2603.19461) (Zhang et al., 2026) puts the task agent and a
meta agent in one repository. The parent's meta agent edits the whole repository, itself
included, so the improvement procedure evolves too.

```python
async def hyperagents(task, *, fixed_meta_agent=None, **dgm_options):
    modify = partial(meta_agent_modify, fixed_meta_agent=fixed_meta_agent, objective=task.objective)
    return await dgm(task, modify=modify, **dgm_options)
```

The meta agent runs as `python meta_agent.py --repo REPO --history HISTORY`. `HISTORY` holds
`archive.jsonl` (every archived agent's lineage and score) and the parent's logs. Ablations:
`fixed_meta_agent=seed_repo` and `open_ended=False`.

## How it differs from the official code

Compared with commit `59a68f6`:

- **A loop, not a system:** no domains, splits, Docker, ensembles or baselines.
- **Full copies** instead of patch chains in git worktrees.
- **Parallel workers** instead of serial generations.
- **A simpler parent rule:** a fixed 0.5 midpoint and 1/(1+children), and no editable selection.
- **The meta agent sees less:** `archive.jsonl` and the parent's logs, with a 30-minute timeout.
- **Generic staging, and one domain at a time.**
- **No ensemble, held-out test or transfer.**
- **Failed meta runs** are `failed` notes; they do not mark the parent invalid.

The same: one repository with a task agent and a self-editing meta agent, an open-ended archive,
the child-discounted parent rule, staged evaluation, and both ablations.
