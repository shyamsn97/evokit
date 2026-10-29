# Darwin Gödel Machine

!!! note "An implementation, not the official code"
    This does not reproduce the paper's SWE-bench setup. See [how it differs](#how-it-differs-from-the-official-code).

In the [Darwin Gödel Machine](https://arxiv.org/abs/2505.22954), the genome is a coding agent.
Every working agent is kept. Parents are sampled by score, discounted by their children. The
parent reads its own evaluation logs and rewrites itself (`self_modify`).

```python
async def worker():
    while not experiment.limits.reached():                    # includes max_children=80
        parents = archive.sample(1, proportional(sharpness, midpoint))
        if child := await experiment.create(self_modify(parents[0], experiment)):
            await experiment.submit(child)
```

```python
evaluate = staged_eval(run_10_tasks, run_50_tasks, promote=lambda e: e.objectives.get("score", 0) >= 0.4)
task = ek.Task(seed=Files.from_directory("seed_agent"), evaluate=evaluate, instructions="...")
best = await recipes.dgm(task, dir="experiments/dgm-0")
```

The evaluation should return logs under `logs/`, which the next diagnosis reads. Ablations:
`fixed_agent=seed_agent` (no self-improvement) and `open_ended=False` (latest agent only).

## How it differs from the official code

Compared with commit `a565fd2`:

- **Benchmark-agnostic:** no SWE-bench or Docker pipeline.
- **The parent diagnoses itself:** there is no separate o1 diagnosis and no answer patch.
- **Full copies on the host** instead of patch chains in containers.
- **Continuous workers** instead of generations.
- **Parent rules:** only `proportional` (which also skips solved parents) and latest-only.
- **Admission:** anything with a score; no `keep_better`.
- **Staging is the caller's** `staged_eval`.
- **Left out:** post-improvement diagnosis, and the official seed agent and ablation flags.
- **Budgets:** `max_cost` and 10 failures in a row.

The same: an open-ended archive, the default parent rule (λ=10, midpoint 0.5, 1/(1+children)),
self-improvement from its own logs, staged evaluation, and both ablations.
