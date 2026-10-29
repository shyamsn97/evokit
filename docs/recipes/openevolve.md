# OpenEvolve

!!! note "An implementation, not the official code"
    See [how it differs](#how-it-differs-from-the-official-code).

[OpenEvolve](https://github.com/codelion/openevolve) is an open AlphaEvolve: islands of
MAP-Elites grids, a parent plus inspirations in the prompt, an LLM ensemble writing diffs, and
ring migration.

```python
async def worker():
    while not experiment.limits.reached():
        parent, *others = archive.sample(1 + inspirations, islands())
        name = archive.rng.choices(list(models), odds)[0]
        if child := await experiment.create(diff(models[name], parent, others, task=task)):
            await experiment.submit(child)
        if experiment.since_note("migrate") >= migrate_every:
            experiment.note("migrate")
            archive.migrate(top=migrate_top)
```

```python
best = await recipes.openevolve(task, {"fast": haiku, "strong": sonnet}, {"fast": 0.8, "strong": 0.2},
                                dir="experiments/oe-0")
```

Cascade evaluation belongs in the task's `evaluate`.

## How it differs from the official code

Compared with commit `515d1bc` and its default config:

- **A thinner prompt:** instructions, the parent's scores, and the code. No history, top or
  diverse program sections.
- **No elite archive:** exploitation picks from the island's top 3.
- **Migration:** every 20 evaluations, top 2, to the next island only. The original uses 10% of
  each island to both neighbours, every 50 generations.
- **4 islands and 4 inspirations**, where the original uses 5 and 2.
- **Features:** line count and `difflib` diversity on fixed ranges.
- **Left out:** the novelty check, the population cap, LLM feedback, checkpoints, full-rewrite
  mode and the size limit.
- **EVOLVE-BLOCK markers are always enforced.**
- **One process with asyncio workers**, where the original uses a process pool.
- **It stops** at `max_evaluations` (1,000) or `max_cost`.

The same: islands of MAP-Elites grids, the 20/70/10 parent split, a weighted LLM ensemble,
SEARCH/REPLACE diffs, and ring migration.
