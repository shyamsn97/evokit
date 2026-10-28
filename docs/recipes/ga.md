# A generational GA

`evokit/recipes/ga.py` is a fixed-size population with tournament selection, one generation at
a time. It needs no model: `mutate` is a plain function from a genome to a new one. It is a
textbook GA, not a port of any one implementation, so there is no official code to compare it
with.

```python
async def ga(task, mutate, *, dir=None, archive=None, generations=100, tournament_size=3,
             max_evaluations=None, max_cost=None, seed=None):
    archive = archive or PopulationArchive(10)
    experiment = task.create_experiment(dir, archive=archive, max_evaluations=max_evaluations, max_cost=max_cost)

    async def child(parent):
        return Individual.new(mutate(parent.genome), [parent.id]), {}

    for _ in range(generations):
        if experiment.limits.reached():
            break
        parents = archive.sample(archive.size, tournament(tournament_size))
        children = await asyncio.gather(*(experiment.create(child(p)) for p in parents))
        await asyncio.gather(*(experiment.submit(c) for c in children if c is not None))
    return experiment.best()
```

[`examples/fixed_mutation`](https://github.com/shyamsn97/evokit/tree/main/examples/fixed_mutation/main.py)
evolves a vector with two mutations.
