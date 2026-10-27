# A generational GA

`evokit/recipes/ga.py` is a fixed-size population with tournament selection, one generation at
a time. The caller seeds a `PopulationArchive`. It is a textbook GA, not a port of any one
implementation, so there is no official code to compare it with.

```python
async def ga(eval_fn, mutate, *, storage, archive, generations=100):
    strategy = Tournament(3)
    for _ in range(generations):
        asked = await evokit.ask(storage, archive, strategy, mutate, n=archive.size)
        evaluations = await asyncio.gather(*map(score, asked.individuals))
        await evokit.tell(storage, archive, asked.individuals, evaluations)
    return archive.best()
```
