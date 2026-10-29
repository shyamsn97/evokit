# A generational GA

A textbook GA with no model: tournament selection into a fixed-size population. `mutate` maps a
genome to a new one.

```python
for _ in range(generations):
    parents = archive.sample(archive.size, tournament(3))
    children = await asyncio.gather(*(experiment.create(child(p)) for p in parents))
    await asyncio.gather(*(experiment.submit(c) for c in children if c))
```

See `examples/fixed_mutation`.
