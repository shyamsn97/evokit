# EvoKit

Evolutionary search where the variation operator can be an LLM call, a headless coding agent,
or the coding agent you are talking to.

!!! warning "Work in progress"
    The API and the on-disk format change often; pin a commit.

!!! note "Recipes are implementations, not official ports"
    They follow the papers and official code but are not one-to-one reproductions. Each recipe
    page lists the differences.

```python
import evokit as ek
from evokit.operators import rewrite_text

task = ek.Task(seed="You are a helpful assistant.", evaluate=score)
experiment = task.create_experiment("experiments/prompt", max_cost=5)
while not experiment.limits.reached():
    parent, *others = experiment.archive.sample(3)
    if child := await experiment.create(rewrite_text(model, parent, others)):
        await experiment.submit(child)
```

- [Primitives](primitives.md): the pieces.
- [Agents](agents.md): Claude Code, Codex, any CLI or model, inside the library or driving it.
- [Recipes](recipes.md): the published algorithms.
- [Tasks](tasks.md): task folders and the CLI.
- [Reading runs](reading-runs.md): `log`, `show`, `diff`, `grep`.
