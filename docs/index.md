# EvoKit

Evolutionary search where the variation operator can be a one-shot LLM call, a headless
coding agent, or the coding agent you are talking to.

!!! warning "Work in progress"
    EvoKit is early research code. The API changes often, without deprecations, and the on-disk
    experiment format may change between commits. Pin a commit if you depend on it.

!!! note "The recipes are implementations, not official ports"
    Each recipe is EvoKit's reading of a published algorithm, written from the paper and, where
    there is one, the official code. None of them is a one-to-one reproduction: prompts,
    defaults, selection details, budgets and infrastructure differ, and several features of the
    originals are left out. Every recipe page ends with how it differs from the original. Results
    from a recipe are results of EvoKit's version, not of the original method.

## The idea

A `Task` says what evolves and how it is scored. An `Experiment` is one search on it, kept in a
directory. A recipe is a plain async function that takes the task and every component, creates
the experiment, and loops: pick individuals from the archive, make a child, submit it.

```python
import evokit as ek
from evokit.operators import rewrite_text

task = ek.Task(seed="You are a helpful assistant.", evaluate=score)    # score(individual) -> Evaluation
experiment = task.create_experiment("experiments/prompt", max_cost=5)  # seeded; continued if it exists
while not experiment.limits.reached():
    parent, *others = experiment.archive.sample(3)
    if child := await experiment.create(rewrite_text(model, parent, others)):
        await experiment.submit(child)
```

## The docs

- [Primitives](primitives.md): `Task`, `Experiment`, archives, strategies, variation functions,
  agents, storage and the history.
- [Recipes](recipes.md): one loop per published algorithm, each with how it differs from the
  original.
- [Tasks](tasks.md): defining a problem as a task folder, and running it from the shell.
- [Reading runs](reading-runs.md): reading an experiment with `log`, `show`, `diff` and `grep`.

The [README](https://github.com/shyamsn97/evokit#readme) has the install, the examples and the
open design work. `docs/research/` and `docs/design/` hold working notes from earlier designs:
they explain why things are the way they are, but most of their code no longer matches the
library.
