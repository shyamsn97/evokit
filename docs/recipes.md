# Recipes

Each published algorithm is one file in
[`evokit/recipes`](https://github.com/shyamsn97/evokit/tree/main/evokit/recipes): a plain async
function that takes the task first, creates its experiment, and runs a loop over
`archive.sample`, `experiment.create` and `experiment.submit` ([primitives](primitives.md)).
Copy a recipe to change it.

```python
from evokit import recipes

task = ek.Task.from_directory("examples/circle_packing")
best = await recipes.avo(task, ek.ClaudeCode(), dir="experiments/avo-0", max_cost=5)
```

!!! note "Implementations, not official ports"
    These are EvoKit's implementations of the published algorithms, written from the papers and
    the official code. None is a one-to-one reproduction. Each page ends with a list of how the
    recipe differs from the original, compared against one specific commit (or the paper, when
    there is no code). Those lists are written by hand and can fall behind both sides. Results
    from a recipe are results of EvoKit's version, and should not be reported as reproductions
    of the original method.

| Recipe | Archive (default) | Selection | Who makes the child | Loop |
| --- | --- | --- | --- | --- |
| [autoresearch](recipes/autoresearch.md) | `SimplerBestArchive` | the best | a coding agent, a fresh session per change | sequential, until a limit |
| [AVO](recipes/avo.md) | `BestArchive(ties="accept")` | none: the agent works from the best | one persistent agent session, with a supervisor | one conversation, until a limit |
| [OpenEvolve](recipes/openevolve.md) | `IslandArchive` (MAP-Elites) | `islands` | an ensemble of LLMs writing diffs | workers in parallel |
| [ShinkaEvolve](recipes/shinka.md) | `IslandArchive(islands=2, features=())` | `parents` | LLMs chosen by a bandit, three edit kinds, novelty-filtered | workers in parallel |
| [Darwin Gödel Machine](recipes/dgm.md) | `UnboundedArchive` | `proportional` | the parent agent (`self_modify`) | workers in parallel |
| [DGM-Hyperagents](recipes/hyperagents.md) | `UnboundedArchive` | `proportional` | the parent's own meta agent (`meta_agent_modify`) | workers in parallel |
| [a generational GA](recipes/ga.md) | `PopulationArchive(10)` | `tournament(3)` | a mutation function | generational |
| [EvoRLM](recipes/evorlm.md) (experimental) | `PopulationArchive` | the root model | one recursive language model and the agents it starts | generations of concurrent agents |

From the shell, `evokit run RECIPE TASK_DIR --dir DIR` runs any of them; its options come from
the recipe's signature (`--max-cost 5`, `--agent claude`). See [tasks](tasks.md).
