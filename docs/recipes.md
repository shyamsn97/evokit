# Recipes

Each recipe is one async function in `evokit/recipes/`: it takes the task, creates an
experiment, and loops over `archive.sample`, `experiment.create` and `experiment.submit`. Copy
one to change it.

```python
best = await recipes.avo(task, ek.ClaudeCode(), dir="experiments/avo-0", max_cost=5)
```

!!! note "Implementations, not official ports"
    Each page ends with how the recipe differs from the original, checked against one commit.

| Recipe | Archive | Selection | Child made by |
| --- | --- | --- | --- |
| [autoresearch](recipes/autoresearch.md) | `SimplerBestArchive` | the best | a fresh agent session per change |
| [AVO](recipes/avo.md) | `BestArchive(ties="accept")` | the best | one persistent agent session |
| [OpenEvolve](recipes/openevolve.md) | `IslandArchive` | `islands` | an LLM ensemble writing diffs |
| [ShinkaEvolve](recipes/shinka.md) | `IslandArchive(2)` | `parents` | bandit-chosen LLMs, three edit kinds |
| [DGM](recipes/dgm.md) | `UnboundedArchive` | `proportional` | the parent agent itself |
| [Hyperagents](recipes/hyperagents.md) | `UnboundedArchive` | `proportional` | the parent's meta agent |
| [GA](recipes/ga.md) | `PopulationArchive(10)` | `tournament(3)` | a mutation function |
| [EvoRLM](recipes/evorlm.md) (experimental) | `PopulationArchive` | the root model | a recursive LM |

From the shell: `evokit run RECIPE TASK_DIR --dir DIR [--option value]`.
