# Recipes

Each published algorithm is one file in [`evokit/recipes`](../evokit/recipes): a complete loop
over [`ask`, `evaluate` and `tell`](primitives.md). Copy a recipe to change it.

| Recipe | Archive the caller passes | Strategy | Who creates the child | Loop |
| --- | --- | --- | --- | --- |
| [autoresearch](recipes/autoresearch.md) | `SimplerBestArchive` | `Greedy` | a coding agent, a fresh session per change | sequential, until a limit |
| [AVO](recipes/avo.md) | `BestArchive` | none: the agent commits the best's child | one persistent agent session, redirected | sequential, until a limit |
| [OpenEvolve](recipes/openevolve.md) | `IslandArchive` (MAP-Elites) | `IslandSampler` | an ensemble of LLMs writing diffs | workers in parallel |
| [ShinkaEvolve](recipes/shinka.md) | `IslandArchive` | `ShinkaParents` | LLMs chosen by a bandit, three edit kinds, novelty-filtered | workers in parallel |
| [Darwin Gödel Machine](recipes/dgm.md) | `UnboundedArchive` | `ProportionalSelection` | the parent agent (`SelfModify`) | workers in parallel |
| [DGM-Hyperagents](recipes/hyperagents.md) | `UnboundedArchive` | `ProportionalSelection` | the parent's own meta agent (`MetaAgentModify`) | workers in parallel |
| [a generational GA](recipes/ga.md) | `PopulationArchive` | `Tournament` | a mutation operator | generational |
