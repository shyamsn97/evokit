# EvoKit

Evolutionary search where the variation operator can be an LLM call, a headless coding agent,
or the coding agent you are talking to.

> **Work in progress.** The API and the on-disk format change often; pin a commit.
>
> **Recipes are implementations, not official ports.** They follow the papers and official code
> but are not one-to-one reproductions. Each recipe's page lists the differences.

```python
import evokit as ek
from evokit.operators import rewrite_text

task = ek.Task(seed="You are a helpful assistant.", evaluate=score)    # score(individual) -> Evaluation
experiment = task.create_experiment("experiments/prompt", max_cost=5)  # seeded; continued if it exists
while not experiment.limits.reached():
    parent, *others = experiment.archive.sample(3)
    if child := await experiment.create(rewrite_text(model, parent, others)):
        await experiment.submit(child)                                 # scored, offered to the archive
```

- **`Task`**: a seed genome (any JSON value, or `Files`) and an `evaluate` function.
  `Task.from_directory(dir)` reads a task folder.
- **`Experiment`**: one search, kept in a directory: its history, archive and limits. Every cost
  is recorded, and reopening a directory continues it.
- **`Archive`**: the population. `sample(n, strategy)` picks members; `top(n)` gives the best.
- **Variation**: a function returning `(child, usage)` (`rewrite_text`, `diff`, `rewrite_files`,
  ...), or an agent started with `experiment.start(agent)`.

```bash
pip install -e ".[mcp]"        # or ".[dev]" for the tests
```

## Agents

```python
async with experiment.start(agent):              # a checkout of the best, plus tools
    while not experiment.limits.reached():
        async for event in agent.prompt("Improve it; submit what is better."):
            if experiment.since_improvement() > 5:
                agent.send("Try a different data structure.")
```

The agent gets tools to read the experiment, `score` drafts, and `submit` versions. Turns are
recorded, and the agent is stopped when a limit is reached. Any harness works:

```python
from evokit.agents import ClaudeCode, CommandAgent, ReplAgent

ClaudeCode(model="sonnet", auth="login")              # headless `claude -p`
CommandAgent(["codex", "exec", "{prompt}"])           # any CLI agent: Codex, cursor-agent, your script
ReplAgent(OpenAIModel("gpt-5-mini"))                  # any chat model, writing Python in a REPL
```

You can also turn it around and let your own agent drive EvoKit: export the skill
(`evokit skill export claude|cursor|codex`), then the agent uses the CLI, or the same commands
over MCP (`claude mcp add evokit -- evokit mcp --dir experiments/0 --task packing`). See
[docs/agents.md](docs/agents.md).

## Recipes

```python
from evokit import recipes

task = ek.Task.from_directory("examples/circle_packing")
best = await recipes.avo(task, ek.ClaudeCode(), dir="experiments/avo-0", max_cost=5)
```

| Recipe | Based on | Differences |
| --- | --- | --- |
| `autoresearch` | [karpathy/autoresearch](https://github.com/karpathy/autoresearch) | [docs](docs/recipes/autoresearch.md#how-it-differs-from-the-official-code) |
| `avo` | [AVO paper](https://arxiv.org/abs/2603.24517) (no official code) | [docs](docs/recipes/avo.md#how-it-differs-from-the-paper-and-the-open-reproduction) |
| `openevolve` | [codelion/openevolve](https://github.com/codelion/openevolve) | [docs](docs/recipes/openevolve.md#how-it-differs-from-the-official-code) |
| `shinka` | [SakanaAI/ShinkaEvolve](https://github.com/SakanaAI/ShinkaEvolve) | [docs](docs/recipes/shinka.md#how-it-differs-from-the-official-code) |
| `dgm` | [paper](https://arxiv.org/abs/2505.22954), [jennyzzt/dgm](https://github.com/jennyzzt/dgm) | [docs](docs/recipes/dgm.md#how-it-differs-from-the-official-code) |
| `hyperagents` | [paper](https://arxiv.org/abs/2603.19461), [facebookresearch/Hyperagents](https://github.com/facebookresearch/Hyperagents) | [docs](docs/recipes/hyperagents.md#how-it-differs-from-the-official-code) |
| `ga` | a textbook GA | none |
| `evorlm` (experimental, needs rlmflow) | [shyamsn97/evorlm](https://github.com/shyamsn97/evorlm) | [docs](evokit/recipes/experimental/evorlm.md#differences-from-the-original) |

Each recipe is one file in `evokit/recipes/`. To change one, copy it.

## From the shell

```bash
evokit init packing                                       # a task folder that runs as it is
evokit run avo packing --dir experiments/0 --max-cost 5   # options come from the recipe's signature
evokit stats --dir experiments/0
evokit log --dir experiments/0                            # also show, diff, grep, notes
```

`--agent claude` (the default) or `--agent "codex exec {prompt}"` picks the harness. With an
outside agent as the operator, it runs `evokit list-individuals`, `checkout-individual ID DIR`,
then `evaluate-directory DIR --parent ID`. See [docs/tasks.md](docs/tasks.md).

## Examples

- [`examples/circle_packing`](examples/circle_packing): 26 circles in the unit square with
  autoresearch, AVO, OpenEvolve and DGM, plus ablations. `compare.py` runs and reports them.
- [`examples/co_bench`](examples/co_bench): a recipe on one
  [CO-Bench](https://github.com/sunnweiwei/CO-Bench) problem, scored on its held-out test set.
- [`examples/fixed_mutation`](examples/fixed_mutation/main.py): `recipes.ga` with no model.

## Docs

[Primitives](docs/primitives.md), [agents](docs/agents.md), [recipes](docs/recipes.md), [tasks](docs/tasks.md),
[reading runs](docs/reading-runs.md). Build them with
`pip install -r docs/requirements.txt && mkdocs serve`. `docs/research/` and `docs/design/` are
old notes.

## TODO

- [ ] Several processes on one experiment (today a second driver exits with `run_busy`).
- [ ] A sandbox for evaluators and agents.
- [ ] Re-evaluation, for noisy benchmarks and Hyperagents-style transfer.
- [ ] Evaluation caching.
- [ ] Sampling that sees what is in flight.
- [ ] Remote storages (`S3Storage`, `ModalStorage`).
- [ ] Multi-objective archives.
- [ ] The agent's `evokit` shim fails when evokit is not installed.
- [ ] Re-check recipes against their originals.
- [ ] API reference pages; fix the cookiecutter `Makefile`.

## Development

```bash
pip install -e ".[dev]"
pytest                      # offline: scripted agents and models stand in for LLMs
```

CLI output is JSON when stdout is not a TTY; errors are JSON on stderr. Exit codes: 0 ok,
1 internal error, 2 usage error.
