# Primitives

A recipe is a loop over a few pieces. No base classes and no hooks: to change a recipe, edit it.

```python
task = ek.Task(seed=genome, evaluate=score)
experiment = task.create_experiment("experiments/0", max_cost=5)
while not experiment.limits.reached():
    parent, *others = experiment.archive.sample(3)
    if child := await experiment.create(vary(parent, others)):
        await experiment.submit(child)
```

## Records

```python
@dataclass(frozen=True)
class Individual:
    id: str
    genome: Any                  # a JSON value, or a type with to_json/from_json (e.g. Files)
    parents: tuple[str, ...]
    info: Mapping[str, Any]      # e.g. {"operator": "diff", "admitted": True}
    evaluation: Evaluation | None
    name: str                    # like a commit subject
    description: str             # like a commit body

@dataclass(frozen=True)
class Evaluation:
    objectives: dict[str, float] # empty means it failed
    feedback: str                # e.g. a traceback
    usage: dict[str, float]      # cost_usd, tokens, seconds, ...
    logs: Mapping[str, bytes]    # kept under artifacts/<id>/
```

## Task

```python
task = ek.Task(seed=1.0, evaluate=score, instructions="...", objective="score", minimize=False)
task = ek.Task.from_directory("examples/circle_packing")    # see tasks.md
```

`evaluate(individual) -> Evaluation`, sync or async, is the only source of scores.
`@ek.evaluator(timeout="10m")` turns crashes and timeouts into failed evaluations.

## Experiment

```python
experiment = task.create_experiment(dir, archive=ek.BestArchive(), max_cost=5,
                                    max_evaluations=None, max_failures=10, stop=[rule])
experiment = ek.open_experiment(dir)
```

A new experiment scores the seed first. An existing directory is continued: the archive replays
its log and unevaluated children are resubmitted.

- `create(make)` awaits a `(child, usage)` and records it. An `OperatorError` becomes a
  `failed` note and returns `None`.
- `submit(individual)` scores it with `task.evaluate`, offers it to the archive, and records both.
- `note(kind, text, usage)` and `discard(individual, reason)` record anything else.
- `limits.reached()` returns why to stop (or `None`). A stop rule is
  `rule(experiment) -> str | None`.
- Reading: `best()`, `cost()`, `counts()`, `evaluated()`, `admitted()`, `pending()`,
  `notes(kind)`, `since_improvement()`, `since_note(kind)`, `best_over_time()`, and
  `log`/`show`/`diff`/`grep`.

Every cost is recorded, so `cost()` covers the whole search across restarts. One process drives
an experiment at a time; reading needs no lock.

## Archives

| Archive | Keeps |
| --- | --- |
| `BestArchive(ties=)` | the best; `ties="accept"` lets equal scores replace it |
| `SimplerBestArchive(size="lines")` | the best; ties go to the smaller one |
| `PopulationArchive(size)` | the top `size` |
| `UnboundedArchive()` | everyone with a score |
| `LatestArchive()` | the newest with a score |
| `NoveltyArchive(behaviour)` | individuals far from their nearest members |
| `IslandArchive(islands, features)` | islands of MAP-Elites grids; `migrate(top)` |

A subclass writes `add(individual) -> bool`, using `insert`, `remove` and `set_state`. Each
change is appended to `archive/changes.jsonl`, so an archive restores itself.

## Strategies

`strategy(candidates, n, archive) -> list[Individual]`, drawing from `archive.rng`. EvoKit ships
`best`, `uniform` and `tournament(size)`; recipe-specific ones live in their recipes.

## Variation functions

Each returns `(child, usage)` or raises `OperatorError`:

| Function | For |
| --- | --- |
| `rewrite_text(model, parent, others)` | string genomes |
| `diff(model, parent, others, task=)` | file genomes, SEARCH/REPLACE edits |
| `rewrite_files(model, parent, others, task=)` | file genomes, whole-file rewrites |
| `self_modify(parent, experiment)` | DGM |
| `meta_agent_modify(parent, experiment)` | Hyperagents |

Models: `AnthropicModel`, `OpenAIModel` (any compatible endpoint) and `FunctionModel`. Pass
`price=(in, out)` in USD per million tokens to track cost.

## Agents

`ClaudeCode`, `CommandAgent` (any CLI) and `ReplAgent`, or your own `Agent`.

```python
async with experiment.start(agent, start=parent, submit=True) as directory:
    async for event in agent.prompt("Improve it."):     # ToolCall events, then a Turn
        ...
```

The agent works in a checkout of `start`. Its tools are `stats`, `log`, `show`, `diff`, `grep`,
`notes`, `remember`, `score`, and `submit` when `submit=True`. Turns are recorded, and the agent
is stopped at a limit. `agent.fresh()` gives a read-only copy, e.g. for a reviewer.

## Storage

`write`, `read`, `exists`, `list`, `append`, `local`, `lock`. `DirectoryStorage` and
`MemoryStorage` ship. An experiment directory:

```text
history/records.jsonl   every record
archive/changes.jsonl   the archive's log
blobs/                  genome contents, content-addressed
artifacts/<id>/         evaluation logs
failed/<id>/            logs of failed variations
sessions/               agent transcripts
```
