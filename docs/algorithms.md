# Algorithms on the ask/tell Experiment

Four published algorithms, each built from EvoKit parts and run on the same small task:
packing 26 circles in the unit square to maximize the sum of their radii
([examples/circle_packing](../examples/circle_packing)). The seed program scores 1.6026; the
best known packing is about 2.635. A fifth, OMNI-EPIC, has no example but is supported; its
mapping is at the end.

**Status: implemented.** Each algorithm is a function in [evokit/recipes](../evokit/recipes)
(`autoresearch`, `avo`, `openevolve`, `dgm`) that assembles an `Experiment` from the domain's
parts: the project or seed, the evaluate function, the agent or models, and the task text.
Its components are in the usual folders: `SimplerBestArchive`, `IslandArchive`,
`UnboundedArchive` and `LatestArchive` in `archives/`; `IslandSampler` and
`ProportionalSelection` in `strategies/`; `SelfModify` in `files/`. The circle-packing
wrappers that call the recipes are in
[examples/circle_packing/algorithms](../examples/circle_packing/algorithms), and
[tests/test_examples.py](../tests/test_examples.py) runs every example offline, with scripted
agents and models in place of LLMs.

## The shape every algorithm uses

```python
experiment = Experiment(archive, strategy, operator=None, seeds=(), store=None)
experiment.run(evaluate, workers=1, max_evaluations=None, max_cost=None, max_time=None)
```

An algorithm is a choice of five things: what survives (`archive`), what to build on
(`strategy`), how new individuals are made (`operator`), how they are scored (`evaluate`),
and how many run at once (`workers`). The loop, the record, the limits, and the CLI are the
same for all of them.

| | autoresearch | AVO | AlphaEvolve / OpenEvolve | Darwin Gödel Machine |
| --- | --- | --- | --- | --- |
| Genome | `main.py` | `main.py` | `main.py`, inside an EVOLVE-BLOCK | a coding agent's repository |
| Archive | the best; ties kept if shorter | the best; ties kept | islands of MAP-Elites grids | every agent that still works |
| Strategy | the best | the best | islands in turn; parent plus inspirations | by score, fewer children preferred |
| Operator | the agent driving the CLI, or a coding agent | one persistent coding-agent session | a fast and a strong model writing diffs | the parent agent edits its own code |
| Evaluate | `evaluate.py`, plus code length | `evaluate.py` | a two-stage cascade | the agent solves packing tasks, in stages |
| Workers | 1 | 1 | 8 | 4 |

Run them from `examples/circle_packing`; each writes `best_main.py` into its run directory:

```bash
python -m algorithms.autoresearch --dir runs/autoresearch-0
python -m algorithms.avo          --dir runs/avo-0
python -m algorithms.openevolve   --dir runs/openevolve-full-0 --variant full
python -m algorithms.dgm.main     --dir runs/dgm-full-0        --variant full
```

All of them share [common.py](../examples/circle_packing/algorithms/common.py): the task text,
the project (`main.py` evolves; `evaluate.py` is the environment, copied into the run
directory once), the evaluation (`ScriptEvaluator(project, "python evaluate.py")`, which fails
invalid packings), and the run flags (`--max-cost`, `--max-evaluations`, `--max-time`,
`--workers`).

## 1. autoresearch

**What it is.** [autoresearch](https://github.com/karpathy/autoresearch) gives a coding agent
one file to edit and a fixed evaluation. The agent edits, evaluates, keeps the change if the
score improved and reverts otherwise, logs every attempt, and never stops. Its instructions
add a simplicity rule: a tiny gain that adds ugly code is not worth it, and an equal result
from deleting code is a win.

**How it maps** ([autoresearch.py](../examples/circle_packing/algorithms/autoresearch.py)).

| autoresearch | EvoKit |
| --- | --- |
| the file the agent edits | the genome: `main.py` |
| the fixed evaluation | `evaluate.py`, plus a `lines` objective (non-blank, non-comment lines of `main.py`) |
| keep if better, else revert | `SimplerBestArchive`: a strictly higher score, or a tie within 1e-6 with fewer lines |
| always continue from the branch head | `Greedy()` |
| a crash, or a run over the time limit | `Fitness("failed")` or `Fitness("timeout")`, with the log tail as feedback |
| `results.tsv` | `events.jsonl`: every attempt, kept or not |
| `program.md` | the exported skill, plus the simplicity rule in the task description |

There are two variants. In `external` the run is built from the task folder's `config.yaml`,
which has no operator: the agent you are talking to follows the skill and loops through the CLI.

```bash
evokit status --task . --dir runs/autoresearch-external-0
evokit skill export claude --dir runs/autoresearch-external-0
# then, in Claude Code: "You are the evokit operator for runs/autoresearch-external-0.
#   Follow the evokit skill. Never stop."
```

Each step is `select-individuals`, `checkout-individual ID DIR`, an edit, then
`evaluate-directory DIR --parent ID --name "..." --description "..."`, which evaluates and tells; the result's
`info.admitted` says whether the change was kept. In `unattended`, a
`CodingAgentOperator(ClaudeCode(), project)` makes each change in a fresh session.

```python
experiment = Experiment(SimplerBestArchive(), Greedy(), CodingAgentOperator(ClaudeCode(), project),
                        seeds=[project.seed_genome()], store=DirectoryStore(directory))
experiment.run(evaluate_with_lines, workers=1, max_cost=5)
```

## 2. AVO

**What it is.** AVO (agentic variation operators) replaces one-shot LLM mutation with a coding
agent in one continuous session. The agent reads earlier versions, tests drafts as often as it
likes, and commits when a version is ready; the lineage moves forward on any commit that is
correct and not worse; and a supervisor reviews the run and suggests new directions when it
stalls. Details and the paper's quotes: [avo.md](avo.md).

**How it maps** ([avo.py](../examples/circle_packing/algorithms/avo.py)).

| AVO | EvoKit |
| --- | --- |
| commit if correct and not worse | `BestArchive(ties="accept")`; invalid packings already fail |
| the agent commits | `$EVOKIT_COMMIT "title" "description"`: recorded, evaluated and told at once; the reply is the verdict |
| one agent session across the run | one Claude Code session, resumed with "continue" while no limit is reached |
| the agent tests drafts as it works | `$EVOKIT_DRAFT_EVAL`, no limit by default |
| the lineage P_t | `evokit log --all`, `show`, `diff`, `grep` |
| the supervisor | after 5 drafts and commits without a strictly better score, a fresh session reads the run and names 3 to 5 directions |

```python
archive = BestArchive("score", ties="accept", storage=storage)   # seeded
await recipes.avo(project, evaluate, ClaudeCode(), storage=storage, archive=archive, max_cost=5)
```

Limits are checked on every draft and commit (the reply then says `"stop"`), and each start or
resume of the session gets the remaining budget as Claude Code's `--max-budget-usd`.

## 3. AlphaEvolve / OpenEvolve

**What it is.** The loop from [AlphaEvolve](https://arxiv.org/abs/2506.13131) and its open
reimplementation [OpenEvolve](https://github.com/codelion/openevolve), whose flagship example is
this same 26-circle task. It differs from the two above in four ways: a population instead of
one lineage, many cheap LLM edits instead of one agent session, other good programs shown in
every prompt, and many evaluations in flight at once.

**How it maps** ([openevolve.py](../examples/circle_packing/algorithms/openevolve.py)).

| AlphaEvolve / OpenEvolve | EvoKit |
| --- | --- |
| `# EVOLVE-BLOCK-START` / `-END` | the seed is marked from `construct_packing()` on; `DiffOperator` rejects edits outside the markers |
| the program database: islands of MAP-Elites grids | `IslandArchive(islands=4, features=("lines", "diversity"), bins=10)` |
| a child joins its parent's island | the archive looks up the island of `parents[0]`; seeds are copied to every island |
| feature dimensions | `lines` from the evaluation; `diversity`, the mean edit distance to a sample of the island, computed in `add` |
| migration between islands | every 20 admissions, each island's top 2 are copied to the next island, in a ring, inside `add` |
| parent and inspiration sampling | `IslandSampler(exploration=0.2, exploitation=0.7, elite=0.1, inspirations=4)`, returning `(parent, *inspirations)` |
| the LLM ensemble | `Choice({"fast": DiffOperator(fast), "strong": DiffOperator(strong)}, weights={"fast": 0.8, "strong": 0.2})` |
| the prompt | the parent's code, score and feedback; the inspirations' code and scores; the task |
| the evaluation cascade | stage 1 runs `evaluate.py` with a 30 s limit; only a timeout goes to stage 2, with 10 minutes |
| the asynchronous controller | `run(..., workers=8)`: new programs are made while others are evaluated |

```python
experiment = Experiment(
    IslandArchive(islands=4, features=("lines", "diversity"), bins=10, migrate_every=20, migrate_top=2),
    IslandSampler(exploration=0.2, exploitation=0.7, elite=0.1, inspirations=4),
    Choice({"fast": DiffOperator(fast, project), "strong": DiffOperator(strong, project)},
           weights={"fast": 0.8, "strong": 0.2}),
    seeds=[with_evolve_block(project.seed_genome())],
)
experiment.run(cascade_evaluate(project), workers=8, max_evaluations=300, max_cost=5)
```

No `Experiment` subclass is needed: the island model is an archive plus a strategy. The
archive's `niches()` are keyed `(island, cell)`, which is all the sampler reads. At the end,
the script prints the best score by evaluation and each island's grid, code length by
diversity.

**Variants**, one swap each (`--variant`): `one-island` (`islands=1`), `no-inspirations`
(`inspirations=0`), `single-model` (only the fast model), and `no-grid` (`features=()`,
`cell_size=50`: each island is one population of 50).

## 4. Darwin Gödel Machine

**What it is.** The [Darwin Gödel Machine](https://arxiv.org/abs/2505.22954) evolves coding
agents that improve themselves. It keeps an archive of every agent that can still edit code
and picks parents by a sigmoid of their score, discounted by how many children they already
have. Each parent reads its own benchmark logs, proposes a change to itself, and implements it
on its own code. Children are evaluated in stages, with the expensive stages only for the
promising ones.

Here the agents are scored on circle packing, so evolution changes the agent, not the packing
program: a better agent is one that writes better packings.

**The genome** is a minimal coding agent in
[seed_agent/](../examples/circle_packing/algorithms/dgm/seed_agent), about 150 lines with no
packing knowledge:

- `agent.py`: ask the model, run the tool calls, repeat until it stops calling tools or
  reaches `MAX_STEPS`. It calls the Anthropic Messages API directly, with no SDK.
- `tools.py`: `bash(command)`, `view_file(path, start, end)`, `edit_file(path, old, new)`.
- `prompts.py`: the system prompt.

It runs as `python agent.py --dir WORKDIR --task "..."` and prints
`{"summary", "cost_usd", "tokens"}` as its last line, which is what `CommandAgent` reads.

**How it maps** ([dgm/](../examples/circle_packing/algorithms/dgm)).

| DGM | EvoKit |
| --- | --- |
| the archive of all agents that still edit code | `UnboundedArchive`: admits every `ok` agent, never drops one |
| parent selection | `ProportionalSelection(sharpness=10, midpoint=0.5)`: weight sigmoid(10 (score − 0.5)) / (1 + children), over agents scoring below 1, sampled with replacement |
| the parent modifies itself | `SelfModify(store)` runs the **parent's own** `agent.py` twice: first to diagnose (it reads `store.artifacts(parent.id)/logs` and writes `proposal.md`), then to implement the proposal on a copy of its own code |
| staged evaluation | `PackingBenchmark(store)`, below |
| 4 in parallel | `run(benchmark, workers=4, max_evaluations=40, max_cost=20)` |

`PackingBenchmark` ([benchmark.py](../examples/circle_packing/algorithms/dgm/benchmark.py)) runs
the candidate agent on packing tasks in scratch copies of
[dgm/task](../examples/circle_packing/algorithms/dgm/task), where `N` comes from `$PACKING_N`:

1. **Stage 1:** N=26, one run. If the agent crashes, changes nothing, or leaves an invalid
   packing, the result is `failed` ("lost editing ability"), and the archive never admits it.
2. **Stage 2:** N = 20 and 32 as well. The score is the mean sum of radii, each divided by
   the best known value in `references.json`.
3. **Stage 3:** only for agents whose stage 2 score beats the seed agent's. Two more rounds
   of all three sizes, averaged, to reduce noise.

Every run's task, agent summary, stderr, `evaluate.py` output and final `main.py` go to
`workdir/logs/`, which is what the next diagnosis reads. The best N=26 packing goes to
`workdir/best_main.py`, and `usage` is the sum of the agents' costs. A fresh `evaluate.py` is
copied in after the agent finishes, so an agent cannot change how it is scored.

**The ablations are one swap each** (`--variant`):

| Paper ablation | Swap |
| --- | --- |
| without self-improvement (a fixed meta agent) | `SelfModify(store, fixed_agent=seed)`: the seed agent does every modification |
| without open-ended exploration | `LatestArchive` (only the newest working agent) and `Greedy()` |
| greedy | `Greedy()` in place of `ProportionalSelection` |

At the end, the script writes the best agent to `best_agent/` and the best packing any agent
produced to `best_main.py`.

## Comparing variants

[compare.py](../examples/circle_packing/compare.py) runs arms and reports them at equal cost:

```bash
python compare.py run autoresearch avo openevolve:full openevolve:one-island --repeats 3
python compare.py run dgm:full dgm:no-self-improve dgm:no-open-ended dgm:greedy --repeats 1
python compare.py report
```

The report reads each run's `events.jsonl` and gives, per algorithm, the best admitted score
at 25%, 50% and 100% of the dollar budget, the final cost, the evaluations (drafts included),
how many attempts were kept, and the wall time. DGM is reported separately because it scores
agents, not packings. An interactive run does not report its model cost; pass it with
`--external-cost RUN=USD`.

## OMNI-EPIC: a supported pattern

[OMNI-EPIC](https://arxiv.org/abs/2405.15568) generates an endless stream of learnable and
interesting reinforcement-learning tasks. It has no example here, because its evaluation is RL
training, but every piece maps onto the same API:

| OMNI-EPIC | EvoKit |
| --- | --- |
| a task: description and environment code | the genome: `{"description": ..., "code": ...}` |
| the task archive, learned and failed, unbounded | an archive with two niches, `learned` and `failed`, that never drops a task |
| a random task and the tasks most similar to it | a strategy that embeds archive members and returns a random one plus its nearest neighbours; `select_individuals` is async, so it can call an embedding model |
| the task and environment generators, with a compile-and-fix loop | an operator: one LLM call for the description, one for the code, up to five fix attempts inside `create_individuals` |
| the model of interestingness | the operator's last step, using the `archive` argument of `create_individuals` to find the archived tasks most similar to the new one; an uninteresting task returns `[]` (or raises `OperatorError` to record the cost) and is never trained |
| modifying a failed task | the same operator, when the selected task is in the `failed` niche |
| RL training and the success detector | `evaluate(individual, workdir)`: train, run the success check, return `{"success": ..., "usage": ...}` |
| continuing from a related task's policy | `evaluate` loads the policy from `store.artifacts(individual.parents[0])` and saves its own to `workdir` |

Two tests in [tests/test_core.py](../tests/test_core.py) cover the parts the circle-packing
examples do not use: an operator that rejects children too similar to archive members, and an
evaluation that warm-starts from its parent's artifact folder.
