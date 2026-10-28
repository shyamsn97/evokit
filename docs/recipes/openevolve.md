# OpenEvolve

!!! note "An implementation, not the official code"
    This is EvoKit's version of OpenEvolve's loop, not a port of `codelion/openevolve`. See
    [how it differs from the official code](#how-it-differs-from-the-official-code).

[OpenEvolve](https://github.com/codelion/openevolve) is an open AlphaEvolve. It has three parts:

- **Islands of MAP-Elites grids.** The default features are code length and diversity.
- **Parents, inspirations and LLMs.** A parent is drawn by exploration, exploitation or elite
  sampling. Inspirations come from the same island. An ensemble of LLMs writes diffs.
- **Ring migration** every few generations.

```python
# evokit/recipes/openevolve.py
async def openevolve(task, models, weights=None, *, dir=None, archive=None, inspirations=4,
                     migrate_every=20, migrate_top=2, workers=4, max_evaluations=1000,
                     max_cost=None, seed=None):
    task = replace(task, evaluate=with_lines(task.evaluate))          # for a Files genome
    archive = archive or IslandArchive(seed=seed or 0)                # 4 islands, ("lines", "diversity")
    experiment = task.create_experiment(dir, archive=archive, max_evaluations=max_evaluations, max_cost=max_cost)
    strategy = islands()

    async def worker():
        while not experiment.limits.reached():
            parent, *others = archive.sample(1 + inspirations, strategy) or [None]
            if parent is None:
                return
            name = archive.rng.choices(list(models), odds)[0]        # the weighted ensemble
            if child := await experiment.create(diff(models[name], parent, others, task=task)):
                await experiment.submit(child.with_info(operator=name))
            if experiment.since_note("migrate") >= migrate_every:
                experiment.note("migrate")
                archive.migrate(top=migrate_top)

    await asyncio.gather(*(worker() for _ in range(workers)))
    return experiment.best()
```

```python
models = {
    "fast": AnthropicModel("claude-haiku-4-5", price=(1.0, 5.0)),      # USD per million tokens in, out
    "strong": AnthropicModel("claude-sonnet-4-5", price=(3.0, 15.0)),
}
best = await recipes.openevolve(task, models, {"fast": 0.8, "strong": 0.2}, dir="experiments/oe-0")
```

OpenEvolve's cascade evaluation (a quick test, then the full one for programs that pass) is
part of the task's `evaluate`, like the Darwin Gödel Machine's [`staged_eval`](dgm.md).
Migration is scheduled by `migrate` notes in the experiment, so a continued run keeps the same
schedule.

## How it differs from the official code

Compared with [codelion/openevolve](https://github.com/codelion/openevolve) at commit `515d1bc`
and its `configs/default_config.yaml`:

- **The prompt is much thinner.** OpenEvolve sends a system and a user message built from
  templates: the parent's metrics and feature coordinates, areas to improve, an evolution history
  of the last 3 attempts and their outcomes, 3 top programs, 2 diverse programs, and optionally
  the evaluator's artifacts. This recipe sends one prompt: the task instructions, the parent's
  scores and feedback, and the full code of the parent and the inspirations.
- **No elite archive.** OpenEvolve keeps an elite archive of 100 programs beside the MAP-Elites
  grids. Its exploitation draws (70%) take a parent from that archive, preferring the current
  island, and the remaining 10% take a uniformly random program from the whole database. Here
  exploitation draws from the current island's top 3, and the last 10% from the top 3 overall.
- **Migration is different in every detail.** OpenEvolve migrates every 50 island generations,
  sends 10% of each island to both ring neighbours, and places a child on the island the
  iteration was scheduled for. Here migration happens every 20 evaluations, sends the top 2 of
  each island to the next island only, and a child joins its parent's island.
- **Fewer islands, more inspirations.** OpenEvolve defaults to 5 islands and 2 inspirations
  (island best, top programs, then nearby grid cells). Here: 4 islands and 4 inspirations
  (island top members, then one program per other cell).
- **Different features.** OpenEvolve's complexity is the code's length in characters and its
  diversity is a fast heuristic against a cached reference set of 20 programs, both scaled to the
  range seen so far. Here complexity is the count of non-blank, non-comment lines, and diversity is
  the `difflib` similarity to up to 8 members of the same island, on a fixed 0 to 1 range.
- **No novelty check.** OpenEvolve can reject near-duplicates with embeddings (threshold 0.99) and
  an LLM judge before they enter the grid. This recipe has none.
- **No population cap.** OpenEvolve keeps at most 1,000 programs and evicts the worst. Here only
  grid displacement removes members.
- **Cascade evaluation is the caller's.** OpenEvolve calls `evaluate_stage1/2/3` with thresholds
  0.5, 0.75 and 0.9, retries failed evaluations up to 3 times, and times out at 300 s. Here the
  task's `evaluate` does any staging; the circle-packing example promotes on a timeout, not a
  score.
- **Diffs only.** OpenEvolve can switch to full rewrites (`diff_based_evolution: false`) and
  rejects programs over 10,000 characters. This recipe always asks for diffs and has no size
  limit.
- **EVOLVE-BLOCK is always enforced.** OpenEvolve ignores the markers unless
  `enforce_evolve_blocks` is set, and then reverts edits outside them. Here a child that edits
  outside the markers is refused.
- **One process.** OpenEvolve runs each iteration (LLM call, diff, evaluation) in a process pool
  of 4. Here 4 asyncio workers share one process, and evaluations run as subprocesses (or
  threads, for a Python `evaluate`).
- **No LLM feedback, artifacts, checkpoints or prompt logs.** OpenEvolve can add an LLM's score to
  the evaluation (weight 0.1), feed evaluator artifacts back into prompts, checkpoint every 10
  iterations, and log every prompt. The experiment's directory is the only persistence here.
- **It stops on evaluations or cost.** OpenEvolve stops after `max_iterations` (100 in the
  default config) or at a target score. Here: `max_evaluations` (default 1,000) or `max_cost`.

The same: islands of MAP-Elites grids with one elite per cell, the 20% exploration and 70%
exploitation split, a weighted ensemble of LLMs (0.8 / 0.2 in both examples), SEARCH/REPLACE
diffs, a parent with inspirations in the prompt, and ring migration of strong members.
