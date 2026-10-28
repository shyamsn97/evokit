# ShinkaEvolve

!!! note "An implementation, not the official code"
    This is EvoKit's version of ShinkaEvolve's loop, not a port of `SakanaAI/ShinkaEvolve`. See
    [how it differs from the official code](#how-it-differs-from-the-official-code).

[ShinkaEvolve](https://github.com/SakanaAI/ShinkaEvolve) (Sakana AI, 2025) adds four ideas to
an AlphaEvolve-style loop:

- islands, with migration every 10 evaluations;
- parents weighted by fitness and by how few children they have, plus inspirations;
- an ensemble of LLMs chosen by a UCB1 bandit, and diff, full-rewrite or crossover edits;
- novelty rejection before evaluation, and a meta-scratchpad of what has worked so far.

```python
# evokit/recipes/shinka.py
KINDS = {"diff": 0.6, "full": 0.3, "cross": 0.1}

async def shinka(task, models, *, embed, judge, summarizer, dir=None, archive=None, workers=4,
                 threshold=0.95, every=10, max_evaluations=150, max_cost=10.0, seed=None):
    archive = archive or IslandArchive(islands=2, features=(), cell_size=50)
    experiment = task.create_experiment(dir, archive=archive, max_evaluations=max_evaluations, max_cost=max_cost)
    strategy, bandit = parents(objective=task.objective), ModelBandit(models, objective=task.objective)

    async def worker():
        while not experiment.limits.reached():
            parent, *others = archive.sample(5, strategy) or [None]
            if parent is None:
                return
            kind = rng.choices(list(KINDS), list(KINDS.values()))[0]
            name = bandit.choose(experiment)                       # UCB1, from every child so far
            notes = experiment.last_note("meta")
            instructions = join_notes(task.instructions, notes and f"What has worked so far:\n{notes}")
            if child := await experiment.create(vary(kind, models[name], parent, others, task, instructions)):
                if await is_novel(child, archive, embed, judge, experiment, threshold):
                    await experiment.submit(child.with_info(operator=f"{kind}/{name}"))
                else:
                    experiment.discard(child, "not novel")
            if experiment.since_note("migrate") >= every:          # migrate every `every` evaluations
                experiment.note("migrate")
                archive.migrate(top=1)
            if experiment.since_note("meta-started") >= every:     # and rewrite the scratchpad
                experiment.note("meta-started")
                reply = await summarizer.generate(META_PROMPT + describe(archive.top(20)))
                experiment.note("meta", reply.text, reply.usage)

    await asyncio.gather(*(worker() for _ in range(workers)))
    return experiment.best()
```

Everything a continued run needs is read from the experiment: the bandit's pulls and rewards
from the evaluated children, the migration and summary schedules from `migrate` and
`meta-started` notes, and the scratchpad from the last `meta` note.

The pieces that belong to this recipe live next to it:

- **`parents`** is a strategy. A parent's weight is sigmoid(normalized score) / (1 + children);
  the other picks, the inspirations, are the island's best members and then the best overall.
- **`ModelBandit`** is UCB1 over models. Its reward is how much a child improved on its parent.
- **`is_novel(child, archive, embed, judge, experiment, threshold)`** compares the child's
  embedding with the archive. It asks the LLM judge when the closest member is within
  `threshold`, and notes the cost of the check in the experiment. A child that fails is
  discarded with `experiment.discard`, before evaluation.

## How it differs from the official code

Compared with [SakanaAI/ShinkaEvolve](https://github.com/SakanaAI/ShinkaEvolve) at commit
`8adc053`, both its library defaults and the settings of its own circle-packing example
(`examples/circle_packing/shinka_small.yaml`), which differ in places:

- **No separate archive of correct programs.** ShinkaEvolve keeps every program in SQLite, plus
  an archive of up to 40 correct programs. Its circle-packing example ranks that archive by
  crowding on score and lines of code. Parents and inspirations come only from the archive. Here
  each island holds up to 50 members by score, there is no correctness flag, and parents come from
  all of an island's members.
- **Parent weights are normalized differently.** Both weight a parent by
  sigmoid(λ × normalized score) / (1 + children), with λ = 10. ShinkaEvolve normalizes with the
  median and the median absolute deviation; here it is min-max over the island. ShinkaEvolve also
  has power-law, beam-search, best-of-n and sequential parent selection; this recipe has only the
  weighted one.
- **Fewer, differently chosen inspirations.** ShinkaEvolve shows 1 archive inspiration and
  1 top-k inspiration, kept to the parent's island and sorted from worst to best. Here: the
  island's 2 best members and the 2 best overall, unsorted. ShinkaEvolve also drops crossover when
  there are no inspirations; here the 0.6 / 0.3 / 0.1 mix never changes.
- **No bootstrap phase.** Until every island has a correct program, ShinkaEvolve mutates only the
  initial program, with no inspirations. Here selection starts as soon as the archive has members.
- **Migration is counted and moved differently.** ShinkaEvolve migrates every 10 generations.
  The library default rate is 0, so it is off; the circle-packing example uses 10%, moving
  (not copying) a random subset of correct programs while keeping each island's best. Here the
  top member of each island is copied to the next island every 10 evaluations.
- **A simpler bandit.** ShinkaEvolve's UCB1 is asymmetric: rewards are shifted by the parent's
  and the initial program's scores and clipped at zero, incorrect children get the worst score,
  it explores with ε = 0.2, and the circle-packing example weighs cost in (coefficient 0.5). Here
  it is plain UCB1 rewarded by the child's improvement over its parent, with no ε and no cost term.
- **Novelty is checked once, against everyone.** ShinkaEvolve embeds code with
  `text-embedding-3-small`, compares only with the parent's island, and calls the judge above
  0.99 similarity. Its library default retries a rejected proposal up to 3 times (the example
  uses 1). It skips the check in generation 0 and until every island is initialized. Here the
  caller's `embed` is compared with the whole archive at 0.95, and a rejected child is discarded
  without a retry.
- **The meta notes are on here and off in the example.** ShinkaEvolve's meta summarizer is a
  three-step pipeline (per-program summaries, a scratchpad, then recommendations), and its output
  goes into the system prompt. The circle-packing example turns it off (`meta_rec_interval:
  null`). This recipe always runs one summarizer call over the 20 best members every
  10 evaluations, and adds the reply to the instructions as "What has worked so far".
- **Generic prompts.** ShinkaEvolve has its own prompts for each patch type, a "fix" mode when no
  program is correct yet, optional text feedback from the evaluator, an evaluation-history block,
  and retries for patches that fail to apply (3). Here the generic file operators are used, and
  crossover is a full rewrite with a "combine these" instruction.
- **More models and sampling settings there.** The circle-packing example samples four models
  (gpt-5.4, gemini-3.1-pro-preview, gemini-3-flash-preview, gpt-5.4-mini) at temperatures 0, 0.5
  and 1.0 and three reasoning efforts. Here the caller's models are called with their own
  settings.
- **Different concurrency and stopping.** ShinkaEvolve runs separate pools of proposal and
  evaluation jobs (5 and 5 in the small example) and stops at 200 generations or $20 in that
  example. Here 4 workers each run select, mutate, check novelty and submit, until
  150 evaluations or $10.
- **Not here at all:** prompt evolution, spawning new islands on stagnation, multi-criteria and
  crowding archives, stored code diffs and their summaries, and ShinkaEvolve's web UI, W&B
  logging, Hydra and Slurm launchers.

The same: two islands, the 0.6 / 0.3 / 0.1 mix of diff, full and crossover edits, children staying
on their parent's island with the seed copied to every island, the child-count discount, a UCB
bandit over models, embedding plus LLM-judge novelty rejection before evaluation, and EVOLVE-BLOCK
markers.
