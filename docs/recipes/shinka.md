# ShinkaEvolve

[ShinkaEvolve](https://github.com/SakanaAI/ShinkaEvolve) (Sakana AI, 2025) adds four ideas to
an AlphaEvolve-style loop:

- islands, with migration every 10 admissions;
- parents weighted by fitness and by how few children they have, plus inspirations;
- an ensemble of LLMs chosen by a UCB1 bandit, and diff, full-rewrite or crossover edits;
- novelty rejection before evaluation, and a meta-scratchpad of what has worked so far.

```python
# evokit/recipes/shinka.py
KINDS = {"diff": 0.6, "full": 0.3, "cross": 0.1}

async def shinka(project, eval_fn, models, *, instructions, embed, judge, summarizer, storage,
                 archive, workers=4, threshold=0.95, every=10, max_cost=10.0, max_evaluated=150,
                 seed=0):
    strategy = ShinkaParents(seed=seed)
    bandit = ModelBandit(models).fit(storage)              # rewards from every child so far
    notes = history.last_note(storage, "meta") or ""

    async def worker():
        while under_budget():
            kind = rng.choices(list(KINDS), list(KINDS.values()))[0]
            name = rng.choices(*zip(*bandit.probabilities().items()))[0]
            prompt = instructions + (f"\n\nWhat has worked so far:\n{notes}" if notes else "")
            operator = make_operator(kind, models[name], project, prompt)   # diff, full rewrite, or crossover

            asked = await evokit.ask(storage, archive, strategy, operator)
            novel = [child for child in asked.individuals
                     if await is_novel(child, archive, embed, judge, storage, threshold)]
            evaluations = await asyncio.gather(*map(score, novel))
            await evokit.tell(storage, archive, novel, evaluations)
            for child in evaluated:
                bandit.reward(child, history.individual(storage, child.parents[0]))
            # migrate every `every` admissions; rewrite the meta notes every `every` evaluations

    await asyncio.gather(*(worker() for _ in range(workers)))
    return archive.best()
```

The caller passes `IslandArchive(islands=2, features=(), cell_size=50, storage=storage)`.
Everything else a continued run needs is read from the storage: the bandit refits from past
children, the migration schedule from `history.counts(storage)`, and the meta notes from the
last `meta` note.

The pieces that belong to this recipe live next to it:

- **`ShinkaParents`** is a strategy. A parent's weight is sigmoid(normalized score) / (1 + children),
  and the island's top members and some archive elites come back as `Selection.inspirations`.
- **`ModelBandit`** is UCB1 over models. Its reward is how much a child improved on its parent.
  `fit(storage)` rewards it for every child already in the history.
- **`is_novel(child, archive, embed, judge, storage, threshold)`** compares the child's embedding
  with the archive. It asks the LLM judge when the closest member is within `threshold`, and notes
  the cost of the check in the history. A child that fails is discarded with `history.discard`,
  before evaluation.

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
  top member of each island is copied to the next island every 10 admissions.
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
  example. Here 4 workers each run select, mutate, check, evaluate and tell, until 150 evaluations
  or $10.
- **Not here at all:** prompt evolution, spawning new islands on stagnation, multi-criteria and
  crowding archives, stored code diffs and their summaries, and ShinkaEvolve's web UI, W&B
  logging, Hydra and Slurm launchers.

The same: two islands, the 0.6 / 0.3 / 0.1 mix of diff, full and crossover edits, children staying
on their parent's island with the seed copied to every island, the child-count discount, a UCB
bandit over models, embedding plus LLM-judge novelty rejection before evaluation, and EVOLVE-BLOCK
markers.
