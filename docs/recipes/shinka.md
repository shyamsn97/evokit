# ShinkaEvolve

!!! note "An implementation, not the official code"
    See [how it differs](#how-it-differs-from-the-official-code).

[ShinkaEvolve](https://github.com/SakanaAI/ShinkaEvolve) (Sakana AI, 2025) adds four things to an
AlphaEvolve-style loop: islands; parents weighted by score and few children; a UCB1 bandit over
LLMs with diff, full and crossover edits; and novelty rejection plus a meta-scratchpad.

```python
async def worker():
    while not experiment.limits.reached():
        parent, *others = archive.sample(5, parents())
        kind = rng.choices(["diff", "full", "cross"], [0.6, 0.3, 0.1])[0]
        name = bandit.choose(experiment)
        if child := await experiment.create(vary(kind, models[name], parent, others, task, instructions)):
            if await is_novel(child, archive, embed, judge, experiment, threshold):
                await experiment.submit(child)
            else:
                experiment.discard(child, "not novel")
        # every `every` evaluations: migrate, and rewrite the "meta" scratchpad
```

All state (bandit rewards, schedules, the scratchpad) is read back from the experiment.

## How it differs from the official code

Compared with commit `8adc053`, both the library defaults and its circle-packing example:

- **No separate archive of correct programs:** each island keeps up to 50 members by score.
- **Min-max normalization** of parent weights instead of median/MAD, and only the weighted
  parent rule.
- **Inspirations:** the island's best two plus the best two overall, unsorted.
- **No bootstrap phase.**
- **Migration** copies the top member every 10 evaluations; the original moves a random 10%.
- **Plain UCB1**, with no ε and no cost term.
- **Novelty** is checked against the whole archive at 0.95, with no retry.
- **The meta notes are always on**, as one summarizer call.
- **Generic prompts**, with no fix mode and no patch retries.
- **Caller-chosen models**, and 4 combined workers.
- **Left out:** prompt evolution, island spawning, crowding archives, the UI, W&B and Hydra.

The same: two islands, the 0.6/0.3/0.1 edit mix, the child-count discount, a UCB bandit,
embedding plus judge novelty, and EVOLVE-BLOCK markers.
