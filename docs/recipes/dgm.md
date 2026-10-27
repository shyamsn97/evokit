# Darwin Gödel Machine

In the [Darwin Gödel Machine](https://arxiv.org/abs/2505.22954), the genome is a coding
agent's repository:

- Every agent that still works is kept, so the archive is open-ended.
- Parents are sampled by score, discounted by how many children they already have.
- The parent agent reads the logs of its own evaluation, proposes one improvement to itself,
  and implements it on a copy of its code. That is `SelfModify`.
- Evaluation is staged: a few tasks first, and the full benchmark only for agents that do well
  on those.

```python
# evokit/recipes/dgm.py
async def dgm(eval_fn, *, storage, archive, operator=None, fixed_agent=None, instructions=None,
              workers=4, max_children=80, sharpness=10.0, midpoint=0.5, open_ended=True):
    strategy = ProportionalSelection(sharpness, midpoint) if open_ended else Greedy()
    operator = operator or SelfModify(storage, fixed_agent=fixed_agent, instructions=instructions)

    async def worker():
        while history.counts(storage)["created"] <= max_children:
            asked = await evokit.ask(storage, archive, strategy, operator)
            if asked.exhausted:
                return
            evaluations = await asyncio.gather(*map(score, asked.individuals))
            await evokit.tell(storage, archive, asked.individuals, evaluations)

    await asyncio.gather(*(worker() for _ in range(workers)))
    return archive.best()


def staged_eval(quick, full, promote):
    """Run `quick`; only if promote(its evaluation), run `full` and return that."""
    quick, full = as_eval_fn(quick), as_eval_fn(full)
    async def staged(individual):
        evaluation = await quick(individual)
        return await full(individual) if promote(evaluation) else evaluation
    return staged
```

The caller brings the seed agent and the evaluation:

```python
storage = DirectoryStorage("runs/dgm")
eval_fn = staged_eval(run_10_tasks, run_50_tasks, promote=lambda e: e.objectives.get("score", 0) >= 0.4)
archive = UnboundedArchive("score", storage=storage)
if not archive.members():
    seed = Individual.new(Files.from_directory("seed_agent"))
    await evokit.tell(storage, archive, [seed], [await evokit.as_eval_fn(eval_fn)(seed)])
await recipes.dgm(eval_fn, storage=storage, archive=archive)
```

The paper's ablations:

- **No self-improvement:** pass `fixed_agent=seed_agent`, so the seed agent makes every
  modification.
- **Not open-ended:** pass `open_ended=False` and a `LatestArchive("score", storage=storage)`,
  so the loop builds greedily on the newest working agent.

## How it differs from the official code

Compared with [jennyzzt/dgm](https://github.com/jennyzzt/dgm) at commit `a565fd2`:

- **A general loop, not a SWE-bench pipeline.** The official code evolves a coding agent on
  SWE-bench Verified (or Polyglot) through Docker harnesses. This recipe is benchmark-agnostic: the
  caller's `eval_fn` scores the agent. The bundled example scores agents on circle packing.
- **Diagnosis is a separate model there, the parent agent here.** The official loop first picks
  one task for the parent to improve on. It is a random unresolved SWE-bench instance, or, with set
  probabilities, a special entry about empty patches, stochastic results, or context length. Then
  `o1-2024-12-17` reads that task's logs, the agent's patch, the test patch and the reference
  answer, and writes a GitHub-style problem statement. Here the parent agent itself (or
  `fixed_agent`) reads all of its own evaluation logs, writes `proposal.md` with one general
  improvement, and is told not to solve benchmark tasks. There is no separate model and no answer
  patch.
- **Patches in Docker versus copies on the host.** Officially, a child is a `model_patch.diff`. The
  loop applies the chain of ancestor patches inside a `dgm` container, commits a baseline, and runs
  `coding_agent.py --self_improve` with a 30-minute timeout. Here the parent's files are copied to a
  temporary directory on the host, and the edited directory is the child.
- **Generations versus continuous workers.** The official loop runs 80 generations; each picks 2
  parents, improves them with 2 threads, then updates the archive. Here 4 workers run
  select, improve, evaluate and tell continuously, until 80 children have been created.
- **Fewer parent rules.** Both default to sigmoid(10 × (score − 0.5)) × 1 / (1 + children). The
  official code also has `score_prop` (no child discount), `best`, `random`, and the `no_darwin`
  baseline (always the latest). Here there are `ProportionalSelection` and `Greedy`, and
  `ProportionalSelection` also skips parents that already score 1.0, which the official code does
  not do.
- **Archive admission.** Officially, every child that "compiles" is kept (`keep_all`): it has
  harness results, at least one non-empty patch, and enough submitted instances. The option
  `keep_better` keeps only children within 0.1 of the initial agent's score. Here every child with
  a score is kept, and there is no `keep_better`.
- **Staged evaluation is fixed there, the caller's here.** The official first stage is 10 tasks;
  an agent that resolves at least 4 goes on to 50 more. A larger third set is loaded, and a
  full-evaluation threshold is passed, but neither is used at this commit. Here `staged_eval` takes
  the caller's two stages and promotion rule. The circle-packing example does its own staging in
  `benchmark.py`: n=26 first, then n=20, 26 and 32, then 2 repeats only if the agent beats the
  seed.
- **No post-improvement diagnosis.** The official loop can have o1 compare the agent's logs
  before and after on the same task (`post_improve_diagnose`, off by default). Not here.
- **Different seed agent.** The official seed is `coding_agent.py` on Claude 3.5 Sonnet (Bedrock),
  with a regression-test flow for SWE-bench, and it outputs a diff. The example's seed here
  (`seed_agent/agent.py`) calls the Anthropic API (`claude-sonnet-4-5` by default) with bash, view
  and edit tools and prints a JSON summary line.
- **Budgets.** The official loop has no cost or failure limit, and each self-improvement attempt
  times out after 1.5 hours. Here `max_cost` and `max_failures` (10 in a row) stop the run.
- **Ablations.** Official flags: `no_selfimprove`, which improves from the initial agent's code
  instead of the parent's, plus `no_darwin`, `shallow_eval`, `keep_better`, `no_full_eval` and
  `polyglot`. Here: `fixed_agent` (the seed makes every modification) and `open_ended=False` with a
  `LatestArchive`. The example adds a greedy variant that the official code does not have.

The same: an open-ended archive of working agents; the default parent rule (λ = 10, midpoint 0.5,
child discount 1 / (1 + children)); the agent improving its own code from its own evaluation logs;
a cheap stage before an expensive one; and ablations for a frozen improver and a latest-only
lineage.

[DGM-Hyperagents](hyperagents.md) is this loop with the improvement procedure itself inside the
genome.
