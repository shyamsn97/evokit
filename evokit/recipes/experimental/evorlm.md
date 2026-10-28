# How EvoRLM works (`evorlm.py`)

EvoRLM ([shyamsn97/evorlm](https://github.com/shyamsn97/evorlm)) is evolutionary search run by one
recursive language model. A root agent reads a scored population, picks parents and concrete
changes to try, and starts one fresh agent per change. Every agent, at every depth, runs the same
policy: it can evaluate candidates itself or start agents of its own. Each agent hands exactly one
candidate back to the agent that started it, and the host measures that candidate rather than
trusting the agent's score.

The model decides the search: which parents, how many changes, what each one should try. The host
only measures, enforces budgets, records lineage, and keeps the population.

## One generation

1. **Observe.** The root calls `population()` and gets the archive best first: id, score, title
   and parent ids. `candidate(id)` gives a member's exact files.
2. **Plan.** It writes ops, each `{"parent_ids": [...], "instruction": "...", "evals": n,
   "agents": k}`. The number of ops is its choice, from 1 to `max_ops`. `evals` defaults to
   `eval_budget` and `agents` to 1, which is just the child itself.
3. **Fan out.** One `await mutate(*ops)` starts a fresh agent for every op, and they run
   concurrently. Each op's evaluations and agent calls come out of the root's own.
4. **Experiment.** Each child starts with its parents' source in its first message. It changes
   candidates and evaluates them. While it is above `max_depth` and still holds agent calls, it
   can call `mutate` too.
5. **Hand up.** Each child submits one candidate: an unchanged `candidate_id` from its population,
   or a new `candidate` value, plus a title and notes.
6. **Verify and admit.** The host checks the candidate and admits it to the caller's population.
   At the root, that is the archive, recorded in the experiment.
7. **Adapt.** For each op, `mutate` returns the instruction and parents, then either the result
   (id, score, title, the child's notes, `admitted`, `unchanged`) or why it was dropped, plus
   `evals_used` and `agents_used`. The root reads them on its next turn and plans the next
   generation.
8. **Finish.** When its generations are spent, the root submits one candidate, which goes through
   the same check and into the archive. If the root submits while it could still mutate, its
   candidate is admitted, and the host sends it one message per generation, "N generations are
   left: run them before you submit.", which reopens it. GPT-5 stopped after one generation in
   every live run until this was added.

```python
population = archive                      # seeded with the task's seed
for generation in range(generations):     # really: the root's successive turns
    ops = root.plan(population.best_first())
    results = await gather(agent(op) for op in ops)      # same policy, fresh context
    for candidate in results:
        population.admit(host.verify(candidate))
return population.best()
```

The `for` loop lives in the root's own turns. `mutate` is allowed once per turn, so the root sees
a generation's results before it plans the next one. After `generations` calls, `mutate` drops out
of the root's tools.

## One agent

The root and an op's agent differ only in what they are given:

| | root | an op's agent |
| --- | --- | --- |
| population | the archive | a `PopulationArchive(population)` under `populations/<agent id>/`, holding the parents its op named, plus what its own ops hand back |
| evaluations | `generations * max_ops * eval_budget` | its op's `evals`, taken from its caller |
| agent calls | `generations * max_ops` | its op's `agents` minus one (itself) |
| told | generations left, `max_ops`, `eval_budget` | its instruction, and its parents' source |
| `mutate` | while generations, evaluations and agent calls remain | while depth < `max_depth`, and evaluations and agent calls remain |

Each agent has:

- **`population(topk=None)`**: its own population, best first. A child never sees the archive or
  its siblings.
- **`candidate(id)`**: a member's exact value, the same shape `evaluate` takes. For a file task,
  that is a dict from each editable file to its full text.
- **`evaluate(candidate)`**: the task's evaluator, once per turn, at one evaluation each. A genome
  already measured anywhere in the run is free.
- **`mutate(*ops)`**: once per turn, at depth below `max_depth`.
- **`submit(...)`**: one candidate, checked against the return schema.

rlmflow's own `launch_agent` is turned off (`max_depth=0` on the config). Recursion only goes
through `mutate`, which is what keeps budgets conserved and populations local.

## The host's rules

- **Scores are measured, never trusted.** `evaluate` keys every measurement by the genome's
  fingerprint: the `Files` digest, or a JSON dump of the genome. The memo is global, so handing up
  a candidate someone already measured costs nothing.
- **A child's return is verified as follows:**
  - A `candidate_id` must be in the child's own population, or the return is dropped.
  - A new `candidate` is looked up in the memo. If it isn't there, it is measured from the child's
    remaining evaluations. If none are left, or it fails, it is dropped.
- **Lineage:** the new candidate's parents are the `parent_ids` it was submitted with, keeping
  those in the agent's population, or else its op's `parent_ids`. The host sets its title and
  notes from what the agent submitted, and records the agent's graph path as `info["agent"]`.
- **Admission below the root:** the candidate is added to the caller's population archive, which
  keeps its best `population` members (elitist truncation) and logs every change in the
  experiment's storage.
- **Admission at the root:** the candidate is recorded as created and evaluated in the experiment
  and offered to the archive (by default `PopulationArchive(population)`).
  - An id the experiment already holds isn't recorded twice.
  - The draft already counted the evaluation's cost, so the recorded evaluation carries none.
  - A parent that exists only in some local population is replaced by its nearest ancestors the
    experiment holds, so every parent id in the history resolves. The full local lineage stays
    in that population's log.
- **The root hands up too.** Its final submission goes through the same check and admission as
  any child's. If it can't be verified, a `failed` note says why.
- **Budgets are conserved, for both evaluations and agent calls.**
  - An op's evaluations and agent calls come out of its caller's. A child spends one agent call on
    itself, so it can only `mutate` if its op gave it more than one.
  - After the child finishes, whatever it didn't spend goes back to its caller.
  - Every op needs at least one of each. Below the root, an op's `evals` defaults to an even
    split of the caller's evaluations.
  - `mutate` also refuses once the run's token budget (`max_tokens`) is spent, since rlmflow's
    `spawn` doesn't check it, and once the experiment's limits (such as `max_cost`) are reached.
- **Generations are feedback barriers:** one `mutate` per turn, and the root stops mutating after
  `generations`.
- **Evaluations are sequential:** one `evaluate` per turn, so an agent reads each result before
  spending another evaluation.

## How it sits on rlmflow

- **`Evolution(Flow)`** is the run. It holds the experiment, each agent's scope (population,
  evaluations left, depth, instruction, op parents), the score memo, and the four tools. `mutate`
  creates children with `self.spawn(...)` and waits for them with rlmflow's `waiting` and
  `queue.join`. The cell is marked as having awaited them, so rlmflow doesn't also deliver their
  results as messages.
- **Nothing is harness-specific.** Every rlmflow harness offers `flow.tools` through its REPL and
  adds `config.prompt` sections to its own, so the recipe uses only those:
  - **`config.prompt`** carries the static sections:
    - `evolution`: the shared policy;
    - `guidance`: the `guidance` argument, for an outer loop that improves the policy between runs;
    - `long cells`: tells agents that reach their REPL through `rlmflow repl` to run `mutate`
      with `--timeout 3600`.

    It also blanks rlmflow's `agents` and `configs` sections, which describe `launch_agent`.
  - **An agent's task** carries its role and starting budget: for the root, the generations;
    for a child, its instruction and its parents' source.
  - **Tool results** carry what is left. `evaluate` returns `evals_left`, and `mutate` returns
    `{"ops": [...], "evals_left", "agents_left"}`, plus `generations_left` at the root.
  - **A tool's `when`** hides `mutate` from an agent that can't use it.
- **`Scratch`** is the Flow's workspace: every agent works in its own empty directory under
  `agents/`, so no agent's file edits reach another's, or the repository.
- **One config runs every agent.** `evorlm(task, agent)` takes an `AgentConfig` or a model name.
  The root and every child use it, as the config named `evorlm`. Its `max_iters` is at least
  `3*generations + 4`, or for a model name the original's `max(6, 2*eval_budget + 4,
  3*generations + 4)`. An op with more evaluations than that allows gets a copy with
  `2*evals + 4` turns, named `evorlm-<turns>`.
- **The policy follows the original's prompt:**
  - a numbered strategy;
  - three REPL examples: reading the population, a `mutate` whose op gets two agent calls, and
    the final submission;
  - "choose width versus recursive depth";
  - "your caller sees only that submission".

  Agents may run code in their REPL to check it; only `evaluate` scores count.
- **Records:** each run's graph is checkpointed under `rlm/generation-N/` in the experiment. Every
  measurement is a `draft` note, every root `mutate` a `generation` note with its ops and results,
  and the run's token usage and reported cost go in an `agents` note at the end.

## Differences from the original

- **No `name` on ops.** Results come back in op order.
- **Lineage through local candidates is traced back** to recorded ancestors in the experiment's
  history, rather than kept in one global store as in the original. The local steps are in the
  population logs.
- **The root is reminded of unspent generations.** The original only asks it, in the prompt, not
  to submit early.
- **Dollar costs reach the experiment only at the end** (the `agents` note). `max_tokens` is the
  live limit, as rlmflow's whole-run `max_budget`. With the REPL harness that note's `cost_usd`
  is 0: rlmflow records tokens, and only external harnesses report dollars.
- **Any harness.** The original was REPL-only; here a config with any rlmflow harness, such as
  `claude_code` or `pi`, runs every agent in it. Claude Code needs its shell timeout raised
  (`BASH_DEFAULT_TIMEOUT_MS`, `BASH_MAX_TIMEOUT_MS`), which the circle-packing example sets.

## Not yet checked

- **For Claude Code and pi, what counts as a turn is unclear.** "One per turn" is keyed by the
  rlmflow node a tool call comes from. For a session harness, that node may span several
  `rlmflow repl` calls, or each call may be its own node; I haven't confirmed which.
- **Large files are passed whole.** Candidates for file tasks travel as `{path: text}` dicts in
  every `candidate`, `evaluate` and `submit` call, which may be heavy for big files or awkward for
  Claude Code. The alternative would give each agent its own checked-out directory.
