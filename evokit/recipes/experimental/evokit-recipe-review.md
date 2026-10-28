# Review: `evokit/recipes/experimental/evorlm.py`

Second review, 2026-10-28, of the revised recipe (437 lines). It is checked
against the EvoRLM reference implementation (`evorlm/flow.py`, `tools.py`,
`budget.py`, `session.py`, `prompts.py`), [`algorithm.md`](algorithm.md), and
rlmflow 0.6 from `~/Code/rlmkit`. Line numbers refer to the revised recipe.

## Verdict

The design is right, and every finding from the first review is fixed. The
recipe now does everything the reference does in the inner loop, and two
things the reference does not yet do:

- It enforces one `evaluate` per turn, which `algorithm.md` asks for.
- It verifies an unmeasured hand-up out of the operator's leftover budget
  instead of dropping it.

All 10 recipe tests and the full evokit suite (109 tests) pass. Both bugs from
the first review were re-run with the same scripted reproductions, and both are
now fixed.

What remains is one risk to test before using the Claude Code or Pi harnesses,
and some small issues. None of them block merging the recipe as experimental.

## First-review findings

| Finding | Status | How it was fixed |
| --- | --- | --- |
| The root's own candidates never reach the archive | Fixed | After the stream ends, the root's result goes through `handed_up` and `admit` (line 173). Re-run: the root's 3.0 is now the run's best. |
| An op that omits `evals` gets 1 | Fixed | The default is `eval_budget` at the root and an even split below it (line 317). Re-run: the op now gets 5. |
| No agent budget, so recursion is nearly free | Fixed | `Scope.agents` is carved and refunded like `evals`; an op's own launch costs one; `mutate` checks the token cap (lines 311, 323–331, 366). |
| Fixed `max_iters` | Partly fixed | Scales with `generations` and `eval_budget` when `agent` is a model name (line 156); see issue 3 below. |
| Parent ids could point at unrecorded candidates | Fixed | Local admissions are kept in `self.local`, and `lineage()` rewrites parents to the nearest recorded ancestors (lines 398–422). |
| A local population is not best-first at first | Fixed | Each operator's population is now a `PopulationArchive`, persisted under `populations/<agent id>/`. |
| The root learns little from each operator | Fixed | Results include `instruction`, `parent_ids`, `notes`, `unchanged`, `evals_used` and `agents_used` (lines 356–366). |
| Operators start without parent source | Fixed | The `CHILD` task message shows each parent's id, score, title and source, capped at 20k characters per file. |
| No hook for a learned policy | Fixed | `guidance=` adds a prompt section for every agent. |

Two additions go beyond the first review:

- The root is reminded once per generation if it submits with generations
  left, and is resumed (lines 167–179).
- `evaluate` now returns a scalar `score` next to `objectives`.

## Risk to test: `mutate` under the Claude Code and Pi harnesses

`await mutate(...)` blocks its cell until every child finishes. A generation
can take many minutes: children run their own model turns, evaluations, and
possibly their own `mutate`.

In the REPL harness this is fine, because worker cells have no execution
timeout (`execution_timeout=None` in rlmflow's `Runtime`). The session
harnesses are different. Their cells go through the `rlmflow repl` client,
which gives up after 600 seconds (`REPL_TIMEOUT` in `agent_tools/cli.py` and
`mcp.py`), and the session's own tool timeout may be shorter. rlmflow's
session prompt warns about exactly this: "A cell that awaits long work can hit
your tool timeout, so launch agents and end the turn instead."

If the client times out, the cell probably keeps running on the host. That
would spend the generation's budget and admit the children's results, but the
model never sees them. It may then retry, which spends another generation.
This has not been run end to end. Only the REPL harness is tested.

**Suggestion:** run one real generation under `claude_code` with children that
take more than 10 minutes. If it breaks, give session harnesses a
non-blocking `mutate`: launch the children, end the turn, and deliver the
results when the agent is resumed, which is rlmflow's normal handoff for
session harnesses.

A related detail: under session harnesses, each `rlmflow repl` cell gets its
own anchor node. So "one evaluate per turn" and "one mutate per turn" mean one
per cell. That still forces the model to read each result before the next
call, so it is fine, but the prompt's word "turn" means something different
there.

## Smaller issues

**1. `mutate` ignores the experiment's own limits.** `measure` checks
`experiment.limits.reached()`, but `mutate` checks only the token cap and
`can_mutate` (lines 311–314). With a `max_evaluations` or `max_cost` limit
reached, the root can still launch a full generation. Every child then burns
turns only to be told "the run is over" on its first `evaluate`. The root
loop's exit condition (line 176) has the same gap. Today this only matters if
the experiment was created with limits, because `evorlm()` doesn't expose
`max_cost`. Exposing it would be worth doing, since it is the natural cost cap
for an agent-heavy recipe. Add the check in both places.

**2. Candidates the root makes itself have no parents.** The root's scope has
`parents=()`, so a candidate the root edits and submits is recorded with no
lineage. Accepting an optional `parent_ids` in `RETURN` would fix this, as
would recording the ids the root passed to `candidate()`.

**3. `max_iters` scales only for a model-name `agent`.** With an explicit
`AgentConfig`, every agent gets the same turn cap (for example
`max_iters=18` in the circle-packing example). With one evaluate per turn, an
op given more evaluations than turns can't spend them; they go back to the
caller unused. The root may also run out of turns on long runs. If
`self.spawn` accepts config overrides, set each child's `max_iters` from its
`evals` (roughly `evals + 3`). Otherwise, document that an explicit config
must size `max_iters` itself.

**4. Draft notes can claim improvements that never reach the archive.**
`measure` marks a draft `improved=True` whenever it beats the archive's best.
An operator's intermediate draft that it doesn't hand up, or that a better
sibling displaces, still counts. Experiment statistics derived from drafts
(attempts to improvement) then overstate progress. Computing `improved` at
admission instead would fix this.

**5. A dropped result loses the operator's notes.** In the `except` branch
(line 363), the result keeps only the instruction, parents and error. The
operator's `notes` are often most useful exactly when its candidate failed.

**6. Two operators handing up the same genome share one id.** The memo maps a
genome to one `Individual`, so the second operator's title, notes and parents
are discarded at the root, and overwrite the first in `self.local`. This is
rare and harmless for scores, but worth knowing when reading lineage.

**7. A core change affects other recipes (evokit `archive.py`).** Archives
now always log changes, to an in-memory store until they are attached. An
archive that is never attached, such as a temporary one built inside a
strategy, now keeps a growing log of encoded individuals for its whole
lifetime. Pickling and `_template` correctly drop the store. If unattached
archives are common, skip logging until `attach`.

## Suggested order

1. Test `mutate` under `claude_code` with a long generation, before advertising
   that harness.
2. Check experiment limits in `mutate` and in the root loop, and expose
   `max_cost` (issue 1).
3. Size `max_iters` per child from its evaluations (issue 3).
4. Root lineage, `improved` at admission, notes on dropped results
   (issues 2, 4, 5).
