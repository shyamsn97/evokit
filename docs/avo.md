# AVO and autoresearch, as published

**Status: AVO implemented** (`evokit/recipes/avo.py`); **autoresearch: proposal.** The
knowledge base K is left out on purpose: a task's `TASK.md` and its files are what the agent has.

Sources:

- the AVO paper, *AVO: Agentic Variation Operators for Autonomous Evolutionary Search*
  (Chen, Ye, Xu et al., NVIDIA, 2026, [arXiv:2603.24517](https://arxiv.org/abs/2603.24517)),
  and NVIDIA's [blog post](https://developer.nvidia.com/blog/nvidia-avo-reaches-100-on-arc-agi-3-demonstrating-a-frontier-level-general-purpose-architecture-for-long-horizon-autonomous-agents/).
  NVIDIA released no code;
- the one open reproduction, [gatordevin/avo](https://github.com/gatordevin/avo)
  (`src/avo/prompts.py`, `loop.py`, `run.py`, and `docs/PAPER_MAP.md`);
- Karpathy's [autoresearch](https://github.com/karpathy/autoresearch), whose whole method is
  `program.md`.

Our first real run (`examples/circle_packing/runs/avo-2`: 1.60 to 2.62 in three steps, $4) worked,
but it was not AVO as published: one Claude Code session per step, ten draft evaluations each,
the end of the session as the commit, and a fixed sentence as the supervisor. This doc says what
AVO is, how the recipe now does it, and what autoresearch still needs.

## What AVO is

The paper writes the variation step as one agent run:

```text
Vary(P_t) = Agent(P_t, K, f)
```

> where P_t = {(x_1, f(x_1)), ..., (x_t, f(x_t))} is the full lineage of solutions and their
> scores, K is a domain-specific knowledge base, and f is the scoring function. (§3.1)

The agent gets exactly those three things and decides everything else:

- **P_t, the lineage.** "Each committed version x_i is persisted as a git commit along with its
  score" (§3.3). "The agent frequently examines multiple prior implementations in P_t within a
  single variation step, comparing their profiling characteristics" (§3.2).
- **K, the knowledge base.** "CUDA programming guides, PTX ISA documentation, Blackwell
  architecture specifications, and existing kernel implementations including FlashAttention-4
  source code" (§3.1).
- **f, the scorer, called by the agent.** "The agent then invokes f to test the result. When a
  candidate fails correctness checks or fails to improve ... the agent diagnoses the issue and
  revises its approach, repeating this edit-evaluate-diagnose cycle until it commits a
  satisfactory x_{t+1}" (§3.2). No limit on calls is mentioned.
- **The commit rule.** "We persist a new committed version only when it passes correctness checks
  and matches or improves the benchmark score relative to the best committed version so far;
  unsuccessful intermediate attempts remain part of the agent's internal search trajectory but
  are not added to the committed lineage" (§3.2).
- **Memory.** "It maintains persistent memory through its conversation history, which accumulates
  the full context of prior edits, compiler outputs, profiling results, and reasoning across the
  evolutionary process" (§4.1). One conversation for the whole run.
- **A continuous loop.** "The AVO agent operates as a continuous loop that periodically produces
  new solutions" (§3.3). The 7-day run: 40 commits out of "over 500 candidate optimization
  directions" (§4.4).
- **The supervisor.** For stalls and "unproductive cycles of edits", "the mechanism reviews the
  overall evolutionary trajectory and steers the search toward several candidate optimization
  directions" (§3.3).
- **Single lineage, on purpose.** "We study the single-lineage setting to isolate the effect of
  the operator itself" (§2.1). Populations and islands are future work.

## How ours does it

| the paper | evokit |
| --- | --- |
| lineage P_t | `evokit log --all`, `show`, `diff`, `grep` over the run's history ([reading-runs.md](reading-runs.md)), with the skill installed in the workspace |
| knowledge base K | not modeled: `TASK.md` and the task's files |
| the agent calls f | `$EVOKIT_DRAFT_EVAL`, no limit by default (`max_draft_evals=None`); nothing is recorded |
| the agent commits | `$EVOKIT_COMMIT "title" "description"`: recorded, evaluated and told at once; the reply is the verdict |
| commit rule | `BestArchive(ties="accept")`, and an `eval_fn` that fails incorrect programs |
| memory | one Claude Code session for the run, resumed with "continue" when it ends |
| continuous loop | the session runs until a limit; the recipe resumes it while none is reached |
| supervisor | after `stall_after=5` drafts and commits without a strictly better score, a fresh read-only session reviews the run and names 3 to 5 directions |
| single lineage | every commit's parent is the archive's best |

### The session

The recipe writes the best version into a workspace and starts one agent run with this
contract (abridged from `PROMPT`):

```text
Work in a loop, and do not stop to ask whether to continue:
1. Study the program and what was tried before. Every earlier version, kept or rejected, with its
   score: `evokit log --all`, `evokit show ID [PATH]`, `evokit diff A [B]`, `evokit grep PATTERN`.
2. Edit the files, then score them: `$EVOKIT_DRAFT_EVAL`. ... Use it as often as you need.
3. When a version is good, commit it: `$EVOKIT_COMMIT "one-line title" "what you changed, why,
   and its measured effect"`. ... after a rejected commit, decide whether to keep working from
   your files or go back to the best one.
4. Go on with the next idea.

When a reply has "stop", the run is over: stop at once. When a reply has "directions", they come
from a supervisor who reviewed the whole run: follow one of them next.
```

When the session ends (turn limit, timeout, the agent deciding it is done), the recipe records
it and, if no limit is reached, resumes the same conversation with "Continue the loop", the
current best, and any pending directions.

```python
# evokit/recipes/avo.py
while not self.stop_reason():
    self.commits = 0
    result = await self._session(prompt, env)   # agent.with_budget(remaining).run(workspace, prompt, session=...)
    self._record(result)                        # a "session" note with its cost; "failed" if nothing was committed
    prompt = self._continue()
```

### A commit

`$EVOKIT_COMMIT` and `$EVOKIT_DRAFT_EVAL` are small scripts that talk to the recipe over a
localhost socket (`AgentCommands`), so a commit goes through the same history and archive as
every other recipe:

```python
async def commit(self, args):
    if reason := self.stop_reason():
        return {"error": f"the run is over ({reason}); stop now", "stop": reason}
    genome = self.project.read(self.workspace)          # edits outside the genome are refused
    best = self.archive.best()
    child = Individual.new(genome, [best.id], {"operator": "avo"}, name=title, description=description)
    history.record_created(self.storage, [child], [best.id])
    (child,) = await evokit.evaluate(self.storage, [child], eval_fn=self.eval_fn)
    kept = bool(await evokit.tell(self.storage, self.archive, [child]))
    ...
    return {"id": child.id, "kept": kept, "objectives": ..., "feedback": ..., "best": {...}, **self._news()}
```

The agent sees, for example:

```json
{"id": "61b9591be3", "kept": true, "objectives": {"score": 1.6401}, "feedback": "valid packing, ...",
 "best": {"id": "61b9591be3", "objectives": {"score": 1.6401}}}
```

An unchanged genome is refused ("nothing changed since the best version"), so a commit is always a
new version. The AVO loop does not go through `ask`, because one agent run produces any number of
versions; the history, archive, limits and driver lock are the same.

### Limits

A continuous session needs its stops. They are the run's `Limits`, checked in four places:

- **Every draft and commit.** Once a limit is reached, the reply carries `"stop": reason` and the
  commit is refused. The agent then has `STOP_GRACE` (60 s) to end; after that it is cancelled.
- **Every API message.** The money spent is what is recorded plus the live cost of the agent runs
  going on now: the session and the supervisor share `max_cost`, with nothing set aside for
  either. When the sum reaches it, the run stops as above.
- **Before resuming.** The session is resumed only while no limit is reached.
- **Inside the agent.** Each start or resume, and each supervisor run, gets the remaining budget
  as its own: `agent.with_budget(max_cost - spent)`, which for Claude Code is `--max-budget-usd`.
  The agent's `timeout` and `max_turns` still apply.

The live cost comes from `agent.with_meter(on_cost)`. Claude Code runs with
`--include-partial-messages`, whose `message_start` and `message_delta` events carry each API
message's token counts; they are priced at the rate Claude Code itself charged for that model,
learned from a finished run's `modelUsage` (the one-turn `check()` before the loop teaches the
run's model). Anthropic's prices keep fixed ratios (output 5x input, cache writes 1.25x, cache
reads 0.1x), so one rate per model is enough. Checked against recorded sessions, this is exact for
the agent's own messages; it reads 1-2% low overall, because Claude Code's background calls
(Haiku) are not in the stream. A model with no learned rate gets no estimate rather than a guess,
and then only the recorded costs and `--max-budget-usd` apply.

When a run ends, its reported cost replaces the estimate. A run that is cancelled before reporting
(after `STOP_GRACE`, or at the end of the run for the supervisor) is recorded at its estimate,
also kept as `estimated_cost_usd`. Every reply carries `budget_left_usd` while the estimate is
known, so the agent can commit before the money runs out, and the log prints the spend every $0.25.

`max_evaluations` counts commits (and the seed); there is no separate commit limit.
`max_failures` counts sessions in a row that end without a commit, whatever the reason; a session
that hits its turn limit after committing is a normal end. So an agent that only drafts cannot
run forever. `max_draft_evals` caps drafts per session if set.

### The supervisor

```python
def _progress(self, attempt: Individual, best: Individual) -> None:
    """Count an attempt toward a stall unless it strictly beats `best`; call the supervisor after `stall_after`."""
```

A stall is counted in attempts: every draft and every commit that does not strictly beat the
best score, reset by one that does (a draft counts as progress before it is committed). Counting
commits alone does not work: the agent commits only versions its drafts showed were better, so
in a real run (`runs/avo-4`) all 9 commits improved while the agent drafted the same score over
and over, and a commit-based trigger never fired. Ties are kept, so counting rejections would
miss a plateau too.

After `stall_after` attempts (default 5; `None` turns the supervisor off), the recipe starts the
supervisor in the background and the count starts over. In the $5 runs (`runs/avo-4`, `avo-5`), the
agent improved every 1 to 4 attempts for most of the run and plateaued for 5 to 10 near the end,
so 5 fires at the plateau without interrupting steady progress. The supervisor is the same agent, in a fresh empty directory (it can read
the run with `evokit log/show/diff/grep` but has no program to edit), given the objective, the
task, the best version, the last 20 commits, the scores of the recent drafts, and a diff of what
the agent is working on against the best:

```text
- 3f2a9c1b07  score=2.61  rejected  SLSQP polish of the outer ring
- 1f7ec51182  score=2.62  kept      Hexagonal core with jittered outer ring
...
The scores of the agent's recent drafts (oldest first): score=2.527, score=2.526, ...

What the agent is working on now, against the best version:
--- 61f900229e/main.py
+++ workspace/main.py
...
Review the trajectory: what has been tried, what worked, and what keeps failing. Then reply with
3 to 5 concrete optimization directions that differ from what was tried, each with one line on
why it could beat the best.
```

It runs in the background because a commit's reply cannot wait minutes (Claude Code's shell
commands time out). Its reply is recorded as a `redirect` note with its cost, and reaches the
agent once: in the next draft or commit reply as `"directions"`, or in the "continue" prompt.

### Records

Per session: a `session` note (commit count, the agent's last message, its cost and draft
evaluations), and its transcript at `sessions/<id>/agent.jsonl`. Per commit: `created` and
`evaluated` records with the agent's title and description as the version's name and description,
so `evokit log` reads like a git log of the run.

## What autoresearch is

All of autoresearch is `program.md`: a prompt telling one agent to run, in one session:

```text
LOOP FOREVER:
1. Look at the git state: the current branch/commit we're on
2. Tune train.py with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: uv run train.py > run.log 2>&1
5. Read out the results: grep "^val_bpb:\|^peak_vram_mb:" run.log
...
8. If val_bpb improved (lower), you "advance" the branch, keeping the git commit
9. If val_bpb is equal or worse, you git reset back to where you started
```

- One conversation for the whole run: "do NOT pause to ask the human if you should continue."
- Every run of the experiment is a recorded attempt (`results.tsv`: commit, val_bpb, memory,
  keep/discard/crash, description). There are no drafts.
- Keep only if strictly better. Equal or worse is reset.
- The simplicity rule is the agent's judgment: "A 0.001 val_bpb improvement that adds 20 lines
  of hacky code? Probably not worth it. ... An improvement of ~0 but much simpler code? Keep."
- No supervisor, no knowledge base beyond the repo, and no lineage beyond the git branch.

### Where ours differs

```python
# evokit/recipes/autoresearch.py (now)
"""... A fresh agent session makes each change."""
operator = CodingAgentOperator(agent, project, instructions=instructions)   # persistent=False
```

| | autoresearch | evokit now |
| --- | --- | --- |
| memory | one conversation | a fresh session for every change |
| what is kept | strictly better; simplicity is the agent's call | `SimplerBestArchive`: better, or tied and fewer lines |
| who runs the experiment | the agent, then it reads the result | evokit, after the session ends |
| `results.tsv` | written by the agent | written by the recipe (same columns minus memory) |

The fresh session is the big one. In the original, the agent remembers every experiment it ran
and why it failed; ours starts each change knowing only the current best.

**The change (not made yet).** Autoresearch is AVO's commit loop with different settings: every
evaluation is an attempt (no drafts), the commit rule is "strictly better", there is no
supervisor, and the prompt is `program.md`'s, simplicity rule included. The line-count archive
goes; the agent judges simplicity, as in the original.

## The two, side by side

With AVO's commit loop in place, autoresearch is the same loop with different settings, which is
the honest comparison:

| | autoresearch | AVO |
| --- | --- | --- |
| an evaluation is | an attempt | a draft, until the agent commits |
| commit rule | strictly better | matches or improves |
| lineage the agent reads | `evokit log` | `evokit log` |
| supervisor | none | after 5 drafts and commits without a new best |
| memory | one conversation | one conversation |
