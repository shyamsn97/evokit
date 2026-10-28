# AVO

*AVO: Agentic Variation Operators for Autonomous Evolutionary Search* (Chen, Ye, Xu et al.,
NVIDIA, 2026, [arXiv:2603.24517](https://arxiv.org/abs/2603.24517);
[blog post](https://developer.nvidia.com/blog/nvidia-avo-reaches-100-on-arc-agi-3-demonstrating-a-frontier-level-general-purpose-architecture-for-long-horizon-autonomous-agents/)).
NVIDIA released no code. The one open reproduction is
[gatordevin/avo](https://github.com/gatordevin/avo). The knowledge base K is left out on
purpose: a task's `TASK.md` and its files are what the agent has.

## What the paper specifies

The variation step is one agent run:

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

## How the recipe does it

| the paper | `evokit/recipes/avo.py` |
| --- | --- |
| lineage P_t | the `log`, `show`, `diff` and `grep` tools over the run ([reading a run](../reading-runs.md)) |
| knowledge base K | not modeled: `TASK.md` and the task's files; agents can leave notes for later with `remember` |
| the agent calls f | the `score` tool, no limit by default; it records the draft's score, nothing else |
| the agent commits | the `submit` tool: recorded, evaluated and told at once; the reply is the verdict |
| commit rule | `BestArchive(ties="accept")`, and an `eval_fn` that fails incorrect programs |
| memory | one agent session for the run, prompted again with "Continue the loop" when a turn ends |
| continuous loop | the session runs until `limits` are reached |
| supervisor | after `stall_after=5` drafts and submissions without a strictly better score, another agent (`reviewer`, by default the same one) reviews the run in an empty directory and names 3 to 5 directions |
| single lineage | every submission's parent is the archive's best |

### The loop

The whole recipe is one session and a loop over its events. The session is opened on the run, in
a checkout of the best version, with the `eval_fn`, the archive and the limits, and that is all
it needs: from them it gives the agent its tools (the read tools and `remember`, `score`, and
`submit`), records every turn in the run, sets each turn's budget, and stops the agent once a
limit is reached. What is left is AVO's own policy:

```python
checkout = project.checkout(best.genome, project.scratch / "agent")
async with agent.session(checkout, run=run, eval_fn=eval_fn, archive=archive, limits=limits, resume=resume) as session:
    while not limits.reached(run):
        async for event in session.prompt(text):
            stalled = stall_after and min(run.since_improvement(), run.since_note("review")) >= stall_after
            if stalled and (review is None or review.done()):
                run.note("review")
                review = asyncio.ensure_future(supervise(reviewer or agent, session, run, limits, instructions))
            if isinstance(event, Turn) and "submit" not in event.calls and not event.stopped:
                run.note("failed", event.error or "the agent's turn ended without submitting anything")
        text = CONTINUE.format(best=brief(archive.best()))
```

The tools reach a shell agent as an `evokit` command on its PATH (`evokit submit "title"
"description"`), and ReplAgent as Python functions. `submit` scores the checkout's files and
offers them to the archive; an unchanged genome is refused ("nothing changed since the best
version"). The agent sees, for example:

```json
{"id": "61b9591be3", "kept": true, "objectives": {"score": 1.6401}, "feedback": "valid packing, ..."}
```

### Limits

`limits` is a `Limits`: `max_evaluations` (submissions and the seed), `max_cost`, `max_failures`
(turns in a row that submit nothing, so an agent that never submits cannot run forever), and any
`stop` rules over the run, e.g. a cap on drafts: `lambda run: "200 drafts" if len(run.drafts()) >= 200 else None`.
The session checks them in three places:

- **Every tool call.** Once a limit is reached, the reply carries `"stop": reason`, further calls
  are refused, and the turn is interrupted `STOP_GRACE` (60 s) later if the agent keeps going.
- **Before each turn.** The session is prompted again only while no limit is reached.
- **Inside the agent.** Each turn, and each review, gets what is left of `max_cost` as its own
  budget, which for Claude Code is `--max-budget-usd`. The agent's `timeout` and `max_turns` still apply.

A turn's cost is known when it ends, so the run can go over `max_cost` by at most what one turn
and one review spend before their own budgets stop them.

### The supervisor

A stall is counted by the run itself: `run.since_improvement()` counts attempts, every draft and
every submission, since the best strictly improved, and `run.since_note("review")` counts them
since the last review. Counting submissions alone does not work: the agent submits only versions
its drafts showed were better, so in a real run all 9 submissions improved while the agent
drafted the same score over and over. Ties are kept, so counting rejections would miss a plateau too.

After `stall_after` attempts (default 5; `None` turns it off), the recipe starts the reviewer in
the background. In the $5 runs, the agent improved every 1 to 4 attempts for most of the run and
plateaued for 5 to 10 near the end, so 5 fires at the plateau without interrupting steady
progress. The reviewer gets a session of its own on the run, so it has the read tools and an empty
directory, and its prompt tells it how to review with them: `stats` for the best version,
`log --all` for every submission and its score, `show` and `diff` for what each changed,
`notes draft` for the drafts' scores, and `notes` for what agents learned. It replies with 3 to 5
directions that differ from what was tried. Unlike the version before, it does not see the
agent's unsubmitted edits.

It runs in the background because a tool reply cannot wait minutes (harnesses time out shell
commands). Its answer is sent to the agent with `session.send`, which delivers it once: with the
agent's next tool reply as `"message"`, or with the next prompt.

### Records

The session records every turn, the agent's and the reviewer's: a `turn` note (the reply, its
cost, the tools it called, and why it stopped, if it was stopped) and its transcript under
`sessions/<session id>/`. Per submission: `created` and `evaluated` records with the agent's
title and description as the version's name and description, so `evokit log` reads like a git
log of the run.

## How it differs from the paper and the open reproduction

NVIDIA released no code, so the references are the paper and
[gatordevin/avo](https://github.com/gatordevin/avo) at commit `f6dad9e`.

Versus the paper:

- **No knowledge base.** The paper gives the agent CUDA and PTX guides, Blackwell specs and
  FlashAttention-4's source. Here the agent has `TASK.md` and the task's files.
- **Lineage is evokit's history, not git.** The paper stores each committed version as a git
  commit with its score. Here the agent reads the lineage with `evokit log`, `show`, `diff` and
  `grep`. The workspace is a plain directory.
- **Committing is an explicit tool.** The paper describes the agent persisting a version once
  it is satisfied. Here it calls `submit`; drafts through `score` are not kept as versions.
- **Rejected submissions are recorded.** In the paper, failed attempts stay only in the agent's
  trajectory. Here a rejected submission is in the run's history (`evokit log --all`), and the
  workspace is left as it is: the prompt tells the agent to decide whether to go back to the best.
- **Correctness is the evaluator's job.** The paper scores incorrect kernels as zero and
  optimizes the geometric mean over a vector of benchmark configurations. Here there is no separate
  correctness flag: `eval_fn` returns objectives, and a task makes incorrect programs score low or
  fail.
- **The stall rule and the supervisor are our choices.** The paper only says the supervisor
  reviews the trajectory when progress stalls. The count (5 drafts and submissions without a strictly
  better score), running it in the background, in an empty directory, and what it is shown are
  evokit's decisions.
- **It stops on money, not time.** The paper ran for 7 days. Here the limits are `max_cost`,
  `max_evaluations`, `max_failures` and any stop rule (a wall-clock limit is `run.elapsed()`).

Versus gatordevin/avo:

- **A step is one agent run there, one submission here.** The reproduction's unattended loop runs the
  agent once per step, then evaluates `work/` and accepts or rejects it. Each step starts a new
  Claude process with no `--resume`; memory is `NOTES.md`, git history and the last rejection.
  Here one session can submit any number of times.
- **Its default mode has no loop.** By default, the open Claude Code or Codex session is the
  operator and calls the reproduction's MCP tools; the framework never calls a model. Its
  `avo run --backend claude_cli` mode is the closest to this recipe.
- **It provides a knowledge base and a lineage table.** It copies a `kb/` folder into the run and
  puts its file index and a rendered lineage table into every prompt. Here the prompt has the
  current best, and the agent runs `evokit log` itself.
- **It reverts on reject.** A rejected step resets `work/` and saves the patch under `rejected/`.
  Here the workspace is not reset.
- **Its supervisor triggers sooner, and blocks.** It fires after 3 steps without a better score
  (ties count as no progress), runs before the next step, and can use a different model. Here it
  fires after 5 drafts and submissions, runs in the background, and is the same agent unless
  `reviewer` is given.
- **It stops on time and steps.** Its limits are `max_steps` and `time_budget_s`. Here they are
  money, evaluations, failed turns, and any stop rule.
- **It drafts in place.** Its `./avo-eval` scores `work/` as the agent left it. Here a draft is
  scored in a copy, so the workspace is untouched.
- **It ships more around the loop:** target folders with seeds, knowledge bases and evaluators,
  baseline measurement at the start of a run, paper-style plots, `avo doctor`, and several agent
  backends.

The same in all three: `Vary(P_t) = Agent(P_t, K, f)` with the agent choosing what to read, edit
and score; no default cap on scoring calls; the commit rule "correct, and matches or improves the
best"; and a single lineage.

[autoresearch](autoresearch.md) is the same single lineage with a fresh session per change, and
with evokit running the evaluation instead of the agent.
