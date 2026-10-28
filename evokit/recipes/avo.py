"""AVO (Agentic Variation Operators, NVIDIA 2026): one coding agent, in one continuous session,
decides itself when a version is ready.

The paper writes the variation step as Vary(P_t) = Agent(P_t, K, f). The agent reads the lineage
P_t with log, show, diff and grep, scores drafts with f (score), and submits a version when it is
satisfied (submit). A submitted version is kept if it is correct and matches or beats the best
(BestArchive(ties="accept") with an evaluate that fails incorrect programs). When `stall_after`
drafts and submissions in a row have not strictly improved the best, a supervisor (the same
agent, or `reviewer`, in a session of its own that can only read the experiment) reviews it and names
new directions, which reach the agent at its next step.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from evokit.agents.harness import Agent, Turn
from evokit.archive import Archive
from evokit.archives import BestArchive
from evokit.experiment import Experiment
from evokit.operators.prompts import brief, describe_individual
from evokit.records import Individual
from evokit.task import Task

PROMPT = """You are the variation operator (AVO). Your working directory holds the current best
program. Improve it, for as long as the run lasts.

{best}

Work in a loop, and do not stop to ask whether to continue:
1. Study the program and what was tried before.
2. Edit the files, then score them, as often as you need: when a version fails or scores worse,
   diagnose why and revise.
3. When a version is good, submit it: a one-line title, and what you changed, why, and its
   measured effect. It is kept if it is correct and scores at least as well as the best so far.
   After a rejected version, decide whether to keep working from your files or go back to the
   best one.
4. Go on with the next idea.

When a tool result has a "message", or a message reaches you, it comes from a supervisor who
reviewed the whole run: follow one of its directions next."""

CONTINUE = """Your turn ended, but the run goes on. Continue the loop: study, edit, score, submit.
The best version is {best}."""

SUPERVISOR = """You are the supervisor of an evolutionary search that has stalled: the agent's last
{stalled} attempts (drafts and submissions) did not improve the best score.{instructions}

Review the trajectory with your tools: stats for the best version and the totals, log --all for
every submission (kept or not) with its score, show and diff for what each one changed, notes
draft for the scores of the agent's drafts, and notes for what agents learned. What has been
tried, what worked, and what keeps failing? Then give 3 to 5 concrete optimization directions that
differ from what was tried, each with one line on why it could beat the best."""

DIRECTIONS = "A supervisor reviewed the whole run. Follow one of these directions next:\n\n{directions}"


@dataclass
class Directions:
    """The new directions."""

    directions: list[str]


async def avo(
    task: Task,
    agent: Agent,
    *,
    dir: str | Path | None = None,
    archive: Archive | None = None,
    reviewer: Agent | None = None,
    max_cost: float | None = 5.0,
    max_evaluations: int | None = None,
    max_failures: int | None = 10,
    stall_after: int | None = 5,
    resume: str | None = None,
) -> Individual | None:
    """Run AVO on `task` until a limit is reached; returns the best individual.

    `archive` defaults to BestArchive(ties="accept"). `max_failures` counts turns in a row that
    submit nothing. `stall_after` attempts without a strictly better score call the supervisor
    (None: none), which is `reviewer` or else a fresh copy of `agent`. `resume` continues an
    earlier conversation by its id.
    """
    archive = archive if archive is not None else BestArchive(ties="accept")
    experiment = task.create_experiment(dir, archive=archive, max_cost=max_cost, max_evaluations=max_evaluations, max_failures=max_failures)
    await agent.check()
    async with experiment.start(agent, resume=resume):
        prompt = PROMPT.format(best=describe_individual(experiment.best()))
        while not experiment.limits.reached():
            async for event in agent.prompt(prompt):
                if stalled(experiment, stall_after):
                    experiment.note("review")
                    if directions := await review(experiment, reviewer or agent.fresh()):
                        agent.send(directions)
                if isinstance(event, Turn) and "submit" not in event.calls and not event.stopped:
                    experiment.note("failed", event.error or "the agent's turn ended without submitting anything")
            prompt = CONTINUE.format(best=brief(experiment.best()))
    experiment.note("stop", experiment.limits.reached() or "the agent stopped")
    return experiment.best()


def stalled(experiment: Experiment, after: int | None) -> bool:
    """Whether the last `after` attempts (drafts and submissions) and the time since the last
    review went by without a strictly better score."""
    return bool(after) and min(experiment.since_improvement(), experiment.since_note("review")) >= after


async def review(experiment: Experiment, reviewer: Agent) -> str | None:
    """`reviewer` reads the stalled experiment and names new directions for the agent."""
    instructions = experiment.task.instructions
    prompt = SUPERVISOR.format(stalled=experiment.since_improvement(), instructions=f"\n\nThe task:\n{instructions}" if instructions else "")
    turn = await reviewer.prompt(prompt, output=Directions)
    if turn.output is None or not turn.output.directions:
        return None
    return DIRECTIONS.format(directions="\n".join(f"- {d}" for d in turn.output.directions))
