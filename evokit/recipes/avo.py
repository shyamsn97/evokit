"""AVO (Agentic Variation Operators, NVIDIA 2026): one coding agent, in one continuous session,
decides itself when a version is ready.

The paper writes the variation step as Vary(P_t) = Agent(P_t, K, f). Here the agent gets the
standard tools (`agent_tools`): it reads the lineage P_t with log, show, diff and grep, scores
drafts with f as often as it likes (score), and submits a version when it is satisfied (submit).
A submitted version is kept if it is correct and matches or beats the best so far
(BestArchive(ties="accept") and an eval_fn that fails incorrect programs); the agent hears the
verdict at once and keeps going in the same conversation. When `stall_after` drafts and
submissions in a row have not strictly improved the best score, a supervisor (a fresh, read-only
session of the same agent) reviews the trajectory and names several directions, which reach the
agent with its next tool result.

The run's limits are checked on every tool call and before the session is resumed. An agent's
spending is only known when its session ends, so the remaining budget is also passed to the agent
as its own (`CodingAgent.with_budget`, e.g. Claude Code's --max-budget-usd).
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import inspect
import logging
import tempfile
from collections import deque
from pathlib import Path

from evokit.agents.base import AgentResult, CodingAgent
from evokit.agents.operator import clear_directory
from evokit.agents.tools import Tool, agent_tools, call, limited, tool_name
from evokit.archive import Archive
from evokit.errors import UsageError
from evokit.files import Project
from evokit.history import History, diff_genomes
from evokit.loop import EvalFn, Limits
from evokit.operators.prompts import describe_individual
from evokit.records import Evaluation, Individual
from evokit.util import new_id, truncate

log = logging.getLogger("evokit")

PROMPT = """You are the variation operator in an evolutionary search run by EvoKit (AVO).

Your working directory holds the current best program. Improve it, for as long as the run lasts.

Editable files (the genome): {editable}
Only changes to these files are kept. Do not modify any other file; edits to them are rejected.

{best}

Work in a loop, and do not stop to ask whether to continue:
1. Study the program and what was tried before: every earlier version, kept or rejected, with its
   score, through the log, show, diff and grep tools.
2. Edit the files, then score them with the score tool. It records nothing. Use it as often as you
   need: when a version fails or scores worse, diagnose why and revise.
3. When a version is good, submit it with the submit tool: a one-line title, and what you changed,
   why, and its measured effect. It is kept if it is correct and scores at least as well as the best
   so far. After a rejected version, decide whether to keep working from your files or go back to
   the best one.
4. Go on with the next idea.

When a tool result has "stop", the run is over: stop at once. When it has a "message", it comes
from a supervisor who reviewed the whole run: follow one of its directions next.

{instructions}"""

CONTINUE = """Your session ended, but the run goes on. Continue the loop: study, edit, score, submit.
The best version is {best}.{directions}"""

SUPERVISOR = """You are the supervisor of an evolutionary search that has stalled: the agent's last
{stalled} attempts (drafts and submissions) did not improve the best score. Do not edit any files.

The objective: {objective}.{instructions}

The best version so far:
{best}

The most recent submissions (newest first; kept means it matched or beat the best):
{attempts}

The scores of the agent's recent drafts (oldest first): {drafts}

What the agent is working on now, against the best version:
{work}

Review the trajectory: what has been tried, what worked, and what keeps failing. Then reply with
3 to 5 concrete optimization directions that differ from what was tried, each with one line on
why it could beat the best."""


async def avo(
    project: Project,
    eval_fn: EvalFn,
    agent: CodingAgent,
    *,
    history: History,
    archive: Archive,
    instructions: str | None = None,
    max_cost: float = 5.0,
    max_evaluations: int | None = None,
    max_failures: int | None = 10,
    max_draft_evals: int | None = None,
    stall_after: int | None = 5,
    session: str | None = None,
) -> Individual | None:
    """Run AVO until a limit is reached. Returns the best individual.

    The caller passes a seeded archive, usually BestArchive(ties="accept"). `max_evaluations`
    counts submissions (and the seed); `max_failures` counts sessions in a row that end without a
    submission; `max_draft_evals` caps drafts per session (None: no cap, as in the paper).
    `stall_after` is the number of drafts and submissions without a strictly better score that calls
    the supervisor (None: no supervisor). `session` resumes an earlier conversation.
    """
    if archive.best() is None:
        raise UsageError("the archive is empty", hint="seed it before running AVO")
    limits = Limits(max_evaluations, max_cost, max_failures)
    await agent.check()
    workspace = project.scratch / "agent"
    workspace.mkdir(parents=True, exist_ok=True)
    await asyncio.to_thread(clear_directory, workspace)
    await asyncio.to_thread(project.write, archive.best().genome, workspace)
    supervisor = Supervisor(agent, history, archive, limits, project, workspace, instructions, stall_after)
    tools = agent_tools(archive, history, eval_fn, project=project, workspace=workspace, submit=True, info={"operator": "avo"})
    prompt = PROMPT.format(
        editable=", ".join(project.editable), best=describe_individual(archive.best()), instructions=instructions or ""
    ).rstrip()

    def stop() -> str | None:
        return limits.reached(history)

    with history.storage.lock("driver"):
        try:
            while not stop():
                session_tools = [limited(t, max_draft_evals) if max_draft_evals and tool_name(t) == "score" else t for t in tools]
                before = history.counts()["evaluated"]
                result = await agent.with_budget(limits.remaining_cost(history)).run(
                    workspace,
                    prompt,
                    session=session,
                    tools=supervisor.watch(session_tools),
                    stop=stop,
                    messages=supervisor.directions,
                )
                session = result.session or session
                record_session(history, result, history.counts()["evaluated"] - before, stopped=bool(stop()))
                best = archive.best()
                directions = supervisor.directions()
                prompt = CONTINUE.format(
                    best=f"{best.id} ({_scores(best.evaluation.objectives)})",
                    directions=f"\n\nA supervisor reviewed the run; follow one of these:\n{directions}" if directions else "",
                )
        finally:
            await supervisor.close()
    history.note("stop", stop() or "the agent stopped")
    return archive.best()


def record_session(history: History, result: AgentResult, submitted: int, stopped: bool) -> None:
    """A session's cost and transcript; one that submitted nothing, unless the run stopped it, is a failure."""
    folder = f"sessions/{new_id()}"
    if result.transcript:
        history.storage.write(f"{folder}/agent.jsonl", result.transcript.encode())
    summary = truncate(result.summary, 2000) or ""
    history.note("session", f"{submitted} submitted; logs: {folder}/\n{summary}".strip(), usage=result.usage)
    if submitted == 0 and not stopped:
        reason = f" ({result.error})" if result.error else ""
        history.note("failed", f"the agent session ended without a submission{reason}")


class Supervisor:
    """Counts the agent's drafts and submissions since the best score last strictly improved; after
    `stall_after` of them, reviews the run in a fresh, read-only session of `agent`, and keeps its
    directions until `directions()` hands them over."""

    def __init__(self, agent, history, archive, limits, project, workspace, instructions, stall_after):
        self.agent, self.history, self.archive, self.limits = agent, history, archive, limits
        self.project, self.workspace, self.instructions = project, workspace, instructions
        self.stall_after = stall_after
        self.since_best = 0
        self.recent_drafts: deque[str] = deque(maxlen=20)
        self.pending: str | None = None
        self.task: asyncio.Task | None = None

    def watch(self, tools: list[Tool]) -> list[Tool]:
        """`tools`, with every score and submit result counted toward a stall."""
        return [self._watched(t) if tool_name(t) in ("score", "submit") else t for t in tools]

    def _watched(self, tool: Tool) -> Tool:
        @functools.wraps(tool)
        async def watched(*args, **kwargs):
            best = self.archive.best()
            result = await call(tool, inspect.signature(tool).bind(*args, **kwargs).arguments)
            if tool_name(tool) == "score":
                self.recent_drafts.append(_scores(result["objectives"]))
            self.attempt(result["objectives"], best)
            return result

        return watched

    def attempt(self, objectives, best: Individual) -> None:
        score, better = getattr(self.archive, "score", None), getattr(self.archive, "better", None)
        if score is None or better is None:
            return
        value, best_value = score(Individual.new(None).with_evaluation(Evaluation(objectives))), score(best)
        if value is not None and (best_value is None or better(value, best_value)):
            self.since_best = 0
            return
        self.since_best += 1
        running = self.task is not None and not self.task.done()
        if self.stall_after is not None and self.since_best >= self.stall_after and not running:
            stalled, self.since_best = self.since_best, 0
            self.task = asyncio.create_task(self.review(stalled))

    def directions(self) -> str | None:
        directions, self.pending = self.pending, None
        return directions

    async def review(self, stalled: int) -> None:
        best = self.archive.best()
        attempts = "\n".join(
            f"- {e['id']}  {_scores(e.get('objectives'))}  {'kept' if e['admitted'] else 'rejected'}  {e['name']}"
            for e in self.history.log(all=True, limit=20)
        )
        try:
            current = await asyncio.to_thread(self.project.read, self.workspace)
            work = diff_genomes(best.genome, current, best.id, "workspace")
        except Exception:
            work = ""
        objective = getattr(self.archive, "objective", "score")
        direction = "minimize" if getattr(self.archive, "minimize", False) else "maximize"
        prompt = SUPERVISOR.format(
            stalled=stalled,
            objective=f"{direction} `{objective}`",
            instructions=f"\n\nThe task:\n{self.instructions}" if self.instructions else "",
            best=describe_individual(best),
            attempts=attempts or "(none)",
            drafts=", ".join(self.recent_drafts) or "(none)",
            work=truncate(work, 6000) or "(no changes)",
        )
        log.info("supervisor: reviewing the run")
        try:
            with tempfile.TemporaryDirectory(prefix="evokit-supervisor-") as directory:
                agent = self.agent.with_budget(self.limits.remaining_cost(self.history))
                result = await agent.run(Path(directory), prompt, tools=agent_tools(self.archive, self.history))
        except asyncio.CancelledError:
            self.history.note("supervisor-failed", "stopped: the run is over")
            raise
        if result.error or not result.summary.strip():
            self.history.note("supervisor-failed", result.error or "no reply", usage=result.usage)
            return
        self.pending = result.summary.strip()
        self.history.note("redirect", self.pending, usage=result.usage)

    async def close(self) -> None:
        if self.task is not None and not self.task.done():
            self.task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.task


def _scores(objectives) -> str:
    return ", ".join(f"{k}={v:.6g}" for k, v in (objectives or {}).items()) or "failed"
