"""AVO (Agentic Variation Operators, NVIDIA 2026): one coding agent, in one continuous session,
decides itself when a version is ready.

The paper writes the variation step as Vary(P_t) = Agent(P_t, K, f). Here the agent reads the
lineage P_t with `evokit log/show/diff/grep`, scores drafts with f as often as it likes
($EVOKIT_DRAFT_EVAL), and commits a version when it is satisfied ($EVOKIT_COMMIT). A commit is
kept if it is correct and matches or beats the best so far (BestArchive(ties="accept") and an
eval_fn that fails incorrect programs); the agent hears the verdict at once and keeps going in
the same conversation. When `stall_after` drafts and commits in a row have not strictly improved
the best score, a supervisor (a fresh, read-only session of the same agent) reviews the trajectory and names
several directions, which reach the agent with its next command's reply.

The run's limits are checked on every draft and commit, and before the session is resumed.
An agent's spending is only known when its session ends, so `max_cost` is also passed to the
agent as its own budget (`CodingAgent.with_budget`, e.g. Claude Code's --max-budget-usd).
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import logging
import tempfile
from collections import deque
from pathlib import Path

import evokit
from evokit import history
from evokit.archive import Archive
from evokit.errors import EnvironmentModified, UsageError
from evokit.files import CodingAgent, Project
from evokit.files.agents import AgentResult
from evokit.files.coding_agent import AgentCommands, Drafts, clear_directory, run_access, run_directory
from evokit.interface.history import HistoryCommands, diff_genomes
from evokit.loop import EvalFn, Limits
from evokit.operators.prompts import describe_individual
from evokit.records import Individual, merge_usage
from evokit.storage import Storage
from evokit.util import NAME_LIMIT, new_id, title_and_body, truncate

log = logging.getLogger("evokit")

PROMPT = """You are the variation operator in an evolutionary search run by EvoKit (AVO).

Your working directory holds the current best program. Improve it, for as long as the run lasts.

Editable files (the genome): {editable}
Only changes to these files are kept. Do not modify any other file; edits to them are rejected.

{best}

Work in a loop, and do not stop to ask whether to continue:
1. Study the program{lineage}.
2. Edit the files, then score them: `"$EVOKIT_DRAFT_EVAL"`. It prints the evaluation as JSON and
   records nothing. Use it as often as you need: when a version fails or scores worse, diagnose why and
   revise.
3. When a version is good, commit it: `"$EVOKIT_COMMIT" "one-line title" "what you changed, why,
   and its measured effect"`. It is kept if it is correct and scores at least as well as the best so far.
   The reply says whether it was kept and gives the best version's id; after a rejected commit,
   decide whether to keep working from your files or go back to the best one.
4. Go on with the next idea.

$EVOKIT_DRAFT_EVAL and $EVOKIT_COMMIT are environment variables holding the two commands' paths;
run them exactly as written. Only $EVOKIT_COMMIT submits a version: do not use
`evokit evaluate-directory` in this run.

Replies show `budget_left_usd`, the run's remaining budget; commit good work before it runs out.
When a reply has "stop", the run is over: stop at once. When a reply has "directions", they come
from a supervisor who reviewed the whole run: follow one of them next.

{instructions}"""

LINEAGE = (
    " and what was tried before. Every earlier version, kept or rejected, with its score: "
    "`evokit log --all`, `evokit show ID [PATH]`, `evokit diff A [B]`, `evokit grep PATTERN`"
)

CONTINUE = """Your session ended, but the run goes on. Continue the loop: study, edit, score with
`"$EVOKIT_DRAFT_EVAL"`, commit with `"$EVOKIT_COMMIT" "title" "description"`.
The best version is {best}.{directions}"""

SUPERVISOR = """You are the supervisor of an evolutionary search that has stalled: the agent's last
{stalled} attempts (drafts and commits) did not improve the best score. Do not edit any files.

The objective: {objective}.{instructions}

The best version so far:
{best}

The most recent commits (newest first; kept means it matched or beat the best):
{attempts}

The scores of the agent's recent drafts (oldest first): {drafts}

What the agent is working on now, against the best version:
{work}
{lineage}
Review the trajectory: what has been tried, what worked, and what keeps failing. Then reply with
3 to 5 concrete optimization directions that differ from what was tried, each with one line on
why it could beat the best."""

STOP_GRACE = 60.0


async def avo(
    project: Project,
    eval_fn: EvalFn,
    agent: CodingAgent,
    *,
    storage: Storage,
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
    counts commits (and the seed); `max_failures` counts sessions in a row that end without a
    commit; `max_draft_evals` caps drafts per session (None: no cap, as in the paper).
    `stall_after` is the number of drafts and commits without a strictly better score that calls
    the supervisor (None: no supervisor). `session` resumes an earlier conversation.
    """
    if archive.best() is None:
        raise UsageError("the archive is empty", hint="seed it before running AVO")
    limits = Limits(max_evaluations, max_cost, max_failures)
    await agent.check()
    run = _Run(project, eval_fn, agent, storage, archive, limits, instructions, max_draft_evals, stall_after, session)
    with storage.lock("driver"):
        await run.go()
    history.note(storage, "stop", limits.reached(storage) or "the agent stopped")
    return archive.best()


class _Run:
    def __init__(self, project, eval_fn, agent, storage, archive, limits, instructions, max_draft_evals, stall_after, session):
        self.project, self.eval_fn, self.agent = project, eval_fn, agent
        self.storage, self.archive, self.limits = storage, archive, limits
        self.instructions, self.stall_after, self.session = instructions, stall_after, session
        self.max_cost = limits.max_cost
        self.workspace = project.scratch / "agent"
        self.drafts = Drafts(project, eval_fn, self.workspace, [archive.best().id], max_draft_evals, stop=self.stop_reason)
        self.history = HistoryCommands(storage, archive)
        self.stopped = asyncio.Event()
        self.directions: str | None = None
        self.supervisor: asyncio.Task | None = None
        self.live: dict[object, float | None] = {}  # running agents' costs so far (None: unknown)
        self.logged_spend = 0
        self.since_best = self.stalled = 0
        self.recent_drafts: deque[str] = deque(maxlen=20)
        self.commits = 0

    def stop_reason(self) -> str | None:
        reason = self.limits.reached(self.storage)
        if not reason and self.max_cost is not None and self.spent() >= self.max_cost:
            reason = f"${self.max_cost:g} spent"
        if reason:
            self.stopped.set()
        return reason

    # spending: what is recorded, plus the live cost of the agent runs going on now (the session and
    # the supervisor share one budget)

    def spent(self) -> float:
        return history.usage(self.storage)["cost_usd"] + sum(c or 0.0 for c in self.live.values())

    def remaining(self) -> float | None:
        if self.max_cost is None:
            return None
        return max(0.0, self.max_cost - self.spent())

    def _metered(self, agent: CodingAgent) -> tuple[CodingAgent, object]:
        """`agent` with the remaining budget as its own limit, reporting its cost into `live`."""
        key = object()
        agent = agent.with_budget(self.remaining())
        metered = agent.with_meter(lambda cost: self._on_cost(key, cost))
        self.live[key] = 0.0 if metered is not agent else None
        return metered, key

    def _on_cost(self, key: object, cost: float | None) -> None:
        if key not in self.live:
            return
        self.live[key] = cost
        if self.max_cost is not None and None not in self.live.values():
            spent = self.spent()
            if int(spent / 0.25) > self.logged_spend:
                self.logged_spend = int(spent / 0.25)
                log.info("spent: ~$%.2f of $%g", spent, self.max_cost)
        self.stop_reason()

    def _settle(self, key: object, usage: dict) -> dict:
        """An agent run's usage to record: its reported cost, or, if it was stopped before reporting
        one, its live estimate (also kept as `estimated_cost_usd`)."""
        estimate = self.live.pop(key, None)
        if "cost_usd" not in usage and estimate:
            usage = {**usage, "cost_usd": estimate, "estimated_cost_usd": estimate}
        return usage

    # the agent's commands

    def _news(self) -> dict:
        """What every reply carries: the supervisor's directions (once), and whether to stop."""
        out: dict = {}
        if self.directions:
            out["directions"], self.directions = self.directions, None
        reason = self.stop_reason()
        if reason:
            out["stop"] = reason
        elif self.max_cost is not None and None not in self.live.values():
            out["budget_left_usd"] = round(self.remaining(), 2)
        return out

    async def draft(self, args: list[str]) -> dict:
        count = self.drafts.count
        reply = await self.drafts(args)
        if self.drafts.count > count:
            draft = self.drafts.last
            self.recent_drafts.append(_scores(draft.evaluation.objectives))
            self._progress(draft, self.archive.best())
        return {**reply, **self._news()}

    async def commit(self, args: list[str]) -> dict:
        reason = self.stop_reason()
        if reason:
            return {"error": f"the run is over ({reason}); stop now", "stop": reason}
        try:
            genome = await asyncio.to_thread(self.project.read, self.workspace)
        except EnvironmentModified as exc:
            return {"error": exc.message, "hint": exc.hint}
        best = self.archive.best()
        if genome == best.genome:
            return {"error": "nothing changed since the best version; edit the files first", "best": best.id}
        name, description = (list(args) + ["", ""])[:2]
        if len(name) > NAME_LIMIT or "\n" in name:
            name, rest = title_and_body(name)
            description = "\n\n".join(p for p in (rest, description) if p)
        child = Individual.new(genome, [best.id], {"operator": "avo"}, name=name, description=truncate(description, 4000))
        history.record_created(self.storage, [child], [best.id])
        (child,) = await evokit.evaluate(self.storage, [child], eval_fn=self.eval_fn)
        kept = bool(await evokit.tell(self.storage, self.archive, [child]))
        self.commits += 1
        self._progress(child, best)
        best = self.archive.best()
        self.drafts.parents[:] = [best.id]
        evaluation = child.evaluation
        reply = {"id": child.id, "kept": kept, "objectives": evaluation.objectives, "feedback": truncate(evaluation.feedback, 4000)}
        reply["best"] = {"id": best.id, "objectives": best.evaluation.objectives}
        return {**reply, **self._news()}

    # the supervisor

    def _progress(self, attempt: Individual, best: Individual) -> None:
        """Count an attempt toward a stall unless it strictly beats `best`; call the supervisor after `stall_after`."""
        score, better = getattr(self.archive, "score", None), getattr(self.archive, "better", None)
        if score is None or better is None:
            return
        value, best_value = score(attempt), score(best)
        if value is not None and (best_value is None or better(value, best_value)):
            self.since_best = 0
            return
        self.since_best += 1
        running = self.supervisor is not None and not self.supervisor.done()
        if self.stall_after is not None and self.since_best >= self.stall_after and not running:
            self.stalled, self.since_best = self.since_best, 0
            self.supervisor = asyncio.create_task(self._supervise())

    async def _supervise(self) -> None:
        best = self.archive.best()
        attempts = "\n".join(
            f"- {e['id']}  {_scores(e.get('objectives'))}  {'kept' if e['admitted'] else 'rejected'}  {e['name']}"
            for e in self.history.log(all=True, limit=20)
        )
        run_dir = run_directory(self.storage)
        lineage = (
            "\nRead more with `evokit log --all`, `evokit show ID [PATH]`, `evokit diff A [B]` and "
            "`evokit grep PATTERN`.\n"
            if run_dir
            else ""
        )
        try:
            current = await asyncio.to_thread(self.project.read, self.workspace)
            work = diff_genomes(best.genome, current, best.id, "workspace")
        except Exception:
            work = ""
        objective = getattr(self.archive, "objective", "score")
        direction = "minimize" if getattr(self.archive, "minimize", False) else "maximize"
        prompt = SUPERVISOR.format(
            stalled=self.stalled,
            objective=f"{direction} `{objective}`",
            instructions=f"\n\nThe task:\n{self.instructions}" if self.instructions else "",
            best=describe_individual(best),
            attempts=attempts or "(none)",
            drafts=", ".join(self.recent_drafts) or "(none)",
            work=truncate(work, 6000) or "(no changes)",
            lineage=lineage,
        )
        log.info("supervisor: reviewing the run")
        agent, key = self._metered(self.agent)
        try:
            with tempfile.TemporaryDirectory(prefix="evokit-supervisor-") as directory:
                env = run_access(self.agent, Path(directory), run_dir)
                result = await agent.run(Path(directory), prompt, env=env)
        except asyncio.CancelledError:
            history.note(self.storage, "supervisor-failed", "stopped: the run is over", usage=self._settle(key, {}))
            raise
        usage = self._settle(key, result.usage)
        if result.error or not result.summary.strip():
            history.note(self.storage, "supervisor-failed", result.error or "no reply", usage=usage)
            return
        self.directions = result.summary.strip()
        history.note(self.storage, "redirect", self.directions, usage=usage)

    # the session

    def _prompt(self, readable: bool) -> str:
        return PROMPT.format(
            editable=", ".join(self.project.editable),
            best=describe_individual(self.archive.best()),
            lineage=LINEAGE if readable else "",
            instructions=self.instructions or "",
        ).rstrip() + "\n"

    def _continue(self) -> str:
        best = self.archive.best()
        directions, self.directions = self.directions, None
        text = f"{best.id} ({_scores(best.evaluation.objectives)})"
        return CONTINUE.format(best=text, directions=f"\n\nA supervisor reviewed the run; follow one of these:\n{directions}" if directions else "")

    async def go(self) -> None:
        self.workspace.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(clear_directory, self.workspace)
        await asyncio.to_thread(self.project.write, self.archive.best().genome, self.workspace)
        commands = AgentCommands({"evokit-eval": self.draft, "evokit-commit": self.commit})
        try:
            async with commands:
                scripts = {name: str(path) for name, path in commands.scripts.items()}
                env = run_access(self.agent, self.workspace, run_directory(self.storage))
                prompt = self._prompt(bool(env))
                env = {**env, "EVOKIT_DRAFT_EVAL": scripts["evokit-eval"], "EVOKIT_COMMIT": scripts["evokit-commit"]}
                while not self.stop_reason():
                    self.commits = 0
                    result = await self._session(prompt, env)
                    self._record(result)
                    prompt = self._continue()
        finally:
            if self.supervisor is not None and not self.supervisor.done():
                self.supervisor.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self.supervisor

    async def _session(self, prompt: str, env: dict) -> AgentResult:
        """One agent run (a start or a resume). When the run's limits are reached mid-session, the
        agent gets STOP_GRACE seconds to finish, and is then stopped."""
        agent, key = self._metered(self.agent)
        task = asyncio.create_task(agent.run(self.workspace, prompt, session=self.session, env=env))
        stopped = asyncio.create_task(self.stopped.wait())
        await asyncio.wait({task, stopped}, return_when=asyncio.FIRST_COMPLETED)
        stopped.cancel()
        result = None
        if not task.done():
            try:
                result = await asyncio.wait_for(asyncio.shield(task), STOP_GRACE)
            except asyncio.TimeoutError:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
                result = AgentResult("", error="stopped: the run is over")
        result = result or task.result()
        return dataclasses.replace(result, usage=self._settle(key, result.usage))

    def _record(self, result: AgentResult) -> None:
        """The session's cost and transcript; a session that committed nothing counts as a failure."""
        if result.session:
            self.session = result.session
        usage = merge_usage(result.usage, self.drafts.usage, {"draft_evals": self.drafts.count})
        self.drafts.usage, self.drafts.count = {}, 0
        folder = f"sessions/{new_id()}"
        if result.transcript:
            self.storage.write(f"{folder}/agent.jsonl", result.transcript.encode())
        summary = truncate(result.summary, 2000) or ""
        history.note(self.storage, "session", f"{self.commits} commits; logs: {folder}/\n{summary}".strip(), usage=usage)
        if self.stopped.is_set():
            return
        if self.commits == 0:
            reason = f" ({result.error})" if result.error else ""
            history.note(self.storage, "failed", f"the agent session ended without a commit{reason}")


def _scores(objectives) -> str:
    return ", ".join(f"{k}={v:.6g}" for k, v in (objectives or {}).items()) or "failed"
