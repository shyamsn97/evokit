# Running real coding agents: what broke, and the fixes

**Status: implemented.** All seven changes are in; each section names its test.

Decisions: a workspace the agent changed before failing is evaluated as a child, with the
error in `info["agent_error"]` (problem 4); `ClaudeCode(auth="env")` stays the default, and
`auth="login"` removes `ANTHROPIC_API_KEY` (problem 1).

The first real AVO run on circle packing (Claude Code as the operator, `examples/circle_packing`)
lost half an hour. Each failure below is a gap in the library, not in the run: the same thing
will happen to anyone who runs a real agent. Each section has what happened, the code that let
it happen, and the change.

Two things that went wrong were the operator's (mine), not the library's, and are listed at the
end so they are not confused with the rest.

## 1. A broken agent is found after the loop starts

**What happened.** The shell had an `ANTHROPIC_API_KEY` that Claude Code rejected
(`Invalid API key`), so every step failed. Later, the `sonnet` alias in Claude Code 2.1.22
resolved to a model the account cannot use (`404 model: claude-sonnet-4-5-20250929`). Both are
setup errors, known before the first step, and both surfaced as a stream of failed steps.

**The code.** Nothing checks the agent before the loop. `ClaudeCode` hands the whole
environment to `claude`, so a stray key silently replaces the Claude Code login:

```python
# evokit/files/agents.py (before)
proc = await asyncio.create_subprocess_exec(*argv, cwd=cwd, env={**os.environ, **env}, ...)
```

**The change (done).** Agents and operators get `check()`, called once before a loop starts. It is a
no-op by default; `ClaudeCode.check()` makes one tiny call with the exact settings the run will
use and turns any error into a usage error with Claude's own message.

```python
# evokit/files/agents.py
class CodingAgent(ABC):
    async def check(self) -> None:
        """Raise UsageError if the agent cannot run. Called once, before a loop starts."""


class ClaudeCode(CodingAgent):
    async def check(self) -> None:
        probe = copy.copy(self)
        probe.max_turns = 1
        with tempfile.TemporaryDirectory() as directory:
            result = await probe.run(Path(directory), "Reply with OK.")
        if result.error:
            raise UsageError(f"Claude Code cannot run: {result.error}",
                             hint="check `claude -p hi` in this shell, the model, and auth=")
```

```python
# evokit/operator.py
class Operator(ABC):
    async def check(self) -> None:
        """Raise UsageError if the operator cannot create anything (bad credentials, unknown model)."""


# evokit/files/coding_agent.py
class CodingAgentOperator(Operator):
    async def check(self) -> None:
        await self.agent.check()
```

`RunCommands.run` and every recipe call `await operator.check()` before their loop; the LLM
operators check their model the same way (one "Reply with OK." request), and `Choice` checks each
of its operators. Tests: `test_a_failing_check_stops_the_loop_before_it_starts`,
`test_claude_code_check_reports_what_claude_said`, `test_claude_code_login_auth_drops_the_api_key`.

Credentials become a choice, not an accident of the shell:

```python
ClaudeCode(auth="env")    # default: the environment as it is (today's behavior)
ClaudeCode(auth="login")  # the Claude Code login: ANTHROPIC_API_KEY is removed from the agent's environment

# in ClaudeCode._exec
env = {**os.environ, **env}
if self.auth == "login":
    env.pop("ANTHROPIC_API_KEY", None)
```

## 2. A loop that cannot stop

**What happened.** Every step failed in about a second and cost nothing. A failed step is not
an evaluation and adds no cost, so neither `--max-evaluations 6` nor `--max-cost 2` could ever
be reached: 124 failed calls in two minutes, and it would have gone on forever.

**The code.** Each loop has its own stop condition, and none of them counts failures:

```python
# evokit/recipes/avo.py (before)
while history.usage(storage)["cost_usd"] < max_cost and (
    max_evaluations is None or history.counts(storage)["evaluated"] < max_evaluations
):
    asked = await evokit.ask(storage, archive, strategy, operator)
    ...

# evokit/interface/run.py (before), the same condition written again
def done() -> bool:
    return (max_evaluations is not None and history.counts(self.storage)["evaluated"] >= max_evaluations) or (
        max_cost is not None and history.usage(self.storage)["cost_usd"] >= max_cost
    )
```

**The change (done).** One `Limits` value that every loop uses, with a third limit: operator failures
in a row. It is read from the history, so it holds across restarts, and the reason is recorded
when the loop stops.

```python
# evokit/history.py: the replayed state counts failures since the last creation
def failures_in_a_row(storage: Storage) -> int:
    """Operator failures ("failed" notes) since the last child was created."""
    return _state(storage).failures_in_a_row


# evokit/loop.py
@dataclass(frozen=True)
class Limits:
    max_evaluations: int | None = None
    max_cost: float | None = None
    max_failures: int | None = 10        # operator failures in a row

    def reached(self, storage: Storage) -> str | None:
        """Why the run should stop, or None."""
        if self.max_evaluations is not None and history.counts(storage)["evaluated"] >= self.max_evaluations:
            return f"{self.max_evaluations} evaluations"
        if self.max_cost is not None and history.usage(storage)["cost_usd"] >= self.max_cost:
            return f"${self.max_cost} spent"
        if self.max_failures is not None and history.failures_in_a_row(storage) >= self.max_failures:
            return f"{self.max_failures} operator failures in a row; the last: {history.last_note(storage, 'failed')}"
        return None
```

```python
# evokit/recipes/avo.py
limits = Limits(max_evaluations, max_cost, max_failures)
while not limits.reached(storage):
    asked = await evokit.ask(storage, archive, strategy, operator)
    ...
history.note(storage, "stop", limits.reached(storage) or "nothing left to select")
```

`evokit run` and every recipe except `ga` (a fixed number of generations) and `hyperagents` (it
runs `dgm`) take `max_failures` (`--max-failures` on the command line, default 10).
Tests: `test_max_failures_stops_a_loop_that_only_fails`,
`test_failures_in_a_row_reset_when_a_child_is_created`.

A child the archive rejects is not a failure: it was created and evaluated. Only an operator that
could not produce a child counts.

## 3. Claude Code finishes but does not exit (fixed)

**What happened.** Claude Code wrote its summary at 23:05:47 (a draft scored 2.53, up from
1.60) and then sat at 0% CPU with its process alive. evokit waited for the process to exit, which
would have taken the full one-hour timeout.

**The code (before).** The answer was whatever was on stdout when the process exited:

```python
out, err = await asyncio.wait_for(proc.communicate(stdin), timeout)
data = json.loads(out.strip().splitlines()[-1])
```

**The change (done).** The contract with Claude Code is its final `result` event, not its exit.
With `--output-format stream-json --verbose` that event is the last line it prints; evokit reads
events as they arrive, takes the result, and gives the process `exit_grace` seconds before
killing it. `test_claude_code_returns_once_it_prints_its_result` covers a `claude` that prints
its result and then never exits.

```python
async def result_event() -> dict:
    async for line in proc.stdout:
        event = json.loads(line)
        if event.get("type") == "result":
            return event
    return {}

data = await asyncio.wait_for(result_event(), self.timeout)
try:
    await asyncio.wait_for(proc.wait(), self.exit_grace if data else None)
except asyncio.TimeoutError:
    await _kill(proc)              # it answered; it just did not exit
```

## 4. A failed agent throws its work away

**What happened.** When the hung Claude Code was killed, it exited with an error, and the
operator discarded the workspace, including the `main.py` whose draft had scored 2.53.

**The code.** An agent error raises before the workspace is read:

```python
# evokit/files/coding_agent.py (before)
if result.error:
    raise OperatorError(f"agent failed: {result.error}...", usage)
genome = await asyncio.to_thread(self.project.read, workspace)
```

**The change (done).** The evaluator, not the agent's exit status, says whether a
program is good. If the agent failed but changed the genome, the workspace becomes a child with
the error in its info; only an agent that failed without changing anything is an operator failure.

```python
genome = await asyncio.to_thread(self.project.read, workspace)
if result.error and (not individuals or genome == base):
    raise OperatorError(f"agent failed: {result.error}...", usage)
info = {"operator": self.name, "summary": truncate(result.summary, 4000), "session": result.session}
if result.error:
    info["agent_error"] = truncate(result.error, 4000)
return Offspring((Individual.new(genome, parents, info),), usage)
```

`ClaudeCode` also reports a run that hit its turn limit (`subtype: "error_max_turns"`) as an
error, so such a run keeps its edits the same way.
Test: `test_an_agent_that_fails_after_editing_still_makes_a_child`.

## 5. Two processes can drive one run

**What happened.** No harm this time (see the end), but nothing stops it: two `python -m
algorithms.avo --dir runs/avo-0` processes both select from, write to, and spend on the same run.

**The code.** The persistent agent's workspace is a fixed path in the run, and every step empties it:

```python
# evokit/files/coding_agent.py (before)
if self.persistent:
    workspace, owned = self.project.scratch / "agent", False
    workspace.mkdir(parents=True, exist_ok=True)
    _clear(workspace)          # a second process clears the first one's workspace mid-edit
```

The fixed path is right (Claude Code keys `--resume` sessions by working directory, so a
continued run must use the same folder); what is missing is the rule that only one process
drives a run.

**The change (done).** A driver lock on the storage, taken by every loop and released when it stops.
A second driver fails at once. Read-only commands (`status`, `list-individuals`, ...) and an
outside agent's `evaluate-directory` do not take it.

```python
# evokit/storage.py
class Storage(ABC):
    @abstractmethod
    def lock(self, name: str) -> ContextManager[None]:
        """Exclusive across processes; raises RunBusy at once if another holder has it."""


# evokit/storages/directory.py
@contextmanager
def lock(self, name: str):
    fd = os.open(self._path(f"run/{name}.lock"), os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        holder = os.read(fd, 64).decode() or "another process"
        os.close(fd)
        raise RunBusy(f"this run is being driven by {holder}") from None
    os.ftruncate(fd, 0)
    os.write(fd, f"pid {os.getpid()}".encode())
    try:
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
```

```python
# every loop
with storage.lock("driver"):
    while not (reason := limits.reached(storage)):
        ...
```

`MemoryStorage` keeps the held names in a set; an S3 backend would use a lease object
with an expiry. Test: `test_one_driver_at_a_time`.

## 6. Killing evokit leaves the agent running

**What happened.** Agents are started in their own process group (`start_new_session=True`),
so a timeout can kill the agent and everything it started. The same flag means that when the
Python process dies from SIGTERM, the agent is not signalled: it keeps running, editing, and
spending.

**The code.** Children are killed only on timeout or task cancellation, and SIGTERM cancels nothing:

```python
# evokit/files/agents.py (before)
except (asyncio.TimeoutError, asyncio.CancelledError):
    await _kill(proc)
```

**The change (done).** One registry of the process groups evokit started (agents and
evaluators). It kills them on SIGTERM or SIGHUP and when Python exits, which covers Ctrl-C: in
the CLI the loop runs on a background thread that Ctrl-C does not cancel.

```python
# evokit/util.py
def spawned(proc) -> None:
    """Kill this child's process group if evokit exits or is terminated (SIGTERM, SIGHUP)."""
    with _children_lock:
        _children.add(proc.pid)
        ...                          # the first child registers kill_children with atexit
    handle_termination()

def handle_termination() -> None:
    """Signal handlers can only be installed from the main thread; other calls do nothing."""
    ...
    for sig in (signal.SIGTERM, signal.SIGHUP):
        previous = signal.getsignal(sig)
        if previous is not signal.SIG_IGN:  # e.g. SIGHUP under nohup: the run is meant to survive it
            signal.signal(sig, functools.partial(_terminate, previous=previous))

def _terminate(signum, frame, previous) -> None:
    kill_children()
    if callable(previous):
        previous(signum, frame)
        return
    signal.signal(signum, signal.SIG_DFL)
    os.kill(os.getpid(), signum)
```

`ScriptEvaluator.run`, `ClaudeCode._exec` and `_exec` call `spawned(proc)` after starting a
child and `reaped(proc)` when it ends; `background.run` installs the handlers from the main thread
before it hands work to the loop thread. Test: `test_sigterm_kills_the_agents_process_group`.

## 7. Nothing to see while it runs

**What happened.** One AVO step is a whole agent session with up to 10 draft evaluations,
5 to 10 minutes of silence. Finding out whether it was working, stuck, or failing meant reading
Claude Code's own session files.

**The code.** The loop logs nothing; the recipe prints a report when it returns.

**The change (done).** Each event is logged as it happens, on the `evokit` logger (the CLI and
the example scripts send it to stderr). `history.note` logs every note (failures, redirects, the
stop reason), `ask` logs each child it creates, `tell` each score, and the draft server each draft.
`ClaudeCode` logs one line per tool call or message as its stream-json events arrive:

```text
evokit: agent: Read main.py
evokit: agent: 'Switching to a hexagonal layout for the outer ring.'
evokit: agent: Edit main.py
evokit: agent: Bash $EVOKIT_DRAFT_EVAL
evokit: draft 1/10: score=2.06
evokit: draft 2/10: score=2.31
evokit: created 7c1d0e2a91 from fb9f294506
evokit: evaluated 7c1d0e2a91: score=2.53 (added)
evokit: stop: $2 spent
```

The whole session is kept too: `AgentResult.transcript` (Claude Code's events, including a
timed-out session's) goes back as `Offspring.logs` / `OperatorError.logs`, and `ask` stores it at
`artifacts/<child id>/agent.jsonl`, or at `failed/<id>/agent.jsonl` for a step that produced
nothing (the failure note names the folder). Test:
`test_claude_code_logs_each_step_and_keeps_its_transcript`.

## Not library problems

- **The vanished run directory.** `rm -rf runs/avo-0` and the command starting the new run were
  sent in parallel; the `rm` landed two seconds after the new run created the folder.
- **Killing the wrong process.** Stopping a run killed its shell wrapper, not the Python process,
  so the old run kept going. Problem 6 makes this safe when the Python process itself is killed.
