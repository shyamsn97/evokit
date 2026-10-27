"""The optional files package: Files genomes over a project directory."""

import asyncio
import functools
import json
import os
import subprocess
import sys
import time

import pytest

import evokit as ek
from evokit.history import History
from evokit.archives import BestArchive
from evokit.agents import AgentResult, ClaudeCode, CodingAgent, CodingAgentOperator, CommandAgent, ReplAgent
from evokit.agents.tools import ToolServer, agent_tools, limited, parse_argv
from evokit.agents.repl import CONFIRM
from evokit.agents.repl_server import Repl
from evokit.files import (
    DiffOperator,
    FileCommands,
    Files,
    Project,
    ScriptEvaluator,
    apply_diff,
    apply_rewrite,
    parse_metrics,
    with_lines,
)
from evokit.interface.run import RunCommands
from evokit.operators import FunctionModel
from evokit.storage import decode_genome, encode_genome
from evokit.storages import DirectoryStorage, MemoryStorage
from evokit.strategies import Greedy
from evokit.util import glob_match


@functools.cache
def history_of(storage) -> History:
    """One History per storage, as a process keeps one per run."""
    return History(storage)


def test_glob():
    assert glob_match("a/b/c.py", ["**/*.py"]) and glob_match("c.py", ["**/*.py"])
    assert not glob_match("a/c.txt", ["*.txt"]) and glob_match("evaluate.py", ["evaluate*"])


def test_files_value_and_json():
    files = Files.from_dict({"b.txt": "two", "a.txt": b"one"})
    assert files.paths() == ["a.txt", "b.txt"] and files.text("b.txt") == "two"
    changed = files.with_files({"a.txt": None, "c.txt": "three"})
    assert changed.paths() == ["b.txt", "c.txt"] and files == Files.from_dict({"a.txt": "one", "b.txt": "two"})
    blobs = {}

    def put(data):
        blobs[str(len(blobs))] = data
        return str(len(blobs) - 1)

    assert Files.from_json(files.to_json(put), blobs.__getitem__) == files
    storage = MemoryStorage()
    data = encode_genome(files, storage)
    assert data["$type"] == "evokit.files.tree:Files" and decode_genome(data, storage) == files
    with pytest.raises(ek.UsageError):
        encode_genome(object(), storage)


def test_diff_and_rewrite():
    files = Files.from_dict({"main.py": "X = 1.0\nY = 2\n"})
    diff = "main.py\n<<<<<<< SEARCH\nX = 1.0\n=======\nX = 2.0\n>>>>>>> REPLACE\n"
    assert apply_diff(files, diff).text("main.py") == "X = 2.0\nY = 2\n"
    assert apply_rewrite(files, "```python\nX = 5\n```").text("main.py") == "X = 5\n"
    with pytest.raises(ValueError):
        apply_diff(files, "nothing here")


def test_parse_metrics():
    assert parse_metrics('noise\n{"score": 1.5}\n') == {"score": 1.5}
    assert parse_metrics("score: 2\nvalid = true\n") == {"score": 2.0, "valid": True}
    assert parse_metrics("nothing") is None


def test_project_split_and_protection(seed, tmp_path):
    project = Project(seed, genome=["main.py"])
    assert project.seed_genome().paths() == ["main.py"]
    assert sorted(p.name for p in project.environment.iterdir()) == ["data.txt", "evaluate.py"]
    work = tmp_path / "work"
    project.write(project.seed_genome().with_files({"main.py": "X = 2.0\n"}), work)
    assert project.read(work).text("main.py") == "X = 2.0\n"
    (work / "evaluate.py").write_text("cheat")
    with pytest.raises(ek.EnvironmentModified, match="evaluate.py"):
        project.read(work)
    (seed / "data.txt").write_text("changed after the snapshot\n")
    assert (project.environment / "data.txt").read_text() == "environment file\n"


def test_script_evaluator(seed, tmp_path):
    project = Project(seed, genome=["main.py"])
    evaluate = ScriptEvaluator(project, "python evaluate.py")
    individual = ek.Individual.new(project.seed_genome())
    assert asyncio.run(evaluate(individual)).objectives == {"score": -2.0}
    failed = asyncio.run(evaluate.evaluate_files(Files.from_dict({"main.py": "X = 3.5\n"})))
    assert failed.objectives == {} and "above 3" in failed.feedback
    broken = asyncio.run(evaluate.evaluate_files(Files.from_dict({"main.py": "X = 1\nBROKEN = True\n"})))
    assert broken.objectives == {} and "BROKEN" in broken.feedback
    lines = asyncio.run(with_lines(evaluate)(individual))
    assert lines.objectives == {"score": -2.0, "lines": 1.0}
    assert asyncio.run(with_lines(evaluate)(ek.Individual.new(Files.from_dict({"main.py": "X = 4\n"})))).objectives == {}


def test_script_evaluator_returns_workdir_files_as_logs(tmp_path):
    seed = tmp_path / "seed"
    seed.mkdir()
    (seed / "main.py").write_text("X = 1\n")
    (seed / "evaluate.py").write_text(
        "import os, pathlib\npathlib.Path(os.environ['EVOKIT_WORKDIR'], 'log.txt').write_text('hello')\nprint('score: 1')\n"
    )
    project = Project(seed, genome=["main.py"])
    evaluation = asyncio.run(ScriptEvaluator(project, "python evaluate.py")(ek.Individual.new(project.seed_genome())))
    assert evaluation.objectives == {"score": 1.0} and evaluation.logs == {"log.txt": b"hello"}


class FileRun:
    """A Files run in memory: the project, its eval function, a seeded archive, and a loop."""

    def __init__(self, seed, operator_for=None):
        self.project = Project(seed, genome=["main.py"])
        self.eval_fn = ScriptEvaluator(self.project, "python evaluate.py")
        self.operator = operator_for(self.project) if operator_for else None
        self.storage = MemoryStorage()
        self.archive = BestArchive(storage=self.storage)
        asyncio.run(self._seed())

    async def _seed(self):
        seeds = [ek.Individual.new(self.project.seed_genome())]
        await ek.tell(history_of(self.storage), self.archive, seeds, await asyncio.gather(*map(ek.as_eval_fn(self.eval_fn), seeds)))

    def run(self, evaluations, max_cost=None):
        async def go():
            while history_of(self.storage).counts()["evaluated"] < evaluations:
                if max_cost is not None and history_of(self.storage).usage()["cost_usd"] >= max_cost:
                    return
                asked = await ek.ask(history_of(self.storage), self.archive, Greedy(), self.operator)
                scores = await asyncio.gather(*map(ek.as_eval_fn(self.eval_fn), asked.individuals))
                await ek.tell(history_of(self.storage), self.archive, asked.individuals, scores)

        asyncio.run(go())


def bump_diff(prompt):
    x = float(prompt.split("X = ")[1].split("\n")[0])
    return f"main.py\n<<<<<<< SEARCH\nX = {x}\n=======\nX = {x + 1.0}\n>>>>>>> REPLACE\n"


def test_diff_operator_hill_climb(seed):
    run = FileRun(seed, lambda p: DiffOperator(FunctionModel(bump_diff), p))
    run.run(5)
    assert run.archive.best().genome.text("main.py") == "X = 3.0\n"
    # 4.0 fails the evaluator, so it is rejected (twice: Greedy keeps trying from 3.0)
    assert [bool(i.evaluation.objectives) for i in history_of(run.storage).evaluated()] == [True, True, True, False, False]


def test_diff_operator_inspirations_and_evolve_blocks():
    prompts = []

    def model(prompt):
        prompts.append(prompt)
        return "main.py\n<<<<<<< SEARCH\nHEADER = 1\n=======\nHEADER = 2\n>>>>>>> REPLACE\n"

    text = "HEADER = 1\n# EVOLVE-BLOCK-START\nX = 1\n# EVOLVE-BLOCK-END\n"
    parent = ek.Individual.new(Files.from_dict({"main.py": text})).with_evaluation(ek.Evaluation({"score": 1.0}))
    other = ek.Individual.new(Files.from_dict({"main.py": "X = 9\n"})).with_evaluation(ek.Evaluation({"score": 9.0}))
    operator = DiffOperator(FunctionModel(model))
    with pytest.raises(ek.OperatorError, match="outside its EVOLVE-BLOCK"):
        asyncio.run(operator.create(ek.Selection((parent,), (other,)), BestArchive()))
    assert "for inspiration" in prompts[0] and "X = 9" in prompts[0] and "EVOLVE-BLOCK-END markers" in prompts[0]

    inside = DiffOperator(FunctionModel(lambda p: "<<<<<<< SEARCH\nX = 1\n=======\nX = 2\n>>>>>>> REPLACE\n"))
    (child,) = asyncio.run(inside.create(ek.Selection((parent,)), BestArchive())).children
    assert child.genome.text("main.py") == text.replace("X = 1", "X = 2") and child.parents == (parent.id,)


def test_file_commands(seed, tmp_path):
    run = FileRun(seed)
    commands = FileCommands(run.project)
    runs = RunCommands(run.storage, run.archive, Greedy(), None, run.eval_fn)
    seed_individual = run.archive.best()
    work = tmp_path / "work"
    commands.checkout_individual(seed_individual, str(work))
    with pytest.raises(ek.UsageError, match="not empty"):
        commands.checkout_individual(seed_individual, str(work))
    (work / "main.py").write_text("X = 2.5\n")
    individual = commands.evaluate_directory(runs, str(work), [seed_individual.id], "closer", "X from 1 to 2.5")
    assert individual.evaluation.objectives == {"score": -0.5} and individual.name == "closer"
    assert individual.description == "X from 1 to 2.5" and individual.info["admitted"]
    assert "+X = 2.5" in runs.history.diff(individual)
    assert runs.history.show(individual, "main.py") == "X = 2.5\n"
    edited = commands.edit_individual(runs, individual, {"main.py": "X = 3.0\n"})
    assert edited.parents == (individual.id,) and edited.info["admitted"]
    with pytest.raises(ek.UsageError, match="not genome files"):
        commands.edit_individual(runs, individual, {"evaluate.py": "cheat"})
    with pytest.raises(ek.UsageError, match="no individual"):
        commands.evaluate_directory(runs, str(work), ["nope"])


def test_coding_agent_operator(seed, fake_agent):
    run = FileRun(seed)
    run.operator = CodingAgentOperator(
        CommandAgent(fake_agent), run.project, eval_fn="python evaluate.py", history=history_of(run.storage), max_draft_evals=3
    )
    run.run(3)
    best = run.archive.best()
    assert best.genome.text("main.py") == "X = 2.0\n" and best.info["operator"] == "coding-agent"
    assert "draft={'score': -1.0}" in best.name
    used = history_of(run.storage).usage()
    assert used["draft_evals"] == 2 and history_of(run.storage).counts()["evaluated"] == 3
    assert used["cost_usd"] == pytest.approx(0.02)


def test_agent_tools_run_from_the_shell_and_stop_with_the_run(seed):
    run = FileRun(seed)
    history = history_of(run.storage)
    tools = agent_tools(run.archive, history, run.eval_fn, project=run.project, workspace=seed)
    reasons = [None, None, None, None, "$5 spent"]

    async def go():
        async with ToolServer(tools, stop=lambda: reasons.pop(0)) as server:
            env = {**os.environ, **server.env}

            def evokit(*args):
                return asyncio.to_thread(subprocess.run, ["evokit", *args], capture_output=True, text=True, env=env)

            return await evokit("score"), await evokit("log", "--all"), await evokit("show", run.archive.best().id, "main.py")

    scored, logged, shown = asyncio.run(go())
    assert scored.returncode == 0 and json.loads(scored.stdout)["objectives"] == {"score": -2.0}
    assert [e["id"] for e in json.loads(logged.stdout)] == [run.archive.best().id]
    assert shown.returncode == 1 and json.loads(shown.stderr)["stop"] == "$5 spent"
    assert history.usage()["draft_evals"] == 1


def test_tools_take_their_arguments_from_the_signature():
    def show(id: str, path: str | None = None, all: bool = False, limit: int = 20, tags: list[str] | None = None):
        """Show it."""

    assert parse_argv(show, ["abc", "main.py", "--all", "--limit=3", "--tags", "a", "--tags", "b"]) == {
        "id": "abc", "path": "main.py", "all": True, "limit": 3, "tags": ["a", "b"]
    }
    with pytest.raises(ek.UsageError, match="missing id"):
        parse_argv(show, [])

    async def score():
        return 1

    once = limited(score, 1)
    assert asyncio.run(once()) == 1
    with pytest.raises(ek.UsageError, match="1 times"):
        asyncio.run(once())


READING_AGENT = """
import json, subprocess, sys
told = "evokit log" in sys.stdin.read()
log = json.loads(subprocess.run(["evokit", "log", "--all"], capture_output=True, text=True).stdout)
open("main.py", "w").write("X = 2.0\\n")
print(json.dumps({"summary": f"told={told} seen={len(log)}\\n\\nread the run first"}))
"""


def test_coding_agent_can_read_the_run(seed, tmp_path):
    script = tmp_path / "reading_agent.py"
    script.write_text(READING_AGENT)
    project = Project(seed, genome=["**"])
    eval_fn = ScriptEvaluator(project, "python evaluate.py")
    storage = DirectoryStorage(tmp_path / "run")
    archive = BestArchive(storage=storage)
    operator = CodingAgentOperator(CommandAgent([sys.executable, str(script)]), project, history=history_of(storage))

    async def go():
        seeds = [ek.Individual.new(project.seed_genome())]
        await ek.tell(history_of(storage), archive, seeds, await asyncio.gather(*map(ek.as_eval_fn(eval_fn), seeds)))
        return await ek.ask(history_of(storage), archive, Greedy(), operator)

    (child,) = asyncio.run(go()).individuals
    assert child.name == "told=True seen=1" and child.description == "read the run first"


def test_coding_agent_tampering_rejected(seed, fake_agent, monkeypatch):
    monkeypatch.setenv("TAMPER", "1")
    run = FileRun(seed, lambda p: CodingAgentOperator(CommandAgent(fake_agent), p))
    run.run(2, max_cost=0.01)
    failed = [r for r in history_of(run.storage).records() if r["kind"] == "note" and r["note"] == "failed"]
    assert failed and "evaluate.py" in failed[0]["text"]
    assert history_of(run.storage).usage()["cost_usd"] == pytest.approx(0.01)


LINGERING_CLAUDE = """#!{python}
import json, sys, time
print(json.dumps({{"type": "system", "subtype": "init"}}), flush=True)
print(json.dumps({{"type": "result", "result": "edited main.py", "session_id": "s1", "num_turns": 3,
                  "total_cost_usd": 0.25, "is_error": False, "usage": {{"input_tokens": 10, "output_tokens": 5}}}}), flush=True)
time.sleep(600)
"""


def test_claude_code_returns_once_it_prints_its_result(tmp_path):
    binary = tmp_path / "claude"
    binary.write_text(LINGERING_CLAUDE.format(python=sys.executable))
    binary.chmod(0o755)
    agent = ClaudeCode(binary=str(binary))
    agent.exit_grace = 0.5
    start = time.monotonic()
    result = asyncio.run(agent.run(tmp_path, "improve it"))
    assert time.monotonic() - start < 10
    assert result.error is None and result.summary == "edited main.py" and result.session == "s1"
    assert result.usage == {"tokens": 15, "turns": 3, "cost_usd": 0.25}


def fake_claude(tmp_path, body: str):
    binary = tmp_path / "claude"
    binary.write_text(f"#!{sys.executable}\nimport json, os, subprocess, sys, time\n{body}\n")
    binary.chmod(0o755)
    return str(binary)


def test_claude_code_check_reports_what_claude_said(tmp_path):
    binary = fake_claude(
        tmp_path, 'print(json.dumps({"type": "result", "is_error": True, "result": "Invalid API key · Fix external API key"}))'
    )
    with pytest.raises(ek.UsageError, match="Invalid API key"):
        asyncio.run(ClaudeCode(binary=binary).check())


def test_claude_code_budget_is_passed_and_only_lowered(tmp_path):
    binary = fake_claude(tmp_path, 'print(json.dumps({"type": "result", "result": " ".join(sys.argv[1:])}))')
    agent = ClaudeCode(binary=binary, max_budget_usd=2.0)
    assert "--max-budget-usd 0.5" in asyncio.run(agent.with_budget(0.5).run(tmp_path, "hi")).summary
    assert agent.with_budget(3.0).max_budget_usd == 2.0 and agent.max_budget_usd == 2.0
    assert agent.with_budget(None) is agent


METERED_CLAUDE = """
model = "claude-test-meter"
def emit(event): print(json.dumps(event), flush=True)
def stream(event): emit({"type": "stream_event", "event": event})
if "Reply with OK." in sys.argv:
    usage = {"inputTokens": 10, "outputTokens": 2, "cacheCreationInputTokens": 8, "cacheReadInputTokens": 100}
    emit({"type": "result", "result": "OK", "modelUsage": {model: {**usage, "costUSD": 40 * 2e-6}}})
    sys.exit()
emit({"type": "system", "argv": sys.argv})
stream({"type": "message_start", "message": {"id": "m1", "model": model, "usage": {"input_tokens": 100, "output_tokens": 1}}})
stream({"type": "content_block_delta", "delta": {"text": "hi"}})
stream({"type": "message_delta", "usage": {"input_tokens": 100, "output_tokens": 50, "cache_read_input_tokens": 1000}})
emit({"type": "result", "result": "done", "total_cost_usd": 0.0011})
"""


def test_claude_code_reports_its_cost_as_it_goes(tmp_path):
    agent = ClaudeCode(binary=fake_claude(tmp_path, METERED_CLAUDE))
    asyncio.run(agent.check())  # teaches the model's rate: $40 x 2e-6 for 40 input-token units
    costs = []
    result = asyncio.run(agent.with_meter(costs.append).run(tmp_path, "improve it"))
    assert costs == [pytest.approx(105 * 2e-6), pytest.approx(450 * 2e-6)]  # input + 5 x output + 0.1 x cache reads
    assert "--include-partial-messages" in result.transcript and "stream_event" not in result.transcript
    assert result.usage["cost_usd"] == 0.0011
    assert "--include-partial-messages" not in " ".join(agent.argv("hi", None))


def test_claude_code_login_auth_drops_the_api_key(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-bad")
    binary = fake_claude(tmp_path, 'print(json.dumps({"type": "result", "result": str("ANTHROPIC_API_KEY" in os.environ)}))')
    assert asyncio.run(ClaudeCode(binary=binary).run(tmp_path, "hi")).summary == "True"
    assert asyncio.run(ClaudeCode(binary=binary, auth="login").run(tmp_path, "hi")).summary == "False"


class EditsThenFails(CodingAgent):
    def __init__(self, edit: bool):
        self.edit = edit

    async def _run(self, directory, prompt, session=None, env=None):
        if self.edit:
            (directory / "main.py").write_text("X = 2.0\n")
        return AgentResult("", {"cost_usd": 0.01}, error="reached the turn limit", transcript='{"type": "system"}\n')


def test_an_agent_that_fails_after_editing_still_makes_a_child(seed):
    run = FileRun(seed, lambda p: CodingAgentOperator(EditsThenFails(edit=True), p))
    run.run(2)
    child = history_of(run.storage).evaluated()[-1]
    assert child.genome.text("main.py") == "X = 2.0\n" and child.info["agent_error"] == "reached the turn limit"

    assert run.storage.read(f"artifacts/{child.id}/agent.jsonl") == b'{"type": "system"}\n'

    run = FileRun(seed, lambda p: CodingAgentOperator(EditsThenFails(edit=False), p))
    run.run(2, max_cost=0.01)
    failed = history_of(run.storage).last_note("failed")
    assert "reached the turn limit" in failed
    folder = failed.split("logs: ")[1].strip()
    assert run.storage.read(folder + "agent.jsonl") == b'{"type": "system"}\n'


DRIVER = """
import asyncio, sys
from pathlib import Path
from evokit.agents import ClaudeCode
asyncio.run(ClaudeCode(binary=sys.argv[1]).run(Path(sys.argv[2]), "hi"))
"""


def test_sigterm_kills_the_agents_process_group(tmp_path):
    import os
    import signal
    import subprocess

    pid_file = tmp_path / "grandchild.pid"
    binary = fake_claude(
        tmp_path,
        f"child = subprocess.Popen(['sleep', '600'])\nopen({str(pid_file)!r}, 'w').write(str(child.pid))\ntime.sleep(600)",
    )
    driver = subprocess.Popen([sys.executable, "-c", DRIVER, binary, str(tmp_path)])
    deadline = time.monotonic() + 20
    while not (pid_file.exists() and pid_file.read_text()):
        assert time.monotonic() < deadline and driver.poll() is None
        time.sleep(0.05)
    grandchild = int(pid_file.read_text())
    driver.send_signal(signal.SIGTERM)
    assert driver.wait(10) == -signal.SIGTERM
    deadline = time.monotonic() + 5
    while True:
        try:
            os.kill(grandchild, 0)
        except ProcessLookupError:
            break
        assert time.monotonic() < deadline, "the agent's child outlived evokit"
        time.sleep(0.05)


WORKING_CLAUDE = """
events = [
    {"type": "assistant", "message": {"content": [{"type": "text", "text": "Trying a\\nbigger X."}]}},
    {"type": "assistant", "message": {"content": [{"type": "tool_use", "name": "Edit", "input": {"file_path": "main.py"}}]}},
    {"type": "result", "result": "set X to 2", "num_turns": 2},
]
open("main.py", "w").write("X = 2.0\\n")
for event in events:
    print(json.dumps(event), flush=True)
"""


def test_claude_code_logs_each_step_and_keeps_its_transcript(seed, tmp_path, caplog):
    binary = fake_claude(tmp_path, WORKING_CLAUDE)
    caplog.set_level("INFO", logger="evokit")
    run = FileRun(seed, lambda p: CodingAgentOperator(ClaudeCode(binary=binary), p))
    run.run(2)
    child = history_of(run.storage).evaluated()[-1]
    assert child.genome.text("main.py") == "X = 2.0\n"
    assert "agent: 'Trying a bigger X.'" in caplog.messages and "agent: Edit main.py" in caplog.messages
    transcript = run.storage.read(f"artifacts/{child.id}/agent.jsonl").decode().splitlines()
    assert len(transcript) == 3 and '"set X to 2"' in transcript[-1]


class ScriptedModel(ek.operators.Model):
    """Replies in order, recording the conversation each call saw; each call costs `cost`."""

    def __init__(self, replies, cost=0.1):
        self.replies, self.cost, self.seen = list(replies), cost, []

    async def generate(self, prompt):
        raise AssertionError("ReplAgent uses chat")

    async def chat(self, messages):
        self.seen.append([dict(m) for m in messages])
        return ek.operators.Completion(self.replies.pop(0), {"cost_usd": self.cost, "tokens": 10})


def repl(code):
    return f"Next.\n```repl\n{code}\n```\n"


def test_repl_runs_blocks_in_one_namespace():
    repl = Repl()
    assert repl.run("x = 41\nprint('hi')") == "hi\n"
    assert repl.run("x + 1") == "42\n"
    assert repl.run("import sys; print('err', file=sys.stderr)") == "err\n"
    error = repl.run("def f():\n    return 1 / 0\nf()")
    assert "ZeroDivisionError" in error and "<block 4>" in error and "return 1 / 0" in error and "repl.py" not in error
    assert repl.run("print((").startswith("  File") and repl.run("x") == "41\n"


def test_repl_agent_keeps_state_and_sees_output(tmp_path):
    model = ScriptedModel([
        repl("x = 41\nopen('main.py', 'w').write('X = 2\\n')"),
        repl("x + 1") + repl("import os\nsh('echo from-shell'); print(os.environ['EVOKIT_DIR'])"),
        repl("def f():\n    return 1 / 0\nf()") + repl("print(("),
        "Set X to 2\n\nIt was 1.",
        "Set X to 2\n\nIt was 1.",
    ])
    result = asyncio.run(ReplAgent(model).run(tmp_path, "make X 2", env={"EVOKIT_DIR": "/the/run"}))
    assert result.error is None and result.summary == "Set X to 2\n\nIt was 1."
    assert (tmp_path / "main.py").read_text() == "X = 2\n"
    assert result.usage == {"cost_usd": pytest.approx(0.5), "tokens": 50, "turns": 5}
    outputs = [m[-1]["content"] for m in model.seen[1:]]
    assert outputs[0] == "Output of block 1:\n6\n"  # the value of write()
    assert "Output of block 1:\n42" in outputs[1] and "from-shell\n/the/run" in outputs[1]
    assert "ZeroDivisionError" in outputs[2] and "<block 5>" in outputs[2] and "return 1 / 0" in outputs[2]
    assert "SyntaxError" in outputs[2] and "<string>" not in outputs[2]
    assert model.seen[0][0]["role"] == "system" and model.seen[0][1] == {"role": "user", "content": "make X 2"}
    assert model.seen[4][-1]["content"] == CONFIRM and len(result.transcript.splitlines()) == 11

    resumed = ScriptedModel(["Nothing to do\n\nDone."] * 2)
    agent = ReplAgent(resumed)
    agent.sessions = {result.session: json.loads("[" + ",".join(result.transcript.splitlines()) + "]")}
    again = asyncio.run(agent.run(tmp_path, "anything else?", session=result.session))
    assert again.session == result.session and len(resumed.seen[0]) == 12
    assert resumed.seen[0][-1]["content"].endswith("anything else?") and "restarted" in resumed.seen[0][-1]["content"]


def test_repl_agent_calls_tools_as_python_functions(tmp_path):
    def add(a: int, b: int = 1) -> int:
        """Add two numbers."""
        return a + b

    model = ScriptedModel([repl("print(add(2, b=3))\nadd()"), "Added\n\nDone.", "Added\n\nDone."])
    result = asyncio.run(ReplAgent(model).run(tmp_path, "add them", tools=[add], stop=lambda: None))
    assert result.error is None and "- `add(a: int, b: int = 1) -> int`: Add two numbers." in model.seen[0][1]["content"]
    output = model.seen[1][-1]["content"]
    assert "5\n" in output and "RuntimeError: missing a required argument: 'a' (add(a: int, b: int = 1) -> int)" in output


def test_repl_agent_asks_again_when_a_reply_only_says_what_it_will_do(tmp_path):
    model = ScriptedModel(["I'll write a.txt now.", repl("open('a.txt', 'w').write('1')"), "Wrote a.txt\n\nDone.", "Wrote a.txt\n\nDone."])
    result = asyncio.run(ReplAgent(model).run(tmp_path, "write a.txt"))
    assert result.error is None and result.summary == "Wrote a.txt\n\nDone." and (tmp_path / "a.txt").read_text() == "1"
    assert model.seen[1][-1]["content"] == CONFIRM and model.seen[3][-1]["content"] == CONFIRM


def test_repl_agent_restarts_a_stuck_block_and_respects_limits(tmp_path):
    model = ScriptedModel([repl("y = 1\nimport time; time.sleep(30)"), repl("print('y' in dir())"), "done", "done"])
    started = time.monotonic()
    result = asyncio.run(ReplAgent(model, block_timeout=1).run(tmp_path, "go"))
    assert time.monotonic() - started < 10 and result.error is None
    assert "timed out after 1s" in model.seen[1][-1]["content"] and "False" in model.seen[2][-1]["content"]

    costs = []
    endless = ScriptedModel([repl("pass")] * 10, cost=0.5)
    result = asyncio.run(ReplAgent(endless).with_budget(1.0).with_meter(costs.append).run(tmp_path, "go"))
    assert result.error == "spent its $1 budget" and costs == [0.5, 1.0]
    result = asyncio.run(ReplAgent(ScriptedModel([repl("pass")] * 10), max_steps=3).run(tmp_path, "go"))
    assert result.error == "no final reply after 3 steps" and result.usage["turns"] == 3


def test_openai_model_chat_sends_the_conversation(monkeypatch):
    import http.server
    import threading

    bodies = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            bodies.append(json.loads(self.rfile.read(int(self.headers["content-length"]))))
            reply = {"choices": [{"message": {"content": "hi"}}], "usage": {"prompt_tokens": 3, "completion_tokens": 2, "cost": 0.01}}
            data = json.dumps(reply).encode()
            self.send_response(200)
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        model = ek.operators.OpenAIModel("m", base_url=f"http://127.0.0.1:{server.server_port}/v1", system="be brief")
        turns = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}, {"role": "user", "content": "c"}]
        completion = asyncio.run(model.chat(turns))
    finally:
        server.shutdown()
    assert completion.text == "hi" and completion.usage["cost_usd"] == 0.01 and completion.usage["tokens"] == 5
    assert bodies[0]["messages"] == [{"role": "system", "content": "be brief"}, *turns]
