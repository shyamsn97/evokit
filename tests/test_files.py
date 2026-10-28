"""The optional files package: Files genomes over a project directory."""

import asyncio
import dataclasses
import json
import os
import subprocess
import sys
import time

import pytest

import evokit as ek
from evokit.agents import Agent, ClaudeCode, CommandAgent, ReplAgent, Turn
from evokit.agents.tools import ShellTools, limited, parse_argv
from evokit.agents.repl import CONFIRM
from evokit.agents.repl_server import Repl
from evokit.files import (
    Files,
    Project,
    ScriptEvaluator,
    apply_diff,
    apply_rewrite,
    parse_metrics,
    with_lines,
)
from evokit.experiment import agent_tools
from evokit.interface.commands import checkout_individual, evaluate_directory
from evokit.operators import FunctionModel, diff
from evokit.storage import decode_genome, encode_genome
from evokit.storages import MemoryStorage
from evokit.util import glob_match


def files_task(seed, genome=("main.py",)) -> ek.Task:
    """The seed folder as a task: `genome` evolves, scored by its evaluate.py."""
    project = Project(seed, genome=list(genome))
    return ek.Task(project.seed_genome(), ScriptEvaluator(project, "python evaluate.py"), files=project)


def climbed(seed, make, **limits) -> ek.Experiment:
    """An experiment on the seed folder, each child made from the best by `make(parent, task)`."""
    task = files_task(seed)
    experiment = task.create_experiment(**limits)

    async def go():
        while not experiment.limits.reached():
            (parent,) = experiment.archive.sample(1)
            if child := await experiment.create(make(parent, task)):
                await experiment.submit(child)

    asyncio.run(go())
    return experiment


def researched(seed, agent, genome=("main.py",), **limits) -> ek.Experiment:
    """`recipes.autoresearch` on the seed folder with `agent`; the experiment, reopened to read."""
    dir = seed.parent / f"run-{ek.util.new_id()}"
    asyncio.run(ek.recipes.autoresearch(files_task(seed, genome), agent, dir=dir, **limits))
    return ek.open_experiment(dir)


def prompt(agent, text, cwd, tools=(), resume=None, env=None, budget=None, system=None, output=None) -> Turn:
    """One prompt in a new session."""

    async def go():
        async with agent.session(cwd, tools, resume, env, system=system) as session:
            return await session.prompt(text, budget, output=output)

    return asyncio.run(go())


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


def bump_diff(prompt):
    x = float(prompt.split("X = ")[1].split("\n")[0])
    return f"main.py\n<<<<<<< SEARCH\nX = {x}\n=======\nX = {x + 1.0}\n>>>>>>> REPLACE\n"


def test_diff_hill_climb(seed):
    e = climbed(seed, lambda parent, task: diff(FunctionModel(bump_diff), parent, task=task), max_evaluations=5)
    assert e.best().genome.text("main.py") == "X = 3.0\n"
    # 4.0 fails the evaluator, so it is rejected (twice: the best stays 3.0)
    assert [bool(i.evaluation.objectives) for i in e.evaluated()] == [True, True, True, False, False]


def test_diff_inspirations_and_evolve_blocks():
    prompts = []

    def model(prompt):
        prompts.append(prompt)
        return "main.py\n<<<<<<< SEARCH\nHEADER = 1\n=======\nHEADER = 2\n>>>>>>> REPLACE\n"

    text = "HEADER = 1\n# EVOLVE-BLOCK-START\nX = 1\n# EVOLVE-BLOCK-END\n"
    parent = ek.Individual.new(Files.from_dict({"main.py": text})).with_evaluation(ek.Evaluation({"score": 1.0}))
    other = ek.Individual.new(Files.from_dict({"main.py": "X = 9\n"})).with_evaluation(ek.Evaluation({"score": 9.0}))
    with pytest.raises(ek.OperatorError, match="outside its EVOLVE-BLOCK"):
        asyncio.run(diff(FunctionModel(model), parent, [other]))
    assert "for inspiration" in prompts[0] and "X = 9" in prompts[0] and "EVOLVE-BLOCK-END markers" in prompts[0]

    inside = FunctionModel(lambda p: "<<<<<<< SEARCH\nX = 1\n=======\nX = 2\n>>>>>>> REPLACE\n")
    child, _ = asyncio.run(diff(inside, parent))
    assert child.genome.text("main.py") == text.replace("X = 1", "X = 2") and child.parents == (parent.id,)


def test_checkout_and_evaluate_a_directory(seed, tmp_path):
    e = files_task(seed).create_experiment()
    seed_individual = e.best()
    work = tmp_path / "work"
    checkout_individual(e, seed_individual.id, str(work))
    with pytest.raises(ek.UsageError, match="not empty"):
        checkout_individual(e, seed_individual.id, str(work))
    (work / "main.py").write_text("X = 2.5\n")
    individual = evaluate_directory(e, str(work), [seed_individual.id], name="closer", description="X from 1 to 2.5")
    assert individual.evaluation.objectives == {"score": -0.5} and individual.name == "closer"
    assert individual.description == "X from 1 to 2.5" and individual.info["admitted"]
    assert "+X = 2.5" in e.diff(individual)
    assert e.show(individual, "main.py") == "X = 2.5\n"
    with pytest.raises(ek.UsageError, match="no individual"):
        evaluate_directory(e, str(work), ["nope"])


def test_an_agent_makes_children_by_editing_their_parent(seed, fake_agent):
    e = researched(seed, CommandAgent(fake_agent), max_evaluations=3)
    best = e.best()
    assert best.genome.text("main.py") == "X = 2.0\n" and best.info["operator"] == "agent"
    assert "draft={'score': -1.0" in best.name and best.description == "bumped X"
    used = e.usage()
    assert used["draft_evals"] == 2 and e.counts()["evaluated"] == 3 and e.counts()["turn"] == 2
    assert used["cost_usd"] == pytest.approx(0.02)


def test_experiment_tools_run_from_the_shell_and_stop_with_the_session(seed, tmp_path):
    e = files_task(seed).create_experiment()
    checkout = e.task.checkout(e.task.seed, tmp_path / "work")
    session = Agent().session(checkout, agent_tools(e, checkout, submitting=False))
    assert "submit" not in session.tools and {"log", "score", "remember"} <= set(session.tools)

    async def go():
        async with ShellTools(session) as shell:
            env = {**os.environ, **shell.env}

            def evokit(*args):
                return asyncio.to_thread(subprocess.run, ["evokit", *args], capture_output=True, text=True, env=env)

            scored, logged = await evokit("score"), await evokit("log", "--all")
            session.stop("$5 spent")
            return scored, logged, await evokit("show", e.best().id, "main.py")

    scored, logged, shown = asyncio.run(go())
    assert scored.returncode == 0 and json.loads(scored.stdout)["objectives"] == {"score": -2.0}
    assert [entry["id"] for entry in json.loads(logged.stdout)] == [e.best().id]
    assert shown.returncode == 1 and json.loads(shown.stderr)["stop"] == "$5 spent"
    assert e.usage()["draft_evals"] == 1


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
subprocess.run(["evokit", "finish", "--title", f"told={told} seen={len(log)}", "--description", "read the experiment first"])
print(json.dumps({"summary": "done"}))
"""


def test_an_agent_can_read_the_experiment(seed, tmp_path):
    script = tmp_path / "reading_agent.py"
    script.write_text(READING_AGENT)
    e = researched(seed, CommandAgent([sys.executable, str(script)]), genome=["**"], max_evaluations=2)
    child = e.evaluated()[-1]
    assert child.name == "told=True seen=1" and child.description == "read the experiment first"


def test_an_agent_that_edits_the_environment_is_rejected(seed, fake_agent, monkeypatch):
    monkeypatch.setenv("TAMPER", "1")
    e = researched(seed, CommandAgent(fake_agent), max_evaluations=2, max_cost=0.01)
    failed = e.notes("failed")
    assert failed and "evaluate.py" in failed[0]["text"]
    assert e.usage()["cost_usd"] == pytest.approx(0.01)


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
    result = prompt(agent, "improve it", tmp_path)
    assert time.monotonic() - start < 10
    assert result.error is None and result.text == "edited main.py" and result.session == "s1"
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
    assert "--max-budget-usd 0.5" in prompt(agent, "hi", tmp_path, budget=0.5).text
    assert "--max-budget-usd 2.0" in prompt(agent, "hi", tmp_path, budget=3.0).text
    assert "--max-budget-usd 2.0" in prompt(agent, "hi", tmp_path).text


@dataclasses.dataclass
class Named:
    """What it is."""

    title: str
    tags: list[str] = dataclasses.field(default_factory=list)


ECHO = 'import json, sys; print(json.dumps({"summary": sys.stdin.read()}))'

FINISHING = """
import json, subprocess, sys
prompt = sys.stdin.read()
if "{when}" in prompt:
    subprocess.run(["evokit", "finish", "named", "--tag", "a", "--tag", "b"])
    late = subprocess.run(["evokit", "stats"], capture_output=True, text=True)
    print(late.stderr.strip(), file=sys.stderr)
print(json.dumps({{"summary": "ok", "cost_usd": 0.01}}))
"""


def test_the_system_prompt_reaches_each_harness(tmp_path):
    binary = fake_claude(tmp_path, 'print(json.dumps({"type": "result", "result": " ".join(sys.argv[1:])}))')
    assert "--append-system-prompt BE TERSE" in prompt(ClaudeCode(binary=binary), "hi", tmp_path, system="BE TERSE").text

    model = ScriptedModel(["Done.", "Done."])
    prompt(ReplAgent(model, system="base"), "hi", tmp_path, system="BE TERSE")
    assert model.seen[0][0] == {"role": "system", "content": "base\n\nBE TERSE"}

    async def twice():
        async with CommandAgent([sys.executable, "-c", ECHO]).session(tmp_path, system="BE TERSE") as session:
            return (await session.prompt("first")).text, (await session.prompt("second")).text

    first, second = asyncio.run(twice())
    assert first == "BE TERSE\n\nfirst" and second == "second"


def test_output_comes_from_the_finish_tool(tmp_path):
    script = tmp_path / "finishing.py"
    script.write_text(FINISHING.format(when="evokit finish"))
    turn = prompt(CommandAgent([sys.executable, str(script)]), "name it", tmp_path, output=Named)
    assert turn.output == Named("named", ["a", "b"]) and turn.error is None and turn.calls == ("finish",)
    assert "you already called finish" in turn.trace

    model = ScriptedModel([repl("finish(title='from python')"), "Done.", "Done."])
    turn = prompt(ReplAgent(model), "name it", tmp_path, output=Named)
    assert turn.output == Named("from python") and "`finish(title: str, tags: list[str] = [])`" in model.seen[0][1]["content"]


def test_an_agent_that_does_not_finish_is_asked_once_more(tmp_path):
    script = tmp_path / "late.py"
    script.write_text(FINISHING.format(when="You have not called"))
    turn = prompt(CommandAgent([sys.executable, str(script)]), "name it", tmp_path, output=Named)
    assert turn.output == Named("named", ["a", "b"]) and turn.usage["cost_usd"] == pytest.approx(0.02)

    turn = prompt(CommandAgent([sys.executable, "-c", ECHO]), "name it", tmp_path, output=Named)
    assert turn.output is None and turn.error == "the agent ended its turn without calling finish"
    assert turn.text.startswith("You have not called `evokit finish TITLE [--tag TAG ...]` yet")


def test_claude_code_allowed_tools_deny_everything_else(tmp_path):
    from evokit.agents import NO_SHELL

    binary = fake_claude(tmp_path, 'print(json.dumps({"type": "result", "result": " ".join(sys.argv[1:])}))')
    argv = prompt(ClaudeCode(binary=binary, allowed_tools=NO_SHELL), "hi", tmp_path).text
    assert "--permission-mode dontAsk --allowedTools Read,Edit,Write,Glob,Grep,Bash(evokit *)" in argv
    assert "bypassPermissions" not in argv
    assert "--permission-mode bypassPermissions" in prompt(ClaudeCode(binary=binary), "hi", tmp_path).text


def test_claude_code_login_auth_drops_the_api_key(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-bad")
    binary = fake_claude(tmp_path, 'print(json.dumps({"type": "result", "result": str("ANTHROPIC_API_KEY" in os.environ)}))')
    assert prompt(ClaudeCode(binary=binary), "hi", tmp_path).text == "True"
    assert prompt(ClaudeCode(binary=binary, auth="login"), "hi", tmp_path).text == "False"


class EditsThenFails(Agent):
    def __init__(self, edit: bool):
        self.edit = edit

    async def _run(self, cwd, prompt, session, env, budget):
        if self.edit:
            (cwd / "main.py").write_text("X = 2.0\n")
        return Turn("", {"cost_usd": 0.01}, error="reached the turn limit", transcript='{"type": "system"}\n')


def test_an_agent_that_fails_after_editing_still_makes_a_child(seed):
    e = researched(seed, EditsThenFails(edit=True), max_evaluations=2)
    child = e.evaluated()[-1]
    assert child.genome.text("main.py") == "X = 2.0\n" and child.info["agent_error"] == "reached the turn limit"
    (turn,) = e.notes("turn")
    assert turn["error"] == "reached the turn limit" and e.storage.read(turn["transcript"]) == b'{"type": "system"}\n'

    e = researched(seed, EditsThenFails(edit=False), max_evaluations=2, max_cost=0.01)
    assert "reached the turn limit" in e.last_note("failed")
    assert e.storage.read(e.notes("turn")[0]["transcript"]) == b'{"type": "system"}\n'


DRIVER = """
import asyncio, sys
from pathlib import Path
from evokit.agents import ClaudeCode

async def main():
    async with ClaudeCode(binary=sys.argv[1]).session(Path(sys.argv[2])) as session:
        await session.prompt("hi")

asyncio.run(main())
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
if any("evokit finish" in a for a in sys.argv):
    subprocess.run(["evokit", "finish", "--title", "set X to 2"], capture_output=True)
for event in events:
    print(json.dumps(event), flush=True)
"""


def test_claude_code_logs_each_step_and_keeps_its_transcript(seed, tmp_path, caplog):
    binary = fake_claude(tmp_path, WORKING_CLAUDE)
    caplog.set_level("INFO", logger="evokit")
    e = researched(seed, ClaudeCode(binary=binary), max_evaluations=2)
    assert e.evaluated()[-1].genome.text("main.py") == "X = 2.0\n" and e.evaluated()[-1].name == "set X to 2"
    assert "agent: 'Trying a bigger X.'" in caplog.messages and "agent: Edit main.py" in caplog.messages
    transcript = e.storage.read(e.notes("turn")[0]["transcript"]).decode().splitlines()
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
    result = prompt(ReplAgent(model), "make X 2", tmp_path, env={"EVOKIT_DIR": "/the/run"})
    assert result.error is None and result.text == "Set X to 2\n\nIt was 1."
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
    again = prompt(agent, "anything else?", tmp_path, resume=result.session)
    assert again.session == result.session and len(resumed.seen[0]) == 12
    assert resumed.seen[0][-1]["content"].endswith("anything else?") and "restarted" in resumed.seen[0][-1]["content"]


def test_repl_agent_calls_tools_as_python_functions(tmp_path):
    def add(a: int, b: int = 1) -> int:
        """Add two numbers."""
        return a + b

    model = ScriptedModel([repl("print(add(2, b=3))\nadd()"), "Added\n\nDone.", "Added\n\nDone."])
    result = prompt(ReplAgent(model), "add them", tmp_path, tools=[add])
    assert result.error is None and "- `add(a: int, b: int = 1) -> int`: Add two numbers." in model.seen[0][1]["content"]
    output = model.seen[1][-1]["content"]
    assert "5\n" in output and "RuntimeError: missing a required argument: 'a' (add(a: int, b: int = 1) -> int)" in output


def test_repl_agent_asks_again_when_a_reply_only_says_what_it_will_do(tmp_path):
    model = ScriptedModel(["I'll write a.txt now.", repl("open('a.txt', 'w').write('1')"), "Wrote a.txt\n\nDone.", "Wrote a.txt\n\nDone."])
    result = prompt(ReplAgent(model), "write a.txt", tmp_path)
    assert result.error is None and result.text == "Wrote a.txt\n\nDone." and (tmp_path / "a.txt").read_text() == "1"
    assert model.seen[1][-1]["content"] == CONFIRM and model.seen[3][-1]["content"] == CONFIRM


def test_repl_agent_restarts_a_stuck_block_and_respects_limits(tmp_path):
    model = ScriptedModel([repl("y = 1\nimport time; time.sleep(30)"), repl("print('y' in dir())"), "done", "done"])
    started = time.monotonic()
    result = prompt(ReplAgent(model, block_timeout=1), "go", tmp_path)
    assert time.monotonic() - started < 10 and result.error is None
    assert "timed out after 1s" in model.seen[1][-1]["content"] and "False" in model.seen[2][-1]["content"]

    endless = ScriptedModel([repl("pass")] * 10, cost=0.5)
    result = prompt(ReplAgent(endless), "go", tmp_path, budget=1.0)
    assert result.error == "spent its $1 budget" and result.usage["cost_usd"] == 1.0
    result = prompt(ReplAgent(ScriptedModel([repl("pass")] * 10), max_steps=3), "go", tmp_path)
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
