"""The optional interface: CLI, MCP, skill, over a task folder and a run directory."""

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import evokit as ek
from evokit.interface.cli import main
from evokit.interface.setup import open_setup
from evokit.interface.skill import SKILL_FILE, export_skill, render_skill

ROOT = Path(__file__).resolve().parents[1]

CONFIG = """
task:
  genome: [main.py]
  evaluate: python evaluate.py
  timeout: 1m
run:
"""

EVO = """
import evokit as ek
from evokit.archives import BestArchive, PopulationArchive
from evokit.files import DiffOperator, ScriptEvaluator
from evokit.operators import FunctionModel
from evokit.strategies import Greedy

class Probe(ScriptEvaluator):
    @ek.command
    def describe(self, verbose: bool = False) -> dict:
        "Describe the evaluator."
        return {"kind": "probe", "verbose": verbose}

def driver(task, storage):
    return PopulationArchive(size=4, storage=storage), Greedy(), Probe(task.project, "python evaluate.py")

def bump(prompt):
    x = float(prompt.split("X = ")[1].split(chr(10))[0])
    return f"main.py{chr(10)}<<<<<<< SEARCH{chr(10)}X = {x}{chr(10)}======={chr(10)}X = {x + 0.5}{chr(10)}>>>>>>> REPLACE{chr(10)}"

def auto(task, storage):
    return BestArchive(storage=storage), Greedy(), DiffOperator(FunctionModel(bump), task.project, instructions=task.instructions)

def no_storage(task, storage):
    return BestArchive(), Greedy()
"""


def make_task(seed: Path, run: str = "") -> Path:
    (seed / "TASK.md").write_text("# Goal\n\nRaise X to 3 without passing it.\n")
    (seed / "config.yaml").write_text(CONFIG + run)
    (seed / "evo.py").write_text(EVO)
    return seed


@pytest.fixture
def task(seed):
    return make_task(seed)


def cli(*args, cwd):
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(ROOT), os.environ.get("PYTHONPATH")]))}
    proc = subprocess.run([sys.executable, "-m", "evokit.interface.cli", *args], cwd=cwd, capture_output=True, text=True, env=env)
    out = json.loads(proc.stdout) if proc.stdout.strip() else None
    last = proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else ""
    err = json.loads(last) if last.startswith("{") else None
    return proc.returncode, out, err


def test_init_writes_a_task_that_runs_as_it_is(tmp_path):
    code, out, _ = cli("init", "toy", cwd=tmp_path)
    assert code == 0 and out["files"] == ["TASK.md", "config.yaml", "evaluate.py", "main.py"]
    code, status, _ = cli("status", "--task", "toy", "--dir", "toy/runs/0", cwd=tmp_path)
    assert code == 0 and status["members"] == 1 and status["best"]["evaluation"]["objectives"] == {"score": -2.0}
    assert status["best"]["name"] == "seed" and status["best"]["info"] == {}
    code, _, err = cli("init", "toy", cwd=tmp_path)
    assert code == 2 and "not empty" in err["error"]


def test_agent_is_the_operator_cli(task):
    code, status, _ = cli("status", "--task", ".", cwd=task)
    assert code == 0 and status["members"] == 1 and status["operator"] is None and status["counts"]["evaluated"] == 1
    seed = status["best"]
    assert seed["parents"] == [] and seed["genome"]["files"] == ["main.py"] and seed["evaluation"]["objectives"] == {"score": -2.0}
    seed_id = seed["id"]
    code, selected, _ = cli("select-individuals", cwd=task)
    assert code == 0 and [s["parents"][0]["id"] for s in selected] == [seed_id] and selected[0]["inspirations"] == []

    work = task / "work"
    code, _, _ = cli("checkout-individual", seed_id, str(work), cwd=task)
    assert code == 0 and sorted(os.listdir(work)) == ["TASK.md", "config.yaml", "data.txt", "evaluate.py", "evo.py", "main.py"]
    code, _, err = cli("checkout-individual", seed_id, str(work), cwd=task)
    assert code == 2 and "not empty" in err["error"]

    (work / "main.py").write_text("X = 2.0\n")
    (work / "evaluate.py").write_text((task / "evaluate.py").read_text() + "# cheat\n")
    code, _, err = cli("evaluate-directory", str(work), "--parent", seed_id, cwd=task)
    assert code == 2 and err["code"] == "environment_modified" and "evaluate.py" in err["error"]
    (work / "evaluate.py").write_text((task / "evaluate.py").read_text())

    # commands work from any directory: the run records its task folder
    code, individual, _ = cli(
        "evaluate-directory", str(work), "--parent", seed_id, "--name", "X to 2", "--description", "closer to 3",
        "--dir", str(task / ".evokit"), cwd=work,
    )
    assert code == 0 and individual["evaluation"]["objectives"]["score"] == -1.0 and individual["parents"] == [seed_id]
    assert individual["info"]["admitted"] is True and individual["name"] == "X to 2"

    # reading the run, like git
    code, diff, _ = cli("diff", individual["id"], cwd=task)
    assert code == 0 and "+X = 2.0" in diff and f"--- {seed_id}/main.py" in diff
    code, got, _ = cli("show", individual["id"], cwd=task)
    assert code == 0 and got["description"] == "closer to 3" and got["files"] == ["main.py"] and got["member"]
    code, text, _ = cli("show", individual["id"], "main.py", cwd=task)
    assert text == "X = 2.0\n"
    code, log, _ = cli("log", cwd=task)
    assert [(e["id"], e["name"]) for e in log] == [(individual["id"], "X to 2"), (seed_id, "seed")]
    (work / "main.py").write_text("X = 0.5\n")
    code, worse, _ = cli("evaluate-directory", str(work), "--parent", individual["id"], "--name", "try 0.5", cwd=task)
    assert worse["info"]["admitted"] is False
    code, log, _ = cli("log", "--all", cwd=task)
    assert [e["id"] for e in log] == [worse["id"], individual["id"], seed_id] and not log[0]["admitted"]
    code, found, _ = cli("grep", "X = 0\\.5", cwd=task)
    assert [f["id"] for f in found] == [worse["id"]] and found[0]["matches"] == [{"path": "main.py", "line": 1, "text": "X = 0.5"}]

    code, status, _ = cli("status", cwd=task)
    assert status["best"]["id"] == individual["id"] and status["counts"]["evaluated"] == 3 and status["counts"]["created"] == 2
    code, events, _ = cli("list-events", cwd=task)
    assert [e["kind"] for e in events] == ["evaluated", "created", "evaluated", "created", "evaluated"] and events[2]["added"]
    code, _, err = cli("run", cwd=task)
    assert code == 2 and "no operator" in err["error"]


def test_cli_run_with_operator(task):
    (task / "config.yaml").write_text(CONFIG + "  factory: evo:auto\n")
    code, best, _ = cli("run", "--task", ".", "--max-evaluations", "5", "--workers", "1", cwd=task)
    assert code == 0 and best["evaluation"]["objectives"]["score"] == -0.0
    code, events, _ = cli("list-events", cwd=task)
    assert [e["kind"] for e in events].count("evaluated") == 5 and events[-1] == {**events[-1], "kind": "note", "note": "stop"}


def test_factory_component_commands_tools_and_mcp(task):
    (task / "config.yaml").write_text(CONFIG + "  factory: evo:driver\n")
    code, out, _ = cli("evaluator", "describe", "--verbose", "--task", ".", cwd=task)
    assert code == 0 and out == {"kind": "probe", "verbose": True}
    code, (seed,), _ = cli("list-individuals", cwd=task)
    directory = task / "new"
    directory.mkdir()
    (directory / "main.py").write_text("X = 2.5\n")
    code, individual, _ = cli("evaluate-directory", str(directory), "--parent", seed["id"], "--name", "manual", cwd=task)
    assert code == 0 and individual["parents"] == [seed["id"]] and individual["evaluation"]["objectives"]["score"] == -0.5
    code, members, _ = cli("list-individuals", cwd=task)
    assert [m["evaluation"]["objectives"]["score"] for m in members] == [-0.5, -2.0]

    setup = open_setup(task / ".evokit")
    tools = ek.tools(*setup.objects)
    assert {"evaluate_genome", "edit_individual", "select_individuals", "run", "evaluator_describe"} <= set(tools)
    best = tools["best_individual"]()
    individual = tools["edit_individual"](best, {"main.py": "X = 3.0\n"})
    assert individual.info["admitted"] and individual.evaluation.objectives["score"] == 0.0

    from evokit.interface.mcp_server import build_server

    server = build_server(str(task / ".evokit"))
    names = {t.name for t in asyncio.run(server.list_tools())}
    assert {"evaluate_directory", "checkout_individual", "list_individuals", "select_individuals", "status", "evaluator_describe"} <= names
    assert {"log", "show", "diff", "grep"} <= names
    assert "edit_individual" not in names and "run" not in names


def test_a_failing_seed_records_nothing_and_can_be_fixed(task):
    (task / "main.py").write_text("X = 4.0\n")
    code, _, err = cli("status", "--task", ".", cwd=task)
    assert code == 2 and "seed failed" in err["error"] and "above 3" in err["error"]
    code, _, err = cli("status", cwd=task)
    assert code == 2 and "--task" in err["hint"]
    (task / "main.py").write_text("X = 1.0\n")
    code, status, _ = cli("status", "--task", ".", cwd=task)
    assert code == 0 and status["members"] == 1 and status["counts"]["evaluated"] == 1


def test_a_run_belongs_to_one_task(task, tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    make_task(other)
    (other / "main.py").write_text("X = 1.0\n")
    (other / "evaluate.py").write_text((task / "evaluate.py").read_text())
    assert cli("status", "--task", ".", cwd=task)[0] == 0
    code, _, err = cli("status", "--task", str(other), cwd=task)
    assert code == 2 and "belongs to the task" in err["error"]


def test_factory_needs_the_run_storage(task):
    (task / "config.yaml").write_text(CONFIG + "  factory: evo:no_storage\n")
    code, _, err = cli("status", "--task", ".", cwd=task)
    assert code == 2 and "storage=storage" in err["hint"]


def test_the_shipped_skill_matches_the_commands():
    assert SKILL_FILE.read_text() == render_skill(None, "claude"), (
        "regenerate it: evokit skill export --stdout > evokit/skills/evokit/SKILL.md"
    )


def test_skill_export(task, tmp_path):
    text = render_skill(None, "claude")
    assert text.startswith("---\nname: evokit")
    assert "`evokit init NAME`" in text and "--task TASK_DIR" in text
    assert "`evokit select-individuals [--n N]`" in text
    assert "`evokit checkout-individual ID DIR`" in text
    assert "`evokit evaluate-directory DIR [--parent ID ...] [--name NAME] [--description DESCRIPTION]`" in text
    assert "`evokit run [--workers WORKERS] [--max-evaluations MAX_EVALUATIONS]" in text
    assert "`evokit log [ID] [--all] [--limit LIMIT]`" in text and "`evokit show ID [PATH]`" in text
    assert "`evokit diff A [B]`" in text and "`evokit grep PATTERN [--limit LIMIT]`" in text
    assert "You are an operator's agent" in text

    mcp = render_skill(open_setup(task / ".evokit", task), "mcp")
    assert "operator: you (the strategy, Greedy, selects individuals)" in mcp and "archive: BestArchive" in mcp
    assert "evolving files: main.py" in mcp and f"task: `{task}`" in mcp
    assert "> Raise X to 3 without passing it." in mcp

    path = export_skill("claude", tmp_path)
    assert path == tmp_path / ".claude" / "skills" / "evokit" / "SKILL.md" and path.read_text() == text
    agents = tmp_path / "AGENTS.md"
    agents.write_text("# Project\n")
    export_skill("codex", tmp_path)
    export_skill("codex", tmp_path)
    content = agents.read_text()
    assert content.startswith("# Project\n") and content.count("<!-- evokit:begin -->") == 1


def test_cli_errors(tmp_path, capsys):
    assert main(["status", "--dir", str(tmp_path / "missing")]) == 2
    err = json.loads(capsys.readouterr().err)
    assert err["code"] == "usage" and "--task" in err["hint"]
    assert main(["status", "--task", str(tmp_path), "--dir", str(tmp_path / "run")]) == 2
    assert "config.yaml" in json.loads(capsys.readouterr().err)["error"]
    assert main(["nonsense"]) == 2
