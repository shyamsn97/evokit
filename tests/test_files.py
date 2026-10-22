"""The optional files package: Files genomes over a project directory."""

import asyncio

import pytest

import evokit as ek
from evokit import history
from evokit.archives import BestArchive
from evokit.files import (
    CodingAgentOperator,
    CommandAgent,
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
from evokit.storages import MemoryStorage
from evokit.strategies import Greedy
from evokit.util import glob_match


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
    assert asyncio.run(evaluate(individual, tmp_path)).objectives == {"score": -2.0}
    failed = asyncio.run(evaluate.evaluate_files(Files.from_dict({"main.py": "X = 3.5\n"})))
    assert failed.objectives == {} and "above 3" in failed.feedback
    broken = asyncio.run(evaluate.evaluate_files(Files.from_dict({"main.py": "X = 1\nBROKEN = True\n"})))
    assert broken.objectives == {} and "BROKEN" in broken.feedback
    lines = asyncio.run(with_lines(evaluate)(individual, tmp_path))
    assert lines.objectives == {"score": -2.0, "lines": 1.0}
    assert asyncio.run(with_lines(evaluate)(ek.Individual.new(Files.from_dict({"main.py": "X = 4\n"})), tmp_path)).objectives == {}


def test_script_evaluator_workdir(tmp_path):
    seed = tmp_path / "seed"
    seed.mkdir()
    (seed / "main.py").write_text("X = 1\n")
    (seed / "evaluate.py").write_text(
        "import os, pathlib\npathlib.Path(os.environ['EVOKIT_WORKDIR'], 'log.txt').write_text('hello')\nprint('score: 1')\n"
    )
    project = Project(seed, genome=["main.py"])
    storage = MemoryStorage()
    (individual,) = asyncio.run(ek.evaluate(storage, [ek.Individual.new(project.seed_genome())], eval_fn=ScriptEvaluator(project, "python evaluate.py")))
    assert individual.objective("score") == 1.0 and storage.read(f"artifacts/{individual.id}/log.txt") == b"hello"


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
        await ek.tell(self.storage, self.archive, await ek.evaluate(self.storage, seeds, eval_fn=self.eval_fn))

    def run(self, evaluations, max_cost=None):
        async def go():
            while history.counts(self.storage)["evaluated"] < evaluations:
                if max_cost is not None and history.usage(self.storage)["cost_usd"] >= max_cost:
                    return
                asked = await ek.ask(self.storage, self.archive, Greedy(), self.operator)
                evaluated = await ek.evaluate(self.storage, asked.individuals, eval_fn=self.eval_fn)
                await ek.tell(self.storage, self.archive, evaluated)

        asyncio.run(go())


def bump_diff(prompt):
    x = float(prompt.split("X = ")[1].split("\n")[0])
    return f"main.py\n<<<<<<< SEARCH\nX = {x}\n=======\nX = {x + 1.0}\n>>>>>>> REPLACE\n"


def test_diff_operator_hill_climb(seed):
    run = FileRun(seed, lambda p: DiffOperator(FunctionModel(bump_diff), p))
    run.run(5)
    assert run.archive.best().genome.text("main.py") == "X = 3.0\n"
    # 4.0 fails the evaluator, so it is rejected (twice: Greedy keeps trying from 3.0)
    assert [bool(i.evaluation.objectives) for i in history.evaluated(run.storage)] == [True, True, True, False, False]


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
    individual = commands.evaluate_directory(runs, str(work), [seed_individual.id], "closer")
    assert individual.evaluation.objectives == {"score": -0.5} and individual.info["note"] == "closer"
    assert individual.info["admitted"]
    assert "+X = 2.5" in commands.diff_individuals(seed_individual, individual)
    assert commands.read_file(individual, "main.py") == "X = 2.5\n"
    edited = commands.edit_individual(runs, individual, {"main.py": "X = 3.0\n"})
    assert edited.parents == (individual.id,) and edited.info["admitted"]
    with pytest.raises(ek.UsageError, match="not genome files"):
        commands.edit_individual(runs, individual, {"evaluate.py": "cheat"})
    with pytest.raises(ek.UsageError, match="no individual"):
        commands.evaluate_directory(runs, str(work), ["nope"])


def test_coding_agent_operator(seed, fake_agent):
    run = FileRun(
        seed, lambda p: CodingAgentOperator(CommandAgent(fake_agent), p, draft_evaluate="python evaluate.py", max_draft_evals=3)
    )
    run.run(3)
    best = run.archive.best()
    assert best.genome.text("main.py") == "X = 2.0\n" and best.info["operator"] == "coding-agent"
    assert "draft={'score': -1.0}" in best.info["summary"]
    used = history.usage(run.storage)
    assert used["draft_evals"] == 2 and history.counts(run.storage)["evaluated"] == 3
    assert used["cost_usd"] == pytest.approx(0.02)


def test_coding_agent_tampering_rejected(seed, fake_agent, monkeypatch):
    monkeypatch.setenv("TAMPER", "1")
    run = FileRun(seed, lambda p: CodingAgentOperator(CommandAgent(fake_agent), p))
    run.run(2, max_cost=0.01)
    failed = [r for r in history.records(run.storage) if r["kind"] == "note" and r["note"] == "failed"]
    assert failed and "evaluate.py" in failed[0]["text"]
    assert history.usage(run.storage)["cost_usd"] == pytest.approx(0.01)
