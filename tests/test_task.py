"""Task folders: load_task, the run section, and opening (seeding or continuing) a run."""

import json

import pytest

import evokit as ek
from evokit import history
from evokit.archives import PopulationArchive
from evokit.files import CodingAgentOperator, DiffOperator, ScriptEvaluator
from evokit.storages import DirectoryStorage, MemoryStorage
from evokit.strategies import Tournament
from evokit.task import TEMPLATE, init_task, open_run

CONFIG = """
task:
  genome: [main.py]
  evaluate: python evaluate.py
  timeout: 1m
  objective: score
"""


@pytest.fixture
def task_dir(seed):
    (seed / "TASK.md").write_text("# Goal\n\nRaise X to 3.\n")
    (seed / "config.yaml").write_text(CONFIG)
    return seed


def test_load_task(task_dir):
    task = ek.load_task(task_dir)
    assert task.directory == task_dir and task.name == "seed" and task.instructions == "# Goal\n\nRaise X to 3."
    assert task.objective == "score" and task.minimize is False and task.run == {}
    assert isinstance(task.eval_fn, ScriptEvaluator) and task.eval_fn.timeout == 60
    assert sorted(task.project.environment.iterdir()) == sorted(task.project.environment / f for f in ("TASK.md", "config.yaml", "data.txt", "evaluate.py"))
    seed = task.seed()
    assert seed.genome.paths() == ["main.py"] and seed.name == "seed" and seed.info == {} and seed.evaluation is None
    assert not task.project.is_genome("config.yaml") and not task.project.is_genome("TASK.md")


def test_open_seeds_a_new_storage_and_continues_an_existing_one(task_dir):
    storage = MemoryStorage()
    run = ek.load_task(task_dir).open(storage)
    (seed,) = run.archive.members()
    assert seed.objective("score") == -2.0 and seed.name == "seed"
    assert json.loads(storage.read("run/task.json")) == {"task": str(task_dir)}
    again = ek.load_task(task_dir).open(storage)
    assert again.archive.members() == [seed] and history.counts(storage)["evaluated"] == 1


def test_a_failing_seed_records_nothing(task_dir):
    (task_dir / "main.py").write_text("X = 4.0\n")
    storage = MemoryStorage()
    with pytest.raises(ek.UsageError, match="above 3"):
        ek.load_task(task_dir).open(storage)
    assert not storage.exists("run/task.json") and history.counts(storage)["evaluated"] == 0


def test_run_table_names_the_components(task_dir):
    (task_dir / "config.yaml").write_text(CONFIG + """
run:
  archive: population
  archive_options: {size: 3}
  strategy: tournament
  strategy_options: {size: 2}
  operator: diff
  price: [3.0, 15.0]
""")
    archive, strategy, operator = ek.load_task(task_dir).components(MemoryStorage())
    assert isinstance(archive, PopulationArchive) and archive.size == 3 and archive.objective == "score"
    assert isinstance(strategy, Tournament) and strategy.size == 2
    assert isinstance(operator, DiffOperator) and operator.instructions == "# Goal\n\nRaise X to 3."
    assert operator.model.price == (3.0, 15.0)

    (task_dir / "config.yaml").write_text(CONFIG + "run:\n  operator: coding-agent\n  agent: my-agent --fast\n  operator_options: {max_draft_evals: 2}\n")
    _, _, operator = ek.load_task(task_dir).components(MemoryStorage())
    assert isinstance(operator, CodingAgentOperator) and operator.agent.argv == ["my-agent", "--fast"]
    assert operator.draft_evaluate is not None

    (task_dir / "config.yaml").write_text(CONFIG + "run:\n  archive: nope\n")
    with pytest.raises(ek.UsageError, match="unknown archive"):
        ek.load_task(task_dir).components(MemoryStorage())


def test_factory_escape_hatch(task_dir):
    (task_dir / "evo.py").write_text(
        "from evokit.archives import BestArchive\nfrom evokit.strategies import Greedy\n"
        "def build(task, storage):\n    return BestArchive(minimize=task.minimize, storage=storage), Greedy()\n"
    )
    (task_dir / "config.yaml").write_text(CONFIG + "run:\n  factory: evo:build\n")
    run = ek.load_task(task_dir).open(MemoryStorage())
    assert len(run.archive) == 1 and run.operator is None and run.eval_fn is run.task.eval_fn


def test_task_folder_errors(tmp_path, task_dir):
    with pytest.raises(ek.UsageError, match="no config.yaml"):
        ek.load_task(tmp_path)
    (task_dir / "config.yaml").write_text("task:\n  genome: [main.py]\n")
    with pytest.raises(ek.UsageError, match="no task.evaluate"):
        ek.load_task(task_dir)
    (task_dir / "config.yaml").write_text(CONFIG)
    (task_dir / "TASK.md").unlink()
    with pytest.raises(ek.UsageError, match="no TASK.md"):
        ek.load_task(task_dir)


def test_open_run_keeps_the_environment_in_the_run_directory(task_dir):
    run = open_run(task_dir / "runs" / "a", task_dir)
    assert run.task.project.environment == task_dir / "runs" / "a" / "env"
    (task_dir / "evaluate.py").write_text("raise SystemExit(1)\n")
    again = open_run(task_dir / "runs" / "a")
    assert (again.task.project.environment / "evaluate.py").read_text() != "raise SystemExit(1)\n"
    assert isinstance(again.storage, DirectoryStorage) and len(again.archive) == 1


def test_template_is_a_working_task(tmp_path):
    directory = init_task(tmp_path / "toy")
    assert sorted(p.name for p in directory.iterdir()) == sorted(p.name for p in TEMPLATE.iterdir() if p.name != "__pycache__")
    run = ek.load_task(directory).open(MemoryStorage())
    assert run.archive.best().objective("score") == -2.0
    with pytest.raises(ek.UsageError, match="not empty"):
        init_task(directory)
