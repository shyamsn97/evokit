"""Tasks: JSON and Files genomes, task folders, and creating, continuing and reopening experiments."""

import json

import pytest

import evokit as ek
from evokit.archives import BestArchive, PopulationArchive
from evokit.experiment import ARCHIVE_FILE, TASK_FILE, task_for
from evokit.files import ScriptEvaluator
from evokit.task import TEMPLATE, init_task

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


def closeness(individual):
    return ek.Evaluation({"score": -abs(individual.genome["x"] - 3)})


def test_a_json_task_checks_out_and_reads_genome_json(tmp_path):
    task = ek.Task({"x": 1}, closeness)
    assert task.editable == ["genome.json"] and task.name == "task"
    path = task.checkout({"x": 2}, tmp_path / "work")
    assert json.loads((path / "genome.json").read_text()) == {"x": 2}
    (path / "genome.json").write_text('{"x": 2.5}')
    assert task.read(path) == {"x": 2.5}
    (path / "genome.json").write_text("not json")
    with pytest.raises(ek.UsageError, match="not valid JSON"):
        task.read(path)


def test_from_directory(task_dir):
    task = ek.Task.from_directory(task_dir)
    assert task.directory == task_dir and task.name == "seed" and task.instructions == "# Goal\n\nRaise X to 3."
    assert task.objective == "score" and task.minimize is False
    assert isinstance(task.evaluate, ScriptEvaluator) and task.evaluate.timeout == 60
    assert sorted(task.files.environment.iterdir()) == sorted(task.files.environment / f for f in ("TASK.md", "config.yaml", "data.txt", "evaluate.py"))
    assert task.seed.paths() == ["main.py"] and task.editable == ["main.py", "except evaluate*, eval/**, .git/**, config.yaml, TASK.md"]
    assert not task.files.is_genome("config.yaml") and not task.files.is_genome("TASK.md")


def test_create_experiment_seeds_a_new_one_and_continues_an_existing_one(task_dir, tmp_path):
    task = ek.Task.from_directory(task_dir)
    experiment = task.create_experiment(tmp_path / "e")
    (seed,) = experiment.archive.members()
    assert seed.objective("score") == -2.0 and seed.name == "seed" and experiment.best() == seed
    assert json.loads(experiment.storage.read(TASK_FILE)) == {"task": str(task_dir)}
    experiment.close()
    again = task.create_experiment(tmp_path / "e")
    assert again.archive.members() == [seed] and again.evaluations() == 1
    again.close()


def test_the_archive_takes_the_tasks_objective_unless_it_has_its_own():
    task = ek.Task({"x": 1}, lambda i: ek.Evaluation({"loss": abs(i.genome["x"] - 3), "score": 0.0}), objective="loss", minimize=True)
    adopted = task.create_experiment(archive=BestArchive()).archive
    assert (adopted.objective, adopted.minimize) == ("loss", True)
    own = task.create_experiment(archive=PopulationArchive(3, objective="score")).archive
    assert (own.objective, own.minimize) == ("score", True)


def test_a_failing_seed_records_nothing(task_dir, tmp_path):
    (task_dir / "main.py").write_text("X = 4.0\n")
    with pytest.raises(ek.UsageError, match="above 3"):
        ek.Task.from_directory(task_dir).create_experiment(tmp_path / "e")
    assert not (tmp_path / "e" / TASK_FILE).exists() and not (tmp_path / "e" / "history").exists()


def test_open_experiment_reopens_the_task_and_the_archive(task_dir, tmp_path):
    experiment = ek.Task.from_directory(task_dir).create_experiment(tmp_path / "e", archive=PopulationArchive(4))
    experiment.close()
    assert (tmp_path / "e" / ARCHIVE_FILE).is_file()
    reopened = ek.open_experiment(tmp_path / "e")
    assert reopened.task.directory == task_dir and isinstance(reopened.archive, PopulationArchive) and reopened.archive.size == 4
    assert [m.id for m in reopened.archive.members()] == [m.id for m in experiment.archive.members()]
    with pytest.raises(ek.UsageError, match="no experiment"):
        ek.open_experiment(tmp_path / "nothing")


def test_an_experiment_belongs_to_one_task(task_dir, tmp_path):
    ek.Task.from_directory(task_dir).create_experiment(tmp_path / "e").close()
    other = tmp_path / "other"
    other.mkdir()
    for name in ("main.py", "evaluate.py", "TASK.md", "config.yaml"):
        (other / name).write_text((task_dir / name).read_text())
    with pytest.raises(ek.UsageError, match="an experiment on the task"):
        ek.Task.from_directory(other).create_experiment(tmp_path / "e")


def test_task_folder_errors(tmp_path, task_dir):
    with pytest.raises(ek.UsageError, match="no config.yaml"):
        ek.Task.from_directory(tmp_path)
    (task_dir / "config.yaml").write_text("task:\n  genome: [main.py]\n")
    with pytest.raises(ek.UsageError, match="no task.evaluate"):
        ek.Task.from_directory(task_dir)
    (task_dir / "config.yaml").write_text(CONFIG)
    (task_dir / "TASK.md").unlink()
    with pytest.raises(ek.UsageError, match="no TASK.md"):
        ek.Task.from_directory(task_dir)


def test_task_for_keeps_the_environment_in_the_experiment_directory(task_dir):
    task = task_for(task_dir, task_dir / "runs" / "a")
    task.create_experiment(task_dir / "runs" / "a").close()
    assert task.files.environment == task_dir / "runs" / "a" / "env"
    (task_dir / "evaluate.py").write_text("raise SystemExit(1)\n")
    again = ek.open_experiment(task_dir / "runs" / "a")
    assert (again.task.files.environment / "evaluate.py").read_text() != "raise SystemExit(1)\n"
    assert len(again.archive) == 1


def test_template_is_a_working_task(tmp_path):
    directory = init_task(tmp_path / "toy")
    assert sorted(p.name for p in directory.iterdir()) == sorted(p.name for p in TEMPLATE.iterdir() if p.name != "__pycache__")
    experiment = ek.Task.from_directory(directory).create_experiment()
    assert experiment.best().objective("score") == -2.0
    with pytest.raises(ek.UsageError, match="not empty"):
        init_task(directory)
