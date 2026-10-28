"""The core: storage, experiments, archives, strategies, variation functions and create/submit, without files."""

import asyncio
import json
import os
import time

import pytest

import evokit as ek
from evokit.archives import (
    BestArchive,
    IslandArchive,
    LatestArchive,
    NoveltyArchive,
    PopulationArchive,
    SimplerBestArchive,
    UnboundedArchive,
)
from evokit.operators import FunctionModel, rewrite_text
from evokit.recipes import islands, proportional
from evokit.storage import decode, encode
from evokit.storages import DirectoryStorage, MemoryStorage
from evokit.strategies import best, tournament


def scored(genome, score, parents=(), **objectives):
    return ek.Individual.new(genome, list(parents)).with_evaluation(ek.Evaluation({"score": score, **objectives}))


def closeness(individual):
    return ek.Evaluation({"score": -abs(individual.genome - 3)})


def bump(step=1.0):
    async def bumped(parent):
        return ek.Individual.new(parent.genome + step, [parent.id], {"operator": "bump"}), {}

    return bumped


def experiment(seed=0.0, evaluate=closeness, archive=None, dir=None, **limits) -> ek.Experiment:
    """A new experiment on Task(seed, evaluate), seeded."""
    return ek.Task(seed, evaluate).create_experiment(dir, archive=archive, **limits)


def unseeded(archive=None, evaluate=closeness, storage=None) -> ek.Experiment:
    return ek.Experiment(storage or MemoryStorage(), ek.Task(None, evaluate), archive if archive is not None else BestArchive())


async def climb(experiment, make, workers=1):
    """Sample, create, submit until a limit: the shape of every recipe."""

    async def worker():
        while not experiment.limits.reached():
            parents = experiment.archive.sample(1, best)
            if not parents:
                return
            if child := await experiment.create(make(parents[0])):
                await experiment.submit(child)

    await asyncio.gather(*(worker() for _ in range(workers)))


# records


def test_evaluation_from_metrics():
    e = ek.Evaluation.from_metrics({"score": 1, "valid": True, "feedback": "hi", "usage": {"cost_usd": 0.5}})
    assert e == ek.Evaluation({"score": 1.0}, "hi", {"cost_usd": 0.5})
    assert ek.Evaluation.from_dict(e.to_dict()) == e
    individual = ek.Individual.new(1)
    assert individual.objective("score") is None and individual.with_evaluation(e).objective("score") == 1.0


# storage


@pytest.mark.parametrize("backend", ["memory", "directory"])
def test_storage_backends(tmp_path, backend):
    storage = MemoryStorage() if backend == "memory" else DirectoryStorage(tmp_path / "run")
    storage.write("a/b.txt", b"one")
    storage.append("a/log", b"x\n")
    storage.append("a/log", b"y\n")
    assert storage.read("a/b.txt") == b"one" and storage.read("a/log") == b"x\ny\n"
    assert storage.exists("a/b.txt") and not storage.exists("nope") and storage.list("a") == ["a/b.txt", "a/log"]
    with storage.local("artifacts/x") as folder:
        (folder / "out.txt").write_text("hello")
    assert storage.read("artifacts/x/out.txt") == b"hello"
    with pytest.raises(ValueError):
        storage.write("../escape", b"")
    individual = scored({"a": [1, 2]}, 1.0).with_info(note="n")
    assert decode(encode(individual, storage), storage) == individual


def test_files_genomes_share_blobs():
    from evokit.files import Files

    storage = MemoryStorage()
    files = Files.from_dict({"main.py": "X = 1\n", "big.txt": "x" * 10_000})
    a, b = ek.Individual.new(files), ek.Individual.new(files.with_files({"main.py": "X = 2\n"}))
    assert decode(encode(a, storage), storage) == a and decode(encode(b, storage), storage) == b
    assert len(storage.list("blobs")) == 3


def test_names_and_descriptions_are_stored_and_old_records_get_them_from_info():
    storage = MemoryStorage()
    named = ek.Individual.new(1, name="one", description="the first\nof many")
    assert decode(encode(named, storage), storage) == named
    old = {k: v for k, v in encode(ek.Individual.new(2, info={"summary": "## Faster loop\n\nVectorized the inner loop."}), storage).items()}
    del old["name"], old["description"]
    individual = decode(old, storage)
    assert individual.name == "Faster loop" and individual.description == "Vectorized the inner loop."
    assert ek.util.title_and_body("x" * 300)[0].endswith("...") and len(ek.util.title_and_body("x" * 300)[0]) == 100


# create, submit and the history


def test_hill_climb_and_history_totals():
    e = experiment(max_evaluations=5)
    asyncio.run(climb(e, bump()))
    assert e.best().genome == 3.0 and e.archive.best() == e.best()
    counts = e.counts()
    assert counts["created"] == 4 and counts["evaluated"] == 5 and counts["added"] == 4
    assert [i.genome for i in e.admitted()] == [0.0, 1.0, 2.0, 3.0]
    assert [r["kind"] for r in e.records()][:3] == ["evaluated", "created", "evaluated"]
    chain = e.admitted()
    assert all(later.parents == (earlier.id,) for earlier, later in zip(chain, chain[1:]))
    assert e.limits.reached() == "5 evaluations"


def test_submit_scores_and_offers_to_the_archive():
    e = experiment()
    kept = asyncio.run(e.submit(ek.Individual.new(2.0, [e.best().id])))
    rejected = asyncio.run(e.submit(ek.Individual.new(9.0, [kept.id])))
    assert kept.info["admitted"] and kept.objective("score") == -1.0 and not rejected.info["admitted"]
    assert e.best().id == kept.id and e.counts()["created"] == 2 and not e.pending()


def test_every_cost_is_in_the_history():
    model = FunctionModel(lambda prompt: ek.operators.Completion("```\nxy\n```", {"cost_usd": 0.25}))

    def priced(individual):
        return ek.Evaluation({"score": len(individual.genome)}, usage={"cost_usd": 0.5})

    e = experiment("a", priced)

    async def go():
        child = await e.create(rewrite_text(model, e.best()))
        await e.submit(child)
        e.note("judge", "a judge's verdict", {"cost_usd": 0.125})

    asyncio.run(go())
    assert e.usage()["cost_usd"] == pytest.approx(0.5 + 0.25 + 0.5 + 0.125)
    assert e.last_note("judge") == "a judge's verdict" and e.counts()["judge"] == 1


async def fails(reason="no idea", usage=None):
    raise ek.OperatorError(reason, usage)


def test_failed_variations_are_recorded_with_their_cost():
    e = unseeded()
    assert e.archive.sample(1) == []

    async def go():
        return [await e.create(fails(usage={"cost_usd": 0.5})) for _ in range(2)]

    assert asyncio.run(go()) == [None, None]
    assert e.usage()["cost_usd"] == 1.0 and e.counts()["failed"] == 2 and e.last_note("failed") == "no idea"


def test_max_failures_stops_a_loop_that_only_fails():
    e = experiment(max_failures=3)
    asyncio.run(climb(e, lambda parent: fails("Invalid API key")))
    assert e.counts()["failed"] == 3
    assert e.limits.reached() == "3 failures in a row; the last: Invalid API key"


def test_failures_in_a_row_reset_when_a_child_is_created():
    calls = []

    async def flaky(parent):
        calls.append(1)
        if len(calls) % 3:
            raise ek.OperatorError("flaky")
        return ek.Individual.new(parent.genome + 1, [parent.id]), {}

    e = experiment(max_evaluations=3, max_failures=3)
    asyncio.run(climb(e, flaky))
    assert e.counts()["evaluated"] == 3 and e.counts()["failed"] == 4
    assert e.failures_in_a_row() == 0


@pytest.mark.parametrize("backend", ["memory", "directory"])
def test_one_driver_at_a_time(tmp_path, backend):
    storage = MemoryStorage() if backend == "memory" else DirectoryStorage(tmp_path / "run")
    with storage.lock("driver"):
        with pytest.raises(ek.RunBusy):
            with storage.lock("driver"):
                pass
        if backend == "directory":
            with pytest.raises(ek.RunBusy, match=f"pid {os.getpid()}"):
                with DirectoryStorage(tmp_path / "run").lock("driver"):
                    pass
        with storage.lock("other"):
            pass
    with storage.lock("driver"):
        pass


def test_eval_fn_decorator_makes_sync_and_async_functions_async():
    @ek.evaluator
    def sync(individual):
        return ek.Evaluation({"score": individual.genome})

    @ek.evaluator(timeout="1s")
    async def coro(individual):
        return ek.Evaluation({"score": individual.genome + 1})

    async def go():
        individual = ek.Individual.new(1)
        return await sync(individual), await coro(individual)

    assert [e.objectives["score"] for e in asyncio.run(go())] == [1, 2]
    assert sync.__name__ == "sync" and sync.is_evaluator


def test_eval_fn_turns_crashes_and_timeouts_into_empty_evaluations():

    def crash(individual):
        raise RuntimeError("boom")

    @ek.evaluator(timeout=0.05)
    async def slow(individual):
        await asyncio.sleep(5)

    def wrong(individual):
        return 1.0

    async def go():
        individual = ek.Individual.new(0)
        return [await ek.as_evaluate(fn)(individual) for fn in (crash, slow, wrong)]

    crashed, timed_out, wrong_type = asyncio.run(go())
    assert crashed.objectives == {} and "boom" in crashed.feedback
    assert timed_out.objectives == {} and "timed out" in timed_out.feedback
    assert "must return an Evaluation" in wrong_type.feedback
    assert not BestArchive().add(ek.Individual.new(0).with_evaluation(crashed))


def test_submit_stores_evaluation_logs_in_the_artifact_folder(tmp_path):
    def logged(individual):
        return ek.Evaluation({"score": 1.0}, logs={"log.txt": b"evaluated 7"})

    e = unseeded(evaluate=logged, storage=DirectoryStorage(tmp_path / "run"))
    individual = ek.Individual.new(7)
    admitted = asyncio.run(e.submit(individual))
    assert e.storage.read(f"artifacts/{individual.id}/log.txt") == b"evaluated 7"
    assert admitted.evaluation.logs == {} and admitted.info["admitted"]
    e.close()


def test_pending_children_survive_a_crash(tmp_path):
    e = experiment(dir=tmp_path / "run")
    child = asyncio.run(e.create(_priced(bump()(e.best()))))
    assert [i.id for i in e.pending()] == [child.id]
    e.close()

    reopened = experiment(dir=tmp_path / "run")  # a new process: its own Experiment, which submits it
    assert not reopened.pending() and child.id in {i.id for i in reopened.evaluated()}
    assert reopened.usage()["cost_usd"] == 1.0
    reopened.close()


async def _priced(make):
    child, _ = await make
    return child, {"cost_usd": 1.0}


def test_concurrent_workers_overlap():
    async def slow(individual):
        await asyncio.sleep(0.05)
        return ek.Evaluation({"score": individual.genome})

    e = experiment(evaluate=slow, archive=UnboundedArchive(), max_evaluations=9)

    async def go():
        started = time.monotonic()
        await climb(e, bump(), workers=4)
        return time.monotonic() - started

    assert asyncio.run(go()) < 0.05 * 8 * 0.75
    assert e.evaluations() >= 9


def test_discarded_children_are_not_pending():
    e = experiment()
    child = asyncio.run(e.create(bump()(e.best())))
    e.discard(child, "not novel")
    assert not e.pending() and e.counts()["discarded"] == 1


# archives


def test_archive_replays_its_log_exactly(tmp_path):
    path = tmp_path / "run"
    storage = DirectoryStorage(path)
    archive = IslandArchive(islands=3, features=("lines",))
    archive.attach(storage)
    seed = scored("program 1", 1.0, lines=30)
    assert archive.add(seed)
    copies = {i: archive.island_members(i)[0] for i in range(3)}
    assert copies[0].id == seed.id and copies[1].parents == (seed.id,) and copies[2].genome == seed.genome
    child = scored("program 2", 1.5, [copies[2].id], lines=30)
    assert archive.add(child) and archive.island_of(child.id) == 2
    archive.migrate(top=1)
    assert archive.island_members(0)[0].genome == child.genome
    assert archive.island_members(0)[0].info["operator"] == "migration"
    assert set(archive.niches()) == {(0, (2,)), (1, (2,)), (2, (2,))}

    reopened = IslandArchive(islands=3, features=("lines",))
    reopened.attach(DirectoryStorage(path))
    assert reopened.members() == archive.members() and reopened.metadata == archive.metadata

    novelty = NoveltyArchive(["x"], k=1, threshold=1.0, name="novelty")
    novelty.attach(storage)
    for x in (0.0, 0.5, 3.0):
        novelty.add(scored(None, 0.0, x=x))
    again = NoveltyArchive(["x"], k=1, threshold=1.0, name="novelty")
    again.attach(storage)
    assert [m["behaviour"] for m in again.metadata.values()] == [[0.0], [3.0]]
    assert again.state == novelty.state and again.state["threshold"] == pytest.approx(1.0 * 1.05 * 0.95 * 1.05)
    assert storage.exists("archive/changes.jsonl") and storage.exists("novelty/changes.jsonl")


def test_best_archive_ties():
    reject, accept = BestArchive(), BestArchive(ties="accept")
    for archive in (reject, accept):
        assert archive.add(scored(0, 1.0)) and not archive.add(scored(1, 0.5))
        assert not archive.add(ek.Individual.new(2).with_evaluation(ek.Evaluation(feedback="crashed")))
    assert not reject.add(scored(3, 1.0)) and accept.add(scored(3, 1.0))
    assert len(accept) == 1 and accept.best().genome == 3


def test_simpler_best_archive():
    archive = SimplerBestArchive()
    assert archive.add(scored(0, 1.0, lines=30))
    assert not archive.add(scored(0, 1.0, lines=30)) and not archive.add(scored(0, 1.0 + 1e-9, lines=31))
    assert archive.add(scored(0, 1.0, lines=29)) and archive.add(scored(0, 1.1, lines=50))
    assert not archive.add(scored(0, 1.05, lines=10))


def test_population_and_latest_archives():
    population = PopulationArchive(size=2)
    for score in (1.0, 3.0, 2.0, 0.5):
        population.add(scored(score, score))
    assert [m.objective("score") for m in population.members()] == [3.0, 2.0]
    latest = LatestArchive()
    latest.add(scored(0, 0.9))
    newest = scored(1, 0.1)
    assert latest.add(newest) and latest.members() == [newest]


# sampling


def test_archives_sample_with_their_default_or_any_strategy():
    unbounded, best_only = UnboundedArchive(seed=0), BestArchive()
    assert unbounded.sample(1) == [] and unbounded.top(1) == []
    members = [scored(k, float(k)) for k in range(10)]
    for m in members:
        unbounded.add(m)
        best_only.add(m)
    assert [m.objective("score") for m in unbounded.top(3)] == [9.0, 8.0, 7.0]
    assert len({m.id for m in unbounded.sample(5)}) == 5 and len(unbounded.sample(50)) == 10
    assert best_only.sample(1) == [members[-1]] and unbounded.sample(2, best) == members[:-3:-1]
    picks = unbounded.sample(3, tournament(10))
    assert len(picks) == 3 and all(p.objective("score") >= 7.0 for p in picks)


def test_proportional_prefers_good_unexplored_individuals():
    archive = UnboundedArchive(seed=0)
    good, weak, solved = scored(0, 0.7), scored(1, 0.3), scored(2, 1.0)
    for a in (good, weak, solved, scored(3, 0.6, [good.id]), scored(4, 0.6, [good.id])):
        archive.add(a)
    picks = [p.id for p in archive.sample(2000, proportional())]
    assert solved.id not in picks
    assert picks.count(good.id) > picks.count(weak.id)


def test_islands_take_turns_and_give_inspirations():
    archive = IslandArchive(islands=2, features=(), cell_size=10, seed=0)
    archive.add(scored("seed", 1.0))
    for k in range(4):
        archive.add(scored(f"child {k}", 2.0 + k, [archive.island_members(k % 2)[0].id]))
    first, second = archive.sample(3, islands()), archive.sample(3, islands())
    for parent, *others in (first, second):
        assert others and parent not in others
    assert {archive.island_of(first[0].id), archive.island_of(second[0].id)} == {0, 1}


def test_rewrite_text_uses_the_instructions_and_shows_others():
    prompts = []
    model = FunctionModel(lambda prompt: prompts.append(prompt) or "```\nbetter\n```")
    parent, other = scored("parent text", 1.0), scored("other text", 2.0)
    child, _ = asyncio.run(rewrite_text(model, parent, [other], instructions="Be brief."))
    assert child.genome == "better" and child.parents == (parent.id,)
    assert "parent text" in prompts[0] and "other text" in prompts[0] and "Be brief." in prompts[0]


def test_rewrite_text_names_its_child_from_the_reply():
    model = FunctionModel(lambda prompt: "Tighter wording\nShorter and clearer.\n```\nbetter\n```")
    child, _ = asyncio.run(rewrite_text(model, scored("text", 1.0)))
    assert child.name == "Tighter wording" and child.description == "Shorter and clearer."
    with pytest.raises(ek.OperatorError, match="did not change"):
        asyncio.run(rewrite_text(FunctionModel(lambda prompt: "```\ntext\n```"), scored("text", 1.0)))


# reading an experiment


def test_reading_an_experiment_like_git(tmp_path, capsys):
    from evokit.interface.cli import main

    def evaluate(individual):
        return ek.Evaluation({"score": -abs(individual.genome["x"] - 3)})

    e = experiment({"x": 0}, evaluate, dir=tmp_path / "run")

    async def go():
        best = e.best()
        better = ek.Individual.new({"x": 2}, [best.id], name="x to 2", description="closer")
        worse = ek.Individual.new({"x": 9}, [best.id], name="x to 9")
        await e.submit(better)
        await e.submit(worse)
        return best, better, worse

    seed, better, worse = asyncio.run(go())
    e.close()
    runs = ek.open_experiment(tmp_path / "run")
    assert [e["id"] for e in runs.log()] == [better.id, seed.id] and runs.log()[0]["name"] == "x to 2"
    assert [e["id"] for e in runs.log(all=True)] == [worse.id, better.id, seed.id]
    assert [(e["admitted"], e["member"]) for e in runs.log(all=True)] == [(False, False), (True, True), (True, False)]
    assert '-  "x": 0\n+  "x": 2' in runs.diff(better)
    assert runs.show(better)["description"] == "closer" and runs.show(better, "genome") == '{\n  "x": 2\n}'
    assert [g["id"] for g in runs.grep('"x": 9')] == [worse.id] and [g["id"] for g in runs.grep("closer")] == [better.id]
    with pytest.raises(ek.UsageError, match="no parent"):
        runs.diff(seed)

    # an experiment on a task made in Python has no task folder; the read commands open it anyway
    assert runs.task is None and isinstance(runs.archive, BestArchive)
    capsys.readouterr()
    assert main(["log", "--all", "--dir", str(tmp_path / "run"), "--format", "json"]) == 0
    assert [entry["id"] for entry in json.loads(capsys.readouterr().out)] == [worse.id, better.id, seed.id]
    assert main(["log", "--dir", str(tmp_path / "missing")]) == 2
    assert "no experiment" in json.loads(capsys.readouterr().err)["error"]


def test_the_experiment_knows_when_the_best_improved():
    history = unseeded(BestArchive(ties="accept"), lambda individual: ek.Evaluation({"score": individual.genome}))

    async def go():
        seed = ek.Individual.new(1.0)
        await history.submit(seed)
        tie = await history.submit(ek.Individual.new(1.0, [seed.id]))
        history.note("draft", objectives={"score": 0.5}, improved=False)
        better = await history.submit(ek.Individual.new(2.0, [tie.id]))
        await history.submit(ek.Individual.new(0.0, [better.id]))
        history.note("review")
        history.note("draft", objectives={"score": 1.5}, improved=False)
        return seed, tie, better

    seed, tie, better = asyncio.run(go())
    assert tie.info["admitted"]                       # a tie is kept, but is no improvement
    assert history.best().id == better.id and [i.id for _, i in history.best_over_time()] == [seed.id, better.id]
    assert history.notes("draft")[0]["objectives"] == {"score": 1.5}
    assert history.best_over_time()[1][0] == 3
    assert history.evaluations() == 4 and len(history.drafts()) == 2
    assert history.since_improvement() == 2 and history.since_improvement(drafts=False) == 1
    assert history.since_note("review") == 1 and history.since_note("never") == 6
    assert history.elapsed() >= 0

    plateau = ek.Limits(stop=[lambda e: "plateau" if e.since_improvement() >= 2 else None], experiment=history)
    assert plateau.reached() == "plateau" and history.limits.reached() is None


def test_one_process_writes_an_experiment(tmp_path):
    storage = DirectoryStorage(tmp_path / "run")
    history = ek.Experiment(storage)
    history.note("hello")
    ek.Experiment(DirectoryStorage(tmp_path / "run")).note("same process")   # shares the lock
    with pytest.raises(ek.RunBusy):
        with DirectoryStorage(tmp_path / "run").lock("driver"):
            pass
    history.close()
    with DirectoryStorage(tmp_path / "run").lock("driver"):
        pass
