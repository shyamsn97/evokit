"""The core: storage, history, archives, strategies, operators and ask/evaluate/tell, without files."""

import asyncio
import functools
import json
import math
import os
import time

import pytest

import evokit as ek
from evokit.history import History
from evokit.archives import (
    BestArchive,
    IslandArchive,
    LatestArchive,
    NoveltyArchive,
    PopulationArchive,
    SimplerBestArchive,
    UnboundedArchive,
)
from evokit.operators import Choice, FunctionModel, TextRewriteOperator
from evokit.storage import decode, encode
from evokit.storages import DirectoryStorage, MemoryStorage
from evokit.strategies import Greedy, IslandSampler, ProportionalSelection, Tournament


@functools.cache
def history_of(storage) -> History:
    """One History per storage, as a process keeps one per run."""
    return History(storage)


def scored(genome, score, parents=(), **objectives):
    return ek.Individual.new(genome, list(parents)).with_evaluation(ek.Evaluation({"score": score, **objectives}))


def closeness(individual):
    return ek.Evaluation({"score": -abs(individual.genome - 3)})


def bump(step=1.0):
    return ek.Mutate(lambda x: x + step)


async def seeded(storage, archive, *genomes, eval_fn=closeness):
    seeds = [ek.Individual.new(g) for g in genomes]
    await ek.tell(history_of(storage), archive, seeds, await asyncio.gather(*map(ek.as_eval_fn(eval_fn), seeds)))
    return archive


async def loop(storage, archive, strategy, operator, evaluations, eval_fn=closeness):
    while history_of(storage).counts()["evaluated"] < evaluations:
        asked = await ek.ask(history_of(storage), archive, strategy, operator)
        if asked.exhausted:
            return
        await ek.tell(history_of(storage), archive, asked.individuals, await asyncio.gather(*map(ek.as_eval_fn(eval_fn), asked.individuals)))


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


# the loop and the history


def test_hill_climb_and_history_totals():
    storage = MemoryStorage()

    async def go():
        archive = await seeded(storage, BestArchive(storage=storage), 0.0)
        await loop(storage, archive, Greedy(), bump(), 5)
        return archive

    archive = asyncio.run(go())
    assert archive.best().genome == 3.0
    counts = history_of(storage).counts()
    assert counts["created"] == 4 and counts["evaluated"] == 5 and counts["added"] == 4
    assert [i.genome for i in history_of(storage).admitted()] == [0.0, 1.0, 2.0, 3.0]
    assert [r["kind"] for r in history_of(storage).records()][:3] == ["evaluated", "created", "evaluated"]
    chain = history_of(storage).admitted()
    assert all(later.parents == (earlier.id,) for earlier, later in zip(chain, chain[1:]))


def test_every_cost_is_in_the_history():
    storage = MemoryStorage()
    model = FunctionModel(lambda prompt: ek.operators.Completion("x", {"cost_usd": 0.25}))

    def priced(individual):
        return ek.Evaluation({"score": len(individual.genome)}, usage={"cost_usd": 0.5})

    async def go():
        archive = await seeded(storage, BestArchive(storage=storage), "a", eval_fn=priced)
        asked = await ek.ask(history_of(storage), archive, Greedy(), TextRewriteOperator(model))
        await ek.tell(history_of(storage), archive, asked.individuals, await asyncio.gather(*map(ek.as_eval_fn(priced), asked.individuals)))
        history_of(storage).note("judge", "a judge's verdict", {"cost_usd": 0.125})

    asyncio.run(go())
    assert history_of(storage).usage()["cost_usd"] == pytest.approx(0.5 + 0.25 + 0.5 + 0.125)
    assert history_of(storage).last_note("judge") == "a judge's verdict" and history_of(storage).counts()["judge"] == 1


def test_operator_failures_and_exhausted():
    storage = MemoryStorage()

    class Fails(ek.Operator):
        async def create(self, selection, archive):
            raise ek.OperatorError("no idea", {"cost_usd": 0.5})

    async def go():
        archive = BestArchive(storage=storage)
        empty = await ek.ask(history_of(storage), archive, Greedy(), Fails())
        await seeded(storage, archive, 0.0)
        failed = await ek.ask(history_of(storage), archive, Greedy(), Fails(), n=2)
        return empty, failed

    empty, failed = asyncio.run(go())
    assert empty.exhausted and not empty.individuals
    assert not failed.exhausted and not failed.individuals and [f.reason for f in failed.failures] == ["no idea"] * 2
    assert failed.failures[0].selection.parents[0].genome == 0.0
    assert history_of(storage).usage()["cost_usd"] == 1.0 and history_of(storage).counts()["failed"] == 2


def test_max_failures_stops_a_loop_that_only_fails():
    from evokit.interface.run import RunCommands

    class Fails(ek.Operator):
        async def create(self, selection, archive):
            raise ek.OperatorError("Invalid API key")

    storage = MemoryStorage()
    archive = asyncio.run(seeded(storage, BestArchive(storage=storage), 0.0))
    RunCommands(storage, archive, Greedy(), Fails(), closeness).run(max_failures=3)
    assert history_of(storage).counts()["failed"] == 3
    assert history_of(storage).last_note("stop") == "3 operator failures in a row; the last: Invalid API key"


def test_failures_in_a_row_reset_when_a_child_is_created():
    calls = []

    class Flaky(ek.Operator):
        async def create(self, selection, archive):
            calls.append(1)
            if len(calls) % 3:
                raise ek.OperatorError("flaky")
            return ek.Offspring((ek.Individual.new(selection.parents[0].genome + 1, [selection.parents[0].id]),))

    storage = MemoryStorage()

    async def go():
        archive = await seeded(storage, BestArchive(storage=storage), 0.0)
        while not ek.Limits(max_evaluations=3, max_failures=3).reached(history_of(storage)):
            asked = await ek.ask(history_of(storage), archive, Greedy(), Flaky())
            await ek.tell(history_of(storage), archive, asked.individuals, await asyncio.gather(*map(ek.as_eval_fn(closeness), asked.individuals)))

    asyncio.run(go())
    assert history_of(storage).counts()["evaluated"] == 3 and history_of(storage).counts()["failed"] == 4
    assert history_of(storage).failures_in_a_row() == 0


def test_a_failing_check_stops_the_loop_before_it_starts():
    from evokit.interface.run import RunCommands

    class NoKey(ek.Operator):
        async def check(self):
            raise ek.UsageError("no API key")

        async def create(self, selection, archive):
            raise AssertionError("create ran")

    storage = MemoryStorage()
    archive = asyncio.run(seeded(storage, BestArchive(storage=storage), 0.0))
    before = len(history_of(storage).records())
    with pytest.raises(ek.UsageError, match="no API key"):
        RunCommands(storage, archive, Greedy(), Choice({"a": NoKey()}), closeness).run()
    assert len(history_of(storage).records()) == before


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
    @ek.eval_fn
    def sync(individual):
        return ek.Evaluation({"score": individual.genome})

    @ek.eval_fn(timeout="1s")
    async def coro(individual):
        return ek.Evaluation({"score": individual.genome + 1})

    async def go():
        individual = ek.Individual.new(1)
        return await sync(individual), await coro(individual)

    assert [e.objectives["score"] for e in asyncio.run(go())] == [1, 2]
    assert sync.__name__ == "sync" and sync.is_eval_fn


def test_eval_fn_turns_crashes_and_timeouts_into_empty_evaluations():

    def crash(individual):
        raise RuntimeError("boom")

    @ek.eval_fn(timeout=0.05)
    async def slow(individual):
        await asyncio.sleep(5)

    def wrong(individual):
        return 1.0

    async def go():
        individual = ek.Individual.new(0)
        return [await ek.as_eval_fn(fn)(individual) for fn in (crash, slow, wrong)]

    crashed, timed_out, wrong_type = asyncio.run(go())
    assert crashed.objectives == {} and "boom" in crashed.feedback
    assert timed_out.objectives == {} and "timed out" in timed_out.feedback
    assert "must return an Evaluation" in wrong_type.feedback
    assert not BestArchive().add(ek.Individual.new(0).with_evaluation(crashed))
    with pytest.raises(ValueError):
        asyncio.run(ek.tell(MemoryStorage(), BestArchive(), [ek.Individual.new(0)], []))


def test_tell_stores_evaluation_logs_in_the_artifact_folder(tmp_path):
    storage = DirectoryStorage(tmp_path / "run")
    individual = ek.Individual.new(7)
    evaluation = ek.Evaluation({"score": 1.0}, logs={"log.txt": b"evaluated 7"})
    (admitted,) = asyncio.run(ek.tell(history_of(storage), BestArchive(storage=storage), [individual], [evaluation]))
    assert storage.read(f"artifacts/{individual.id}/log.txt") == b"evaluated 7"
    assert admitted.evaluation.logs == {}


def test_known_seed_skips_evaluate():
    storage = MemoryStorage()
    archive = BestArchive(storage=storage)
    seed = ek.Individual.new(1.0).with_evaluation(ek.Evaluation({"score": 0.52}))
    assert asyncio.run(ek.tell(history_of(storage), archive, [seed], [seed.evaluation])) == [seed] and archive.best() == seed


def test_pending_children_survive_a_crash(tmp_path):
    path = tmp_path / "run"
    storage = DirectoryStorage(path)
    made = []

    class Counting(ek.Operator):
        async def create(self, selection, archive):
            child = ek.Individual.new(selection.parents[0].genome + 1, [selection.parents[0].id])
            made.append(child.id)
            return ek.Offspring((child,), {"cost_usd": 1.0})

    async def crash_before_evaluating():
        archive = await seeded(storage, BestArchive(storage=storage), 0.0)
        return await ek.ask(history_of(storage), archive, Greedy(), Counting())

    (child,) = asyncio.run(crash_before_evaluating()).individuals
    assert [i.id for i in history_of(storage).pending()] == [child.id]

    reopened = DirectoryStorage(path)  # a new process: its own history cache

    async def resume():
        archive = BestArchive(storage=reopened)
        asked = await ek.ask(history_of(reopened), archive, Greedy(), Counting())
        await ek.tell(history_of(reopened), archive, asked.individuals, await asyncio.gather(*map(ek.as_eval_fn(closeness), asked.individuals)))
        return asked

    asked = asyncio.run(resume())
    assert [i.id for i in asked.individuals] == [child.id] and made == [child.id]
    assert history_of(reopened).usage()["cost_usd"] == 1.0 and not history_of(reopened).pending()


def test_concurrent_workers_overlap():
    storage = MemoryStorage()

    async def slow(individual):
        await asyncio.sleep(0.05)
        return ek.Evaluation({"score": individual.genome})

    async def go():
        archive = await seeded(storage, UnboundedArchive(storage=storage), 0.0, eval_fn=slow)
        started = time.monotonic()
        await asyncio.gather(*(loop(storage, archive, Greedy(), bump(), 9, eval_fn=slow) for _ in range(4)))
        return time.monotonic() - started

    assert asyncio.run(go()) < 0.05 * 8 * 0.75
    assert history_of(storage).counts()["evaluated"] >= 9


def test_discarded_children_are_not_pending():
    storage = MemoryStorage()

    async def go():
        archive = await seeded(storage, BestArchive(storage=storage), 0.0)
        (child,) = (await ek.ask(history_of(storage), archive, Greedy(), bump())).individuals
        history_of(storage).discard(child, "not novel")

    asyncio.run(go())
    assert not history_of(storage).pending() and history_of(storage).counts()["discarded"] == 1


# archives


def test_archive_replays_its_log_exactly(tmp_path):
    path = tmp_path / "run"
    storage = DirectoryStorage(path)
    archive = IslandArchive(islands=3, features=("lines",), storage=storage)
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

    reopened = IslandArchive(islands=3, features=("lines",), storage=DirectoryStorage(path))
    assert reopened.members() == archive.members() and reopened.metadata == archive.metadata

    novelty = NoveltyArchive(["x"], k=1, threshold=1.0, storage=storage, name="novelty")
    for x in (0.0, 0.5, 3.0):
        novelty.add(scored(None, 0.0, x=x))
    again = NoveltyArchive(["x"], k=1, threshold=1.0, storage=storage, name="novelty")
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


# strategies and operators


def test_proportional_selection_prefers_good_unexplored_individuals():
    archive = UnboundedArchive()
    good, weak, solved = scored(0, 0.7), scored(1, 0.3), scored(2, 1.0)
    for a in (good, weak, solved, scored(3, 0.6, [good.id]), scored(4, 0.6, [good.id])):
        archive.add(a)
    strategy = ProportionalSelection(seed=0)
    picks = [s.parents[0].id for s in asyncio.run(strategy.select(archive, 2000))]
    assert solved.id not in picks
    members = archive.members()
    w_good, w_weak = (strategy.weights(members)[members.index(a)] for a in (good, weak))
    assert w_good == pytest.approx(1 / (1 + math.exp(-2)) / 3) and w_weak == pytest.approx(1 / (1 + math.exp(2)))
    assert picks.count(good.id) > picks.count(weak.id)


def test_tournament_and_island_sampler():
    archive = UnboundedArchive()
    for score in range(10):
        archive.add(scored(score, float(score)))
    picks = asyncio.run(Tournament(size=10, parents=2, seed=0).select(archive, 3))
    assert all(len(s.parents) == 2 and s.parents[0].objective("score") == 9.0 for s in picks)
    assert asyncio.run(Greedy().select(UnboundedArchive(), 1)) == []

    islands = IslandArchive(islands=2, features=(), cell_size=10)
    islands.add(scored("seed", 1.0))
    for k in range(4):
        islands.add(scored(f"child {k}", 2.0 + k, [islands.island_members(k % 2)[0].id]))
    selections = asyncio.run(IslandSampler(inspirations=2, seed=0).select(islands, 4))
    for selection in selections:
        (parent,) = selection.parents
        assert selection.inspirations and parent not in selection.inspirations


def test_choice_labels_children():
    storage = MemoryStorage()
    operator = Choice({"up": bump(1.0), "down": bump(-1.0)}, seed=0)

    async def go():
        archive = await seeded(storage, UnboundedArchive(storage=storage), 0.0)
        await loop(storage, archive, Greedy(), operator, 20)

    asyncio.run(go())
    assert {i.info["operator"] for i in history_of(storage).evaluated() if i.parents} == {"up", "down"}


def test_text_operator_shows_inspirations():
    prompts = []
    model = FunctionModel(lambda prompt: prompts.append(prompt) or "```\nbetter\n```")
    parent, other = scored("parent text", 1.0), scored("other text", 2.0)
    offspring = asyncio.run(TextRewriteOperator(model).create(ek.Selection((parent,), (other,)), BestArchive()))
    (child,) = offspring.children
    assert child.genome == "better" and child.parents == (parent.id,)
    assert "parent text" in prompts[0] and "other text" in prompts[0]


def test_llm_operator_names_its_child_from_the_reply():
    model = FunctionModel(lambda prompt: "Tighter wording\nShorter and clearer.\n```\nbetter\n```")
    (child,) = asyncio.run(TextRewriteOperator(model).create(ek.Selection((scored("text", 1.0),)), BestArchive())).children
    assert child.name == "Tighter wording" and child.description == "Shorter and clearer."


# reading a run


def test_reading_a_run_like_git(tmp_path, capsys):
    from evokit.interface.cli import main
    from evokit.interface.history import HistoryCommands

    storage = DirectoryStorage(tmp_path / "run")

    def evaluate(individual):
        return ek.Evaluation({"score": -abs(individual.genome["x"] - 3)})

    async def go():
        archive = await seeded(storage, BestArchive(storage=storage), {"x": 0}, eval_fn=evaluate)
        best = archive.best()
        better = ek.Individual.new({"x": 2}, [best.id], name="x to 2", description="closer")
        worse = ek.Individual.new({"x": 9}, [best.id], name="x to 9")
        await ek.tell(history_of(storage), archive, [better, worse], await asyncio.gather(*map(ek.as_eval_fn(evaluate), [better, worse])))
        return best, better, worse

    seed, better, worse = asyncio.run(go())
    runs = HistoryCommands(DirectoryStorage(tmp_path / "run"))
    assert [e["id"] for e in runs.log()] == [better.id, seed.id] and runs.log()[0]["name"] == "x to 2"
    assert [e["id"] for e in runs.log(all=True)] == [worse.id, better.id, seed.id]
    assert [(e["admitted"], e["member"]) for e in runs.log(all=True)] == [(False, False), (True, True), (True, False)]
    assert '-  "x": 0\n+  "x": 2' in runs.diff(better)
    assert runs.show(better)["description"] == "closer" and runs.show(better, "genome") == '{\n  "x": 2\n}'
    assert [g["id"] for g in runs.grep('"x": 9')] == [worse.id] and [g["id"] for g in runs.grep("closer")] == [better.id]
    with pytest.raises(ek.UsageError, match="no parent"):
        runs.diff(seed)

    # a run a Python loop wrote has no run/task.json; the read commands open it anyway
    assert not (tmp_path / "run" / "run" / "task.json").exists()
    capsys.readouterr()
    assert main(["log", "--all", "--dir", str(tmp_path / "run"), "--format", "json"]) == 0
    assert [e["id"] for e in json.loads(capsys.readouterr().out)] == [worse.id, better.id, seed.id]
    assert main(["log", "--dir", str(tmp_path / "missing")]) == 2
    assert "no run" in json.loads(capsys.readouterr().err)["error"]
