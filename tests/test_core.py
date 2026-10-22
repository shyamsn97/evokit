"""The core: storage, history, archives, strategies, operators and ask/evaluate/tell, without files."""

import asyncio
import math
import time

import pytest

import evokit as ek
from evokit import history
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


def scored(genome, score, parents=(), **objectives):
    return ek.Individual.new(genome, list(parents)).with_evaluation(ek.Evaluation({"score": score, **objectives}))


def closeness(individual, workdir):
    return ek.Evaluation({"score": -abs(individual.genome - 3)})


def bump(step=1.0):
    return ek.Mutate(lambda x: x + step)


async def seeded(storage, archive, *genomes, eval_fn=closeness):
    seeds = [ek.Individual.new(g) for g in genomes]
    await ek.tell(storage, archive, await ek.evaluate(storage, seeds, eval_fn=eval_fn))
    return archive


async def loop(storage, archive, strategy, operator, evaluations, eval_fn=closeness):
    while history.counts(storage)["evaluated"] < evaluations:
        asked = await ek.ask(storage, archive, strategy, operator)
        if asked.exhausted:
            return
        await ek.tell(storage, archive, await ek.evaluate(storage, asked.individuals, eval_fn=eval_fn))


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


# the loop and the history


def test_hill_climb_and_history_totals():
    storage = MemoryStorage()

    async def go():
        archive = await seeded(storage, BestArchive(storage=storage), 0.0)
        await loop(storage, archive, Greedy(), bump(), 5)
        return archive

    archive = asyncio.run(go())
    assert archive.best().genome == 3.0
    counts = history.counts(storage)
    assert counts["created"] == 4 and counts["evaluated"] == 5 and counts["added"] == 4
    assert [i.genome for i in history.admitted(storage)] == [0.0, 1.0, 2.0, 3.0]
    assert [r["kind"] for r in history.records(storage)][:3] == ["evaluated", "created", "evaluated"]
    chain = history.admitted(storage)
    assert all(later.parents == (earlier.id,) for earlier, later in zip(chain, chain[1:]))


def test_every_cost_is_in_the_history():
    storage = MemoryStorage()
    model = FunctionModel(lambda prompt: ek.operators.Completion("x", {"cost_usd": 0.25}))

    def priced(individual, workdir):
        return ek.Evaluation({"score": len(individual.genome)}, usage={"cost_usd": 0.5})

    async def go():
        archive = await seeded(storage, BestArchive(storage=storage), "a", eval_fn=priced)
        asked = await ek.ask(storage, archive, Greedy(), TextRewriteOperator(model))
        await ek.tell(storage, archive, await ek.evaluate(storage, asked.individuals, eval_fn=priced))
        history.note(storage, "judge", "a judge's verdict", {"cost_usd": 0.125})

    asyncio.run(go())
    assert history.usage(storage)["cost_usd"] == pytest.approx(0.5 + 0.25 + 0.5 + 0.125)
    assert history.last_note(storage, "judge") == "a judge's verdict" and history.counts(storage)["judge"] == 1


def test_operator_failures_and_exhausted():
    storage = MemoryStorage()

    class Fails(ek.Operator):
        async def create(self, selection, archive):
            raise ek.OperatorError("no idea", {"cost_usd": 0.5})

    async def go():
        archive = BestArchive(storage=storage)
        empty = await ek.ask(storage, archive, Greedy(), Fails())
        await seeded(storage, archive, 0.0)
        failed = await ek.ask(storage, archive, Greedy(), Fails(), n=2)
        return empty, failed

    empty, failed = asyncio.run(go())
    assert empty.exhausted and not empty.individuals
    assert not failed.exhausted and not failed.individuals and [f.reason for f in failed.failures] == ["no idea"] * 2
    assert failed.failures[0].selection.parents[0].genome == 0.0
    assert history.usage(storage)["cost_usd"] == 1.0 and history.counts(storage)["failed"] == 2


def test_evaluate_turns_crashes_and_timeouts_into_empty_evaluations():
    storage = MemoryStorage()

    def crash(individual, workdir):
        raise RuntimeError("boom")

    async def slow(individual, workdir):
        await asyncio.sleep(5)

    def wrong(individual, workdir):
        return 1.0

    async def go():
        individuals = [ek.Individual.new(0)]
        return (
            await ek.evaluate(storage, individuals, eval_fn=crash),
            await ek.evaluate(storage, individuals, eval_fn=slow, timeout=0.05),
            await ek.evaluate(storage, individuals, eval_fn=wrong),
        )

    (crashed,), (timed_out,), (wrong_type,) = asyncio.run(go())
    assert crashed.evaluation.objectives == {} and "boom" in crashed.evaluation.feedback
    assert timed_out.evaluation.objectives == {} and "timed out" in timed_out.evaluation.feedback
    assert "must return an Evaluation" in wrong_type.evaluation.feedback
    assert not BestArchive().add(crashed)
    with pytest.raises(ValueError):
        asyncio.run(ek.tell(storage, BestArchive(), [ek.Individual.new(0)]))


def test_eval_fn_gets_its_artifact_folder(tmp_path):
    storage = DirectoryStorage(tmp_path / "run")

    def write_log(individual, workdir):
        (workdir / "log.txt").write_text(f"evaluated {individual.genome}")
        return ek.Evaluation({"score": 1.0})

    (individual,) = asyncio.run(ek.evaluate(storage, [ek.Individual.new(7)], eval_fn=write_log))
    assert storage.read(f"artifacts/{individual.id}/log.txt") == b"evaluated 7"


def test_known_seed_skips_evaluate():
    storage = MemoryStorage()
    archive = BestArchive(storage=storage)
    seed = ek.Individual.new(1.0).with_evaluation(ek.Evaluation({"score": 0.52}))
    assert asyncio.run(ek.tell(storage, archive, [seed])) == [seed] and archive.best() == seed


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
        return await ek.ask(storage, archive, Greedy(), Counting())

    (child,) = asyncio.run(crash_before_evaluating()).individuals
    assert [i.id for i in history.pending(storage)] == [child.id]

    reopened = DirectoryStorage(path)  # a new process: its own history cache

    async def resume():
        archive = BestArchive(storage=reopened)
        asked = await ek.ask(reopened, archive, Greedy(), Counting())
        await ek.tell(reopened, archive, await ek.evaluate(reopened, asked.individuals, eval_fn=closeness))
        return asked

    asked = asyncio.run(resume())
    assert [i.id for i in asked.individuals] == [child.id] and made == [child.id]
    assert history.usage(reopened)["cost_usd"] == 1.0 and not history.pending(reopened)


def test_concurrent_workers_overlap():
    storage = MemoryStorage()

    async def slow(individual, workdir):
        await asyncio.sleep(0.05)
        return ek.Evaluation({"score": individual.genome})

    async def go():
        archive = await seeded(storage, UnboundedArchive(storage=storage), 0.0, eval_fn=slow)
        started = time.monotonic()
        await asyncio.gather(*(loop(storage, archive, Greedy(), bump(), 9, eval_fn=slow) for _ in range(4)))
        return time.monotonic() - started

    assert asyncio.run(go()) < 0.05 * 8 * 0.75
    assert history.counts(storage)["evaluated"] >= 9


def test_discarded_children_are_not_pending():
    storage = MemoryStorage()

    async def go():
        archive = await seeded(storage, BestArchive(storage=storage), 0.0)
        (child,) = (await ek.ask(storage, archive, Greedy(), bump())).individuals
        history.discard(storage, child, "not novel")

    asyncio.run(go())
    assert not history.pending(storage) and history.counts(storage)["discarded"] == 1


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
    assert {i.info["operator"] for i in history.evaluated(storage) if i.parents} == {"up", "down"}


def test_text_operator_shows_inspirations():
    prompts = []
    model = FunctionModel(lambda prompt: prompts.append(prompt) or "```\nbetter\n```")
    parent, other = scored("parent text", 1.0), scored("other text", 2.0)
    offspring = asyncio.run(TextRewriteOperator(model).create(ek.Selection((parent,), (other,)), BestArchive()))
    (child,) = offspring.children
    assert child.genome == "better" and child.parents == (parent.id,)
    assert "parent text" in prompts[0] and "other text" in prompts[0]
