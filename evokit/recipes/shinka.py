"""ShinkaEvolve (Sakana AI, 2025): islands, child-discounted parents, a bandit over LLMs, novelty rejection."""

from __future__ import annotations

import asyncio
import inspect
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from evokit.archive import Archive
from evokit.archives import IslandArchive
from evokit.archives.islands import genome_text
from evokit.experiment import Experiment
from evokit.operators import Model, diff, rewrite_files
from evokit.operators.prompts import describe_individual, join_notes
from evokit.records import Individual
from evokit.strategies import Strategy
from evokit.task import Task
from evokit.util import truncate

KINDS = {"diff": 0.6, "full": 0.3, "cross": 0.1}

COMBINE = (
    "\n\nCombine the best ideas of the current program and the other programs shown into one "
    "program that is better than each of them."
)

META_PROMPT = (
    "Below are the best programs of an evolutionary search so far, with their scores and how each "
    "was made. Write a short scratchpad for the next attempts: which ideas worked, which did not, "
    "and what to try next. Plain text, at most 20 lines.\n\n"
)

JUDGE_PROMPT = (
    "Are these two programs meaningfully different approaches, or is the second one essentially "
    "the first with cosmetic changes? Answer NOVEL or SAME on the first line.\n\n"
    "First program:\n{a}\n\nSecond program:\n{b}\n"
)


def vary(kind: str, model: Model, parent: Individual, others: Sequence[Individual], task: Task, instructions: str):
    """The child of one edit kind: a diff, a full rewrite, or a crossover (a rewrite combining them)."""
    if kind == "diff":
        return diff(model, parent, others, task=task, instructions=instructions)
    if kind == "full":
        return rewrite_files(model, parent, others, task=task, instructions=instructions)
    return rewrite_files(model, parent, others, task=task, instructions=instructions + COMBINE)


async def shinka(
    task: Task,
    models: Mapping[str, Model],
    *,
    embed: Callable[[str], Sequence[float]],
    judge: Model,
    summarizer: Model,
    dir: str | Path | None = None,
    archive: IslandArchive | None = None,
    workers: int = 4,
    threshold: float = 0.95,
    every: int = 10,
    max_evaluations: int | None = 150,
    max_cost: float | None = 10.0,
    seed: int | None = None,
) -> Individual | None:
    """Each attempt picks an edit kind (diff, full rewrite, crossover) and a model (UCB1 over
    models, rewarded by how much a child improves on its parent). Children too similar to an
    archive member are discarded before evaluation. Every `every` evaluations the islands
    migrate and `summarizer` rewrites a scratchpad of what has worked (a "meta" note), which goes
    into later prompts.

    `embed(text)` returns a vector (sync or async); `judge` decides near-duplicates. `archive`
    defaults to IslandArchive(islands=2, features=(), cell_size=50).
    """
    archive = archive if archive is not None else IslandArchive(islands=2, features=(), cell_size=50)
    experiment = task.create_experiment(dir, archive=archive, max_evaluations=max_evaluations, max_cost=max_cost)
    strategy, inspirations = parents(objective=task.objective), 4
    bandit = ModelBandit(models, objective=task.objective, seed=seed)
    rng = random.Random(seed)

    async def worker() -> None:
        while not experiment.limits.reached():
            parent, *others = archive.sample(1 + inspirations, strategy) or [None]
            if parent is None:
                return
            kind = rng.choices(list(KINDS), list(KINDS.values()))[0]
            name = bandit.choose(experiment)
            notes = experiment.last_note("meta")
            instructions = join_notes(task.instructions, notes and f"What has worked so far:\n{notes}") or ""
            if child := await experiment.create(vary(kind, models[name], parent, others, task, instructions)):
                if await is_novel(child, archive, embed, judge, experiment, threshold):
                    await experiment.submit(child.with_info(operator=f"{kind}/{name}"))
                else:
                    experiment.discard(child, "not novel")

            if experiment.since_note("migrate") >= every:
                experiment.note("migrate")
                archive.migrate(top=1)
            if experiment.since_note("meta-started") >= every:
                experiment.note("meta-started")
                reply = await summarizer.generate(META_PROMPT + "\n\n".join(map(describe_individual, archive.top(20))))
                experiment.note("meta", reply.text, reply.usage)

    for model in {id(m): m for m in (*models.values(), judge, summarizer)}.values():
        await model.check()
    await asyncio.gather(*(worker() for _ in range(workers)))
    experiment.note("stop", experiment.limits.reached() or "nothing left to select")
    return experiment.best()


def parents(sharpness: float = 10.0, objective: str = "score") -> Strategy:
    """Shinka's sampling, from one island's members (IslandArchive offers one island per call): the
    parent is weighted by sigmoid(normalized score) / (1 + children); the others are inspirations,
    the island's best, then half of them from the best overall."""

    def pick(candidates: Sequence[Individual], n: int, archive: Archive) -> list[Individual]:
        everyone = archive.top(len(archive))
        children = Counter(parent for m in everyone for parent in m.parents)
        members = list(candidates)
        scores = [float(m.objective(objective) or 0.0) for m in members]
        low, high = min(scores), max(scores)
        spread = (high - low) or 1.0
        weights = [
            1 / (1 + math.exp(-sharpness * ((s - low) / spread - 0.5))) / (1 + children[m.id])
            for m, s in zip(members, scores)
        ]
        parent = archive.rng.choices(members, weights)[0]
        chosen = [parent, *[m for m in members if m.id != parent.id][: (n - 1) - (n - 1) // 2]]
        seen = {m.id for m in chosen}
        return (chosen + [m for m in everyone if m.id not in seen])[:n]

    return pick


class ModelBandit:
    """UCB1 over models. A child's reward is how much it improved on its parent's objective (0 if
    not); pulls and rewards are read from the experiment, so a continued one keeps what it learned."""

    def __init__(self, models: Mapping[str, Any], objective: str = "score", exploration: float = 1.0, seed: int | None = None):
        self.names = list(models)
        self.objective, self.exploration = objective, exploration
        self.rng = random.Random(seed)

    def choose(self, experiment: Experiment) -> str:
        pulls = {name: 0 for name in self.names}
        total = {name: 0.0 for name in self.names}
        for child in experiment.evaluated():
            name = str(child.info.get("operator", "")).partition("/")[2]
            if name in pulls:
                pulls[name] += 1
                total[name] += self.reward(child, experiment)
        untried = [name for name in self.names if not pulls[name]]
        if untried:
            return self.rng.choice(untried)
        n = sum(pulls.values())
        return max(self.names, key=lambda name: total[name] / pulls[name] + self.exploration * math.sqrt(2 * math.log(n) / pulls[name]))

    def reward(self, child: Individual, experiment: Experiment) -> float:
        try:
            parent = experiment.individual(child.parents[0]) if child.parents else None
        except KeyError:
            return 0.0
        new, old = child.objective(self.objective), parent.objective(self.objective) if parent else None
        return max(0.0, new - old) if new is not None and old is not None else 0.0


_embeddings: dict[str, Sequence[float]] = {}


async def _embed(embed: Callable[[str], Any], individual: Individual) -> Sequence[float]:
    if individual.id not in _embeddings:
        vector = embed(genome_text(individual))
        _embeddings[individual.id] = await vector if inspect.isawaitable(vector) else vector
    return _embeddings[individual.id]


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


async def is_novel(
    child: Individual,
    archive: IslandArchive,
    embed: Callable[[str], Any],
    judge: Model,
    experiment: Experiment,
    threshold: float = 0.95,
) -> bool:
    """False if the child is a near-copy of an archive member: its embedding is within `threshold`
    cosine similarity of the closest member's, and `judge` says they are the same approach."""
    members = archive.members()
    if not members:
        return True
    vector = await _embed(embed, child)
    similarities = [(_cosine(vector, await _embed(embed, m)), m) for m in members]
    similarity, closest = max(similarities, key=lambda pair: pair[0])
    if similarity < threshold:
        return True
    reply = await judge.generate(JUDGE_PROMPT.format(a=truncate(genome_text(closest), 20_000), b=truncate(genome_text(child), 20_000)))
    experiment.note("novelty-judge", reply.text[:200], reply.usage)
    return not reply.text.strip().upper().startswith("SAME")
