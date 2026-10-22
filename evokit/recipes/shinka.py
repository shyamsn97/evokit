"""ShinkaEvolve (Sakana AI, 2025): islands, child-discounted parents, a bandit over LLMs, novelty rejection."""

from __future__ import annotations

import asyncio
import inspect
import math
import random
from typing import Any, Callable, Mapping, Sequence

import evokit
from evokit import history
from evokit.archives import IslandArchive
from evokit.archives.islands import genome_text
from evokit.files import DiffOperator, FileRewriteOperator, Project
from evokit.loop import EvalFn
from evokit.operators import Model
from evokit.operators.prompts import describe_individual
from evokit.records import Individual, Selection
from evokit.storage import Storage
from evokit.strategy import Strategy
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


def make_operator(kind: str, model: Model, project: Project, instructions: str):
    if kind == "diff":
        return DiffOperator(model, project, instructions=instructions)
    if kind == "full":
        return FileRewriteOperator(model, project, instructions=instructions)
    return FileRewriteOperator(model, project, instructions=instructions + COMBINE)


async def shinka(
    project: Project,
    eval_fn: EvalFn,
    models: Mapping[str, Model],
    *,
    instructions: str,
    embed: Callable[[str], Sequence[float]],
    judge: Model,
    summarizer: Model,
    storage: Storage,
    archive: IslandArchive,
    objective: str = "score",
    workers: int = 4,
    threshold: float = 0.95,
    every: int = 10,
    max_cost: float = 10.0,
    max_evaluations: int = 150,
    seed: int | None = None,
) -> Individual | None:
    """Each attempt picks an edit kind (diff, full rewrite, crossover) and a model (UCB1 over
    models, rewarded by how much a child improves on its parent). Children too similar to an
    archive member are discarded before evaluation. Every `every` admissions the islands
    migrate, and every `every` evaluations `summarizer` rewrites a scratchpad of what has worked,
    which goes into later prompts.

    `embed(text)` returns a vector (sync or async); `judge` decides near-duplicates. The caller
    passes a seeded archive, e.g. IslandArchive(islands=2, features=(), cell_size=50).
    """
    strategy = ShinkaParents(objective=objective, seed=seed)
    bandit = ModelBandit(models, objective=objective, seed=seed).fit(storage)
    rng = random.Random(seed)
    notes = history.last_note(storage, "meta") or ""

    def parent_of(child: Individual) -> Individual | None:
        if not child.parents:
            return None
        found = archive.get(child.parents[0])
        if found is not None:
            return found
        try:
            return history.individual(storage, child.parents[0])
        except KeyError:
            return None

    async def worker() -> None:
        nonlocal notes
        while history.usage(storage)["cost_usd"] < max_cost and history.counts(storage)["evaluated"] < max_evaluations:
            kind = rng.choices(list(KINDS), list(KINDS.values()))[0]
            name = bandit.choose()
            prompt = instructions + (f"\n\nWhat has worked so far:\n{notes}" if notes else "")
            operator = make_operator(kind, models[name], project, prompt)

            asked = await evokit.ask(storage, archive, strategy, operator)
            if asked.exhausted:
                return
            novel = []
            for child in asked.individuals:
                if await is_novel(child, archive, embed, judge, storage, threshold):
                    novel.append(child.with_info(operator=f"{kind}/{name}"))
                else:
                    history.discard(storage, child, "not novel")
            evaluated = await evokit.evaluate(storage, novel, eval_fn=eval_fn)
            before = history.counts(storage)
            await evokit.tell(storage, archive, evaluated)
            for child in evaluated:
                bandit.reward(child, parent_of(child))
            after = history.counts(storage)
            if before["added"] // every < after["added"] // every:
                archive.migrate(top=1)
            if before["evaluated"] // every < after["evaluated"] // every:
                reply = await summarizer.generate(META_PROMPT + "\n\n".join(describe_individual(m) for m in archive.best_k(20)))
                notes = reply.text
                history.note(storage, "meta", notes, reply.usage)

    await asyncio.gather(*(worker() for _ in range(workers)))
    return archive.best()


class ShinkaParents(Strategy):
    """An island at random; in it, a parent weighted by sigmoid(normalized score) / (1 + children).
    Inspirations: the island's `top` best, then `elites` of the best overall."""

    def __init__(self, top: int = 2, elites: int = 2, sharpness: float = 10.0, objective: str = "score", seed: int | None = None):
        self.top, self.elites, self.sharpness, self.objective = top, elites, sharpness, objective
        self.rng = random.Random(seed)

    async def select(self, archive: IslandArchive, n: int) -> list[Selection]:
        everyone = archive.members()
        if not everyone:
            return []
        children: dict[str, int] = {}
        for member in everyone:
            for parent in member.parents:
                children[parent] = children.get(parent, 0) + 1
        islands = [members for i in range(archive.islands) if (members := archive.island_members(i))]
        selections = []
        for _ in range(n):
            members = self.rng.choice(islands)
            scores = [float(m.objective(self.objective) or 0.0) for m in members]
            low, high = min(scores), max(scores)
            spread = (high - low) or 1.0
            weights = [
                1 / (1 + math.exp(-self.sharpness * ((s - low) / spread - 0.5))) / (1 + children.get(m.id, 0))
                for m, s in zip(members, scores)
            ]
            parent = self.rng.choices(members, weights)[0]
            inspirations = [m for m in members if m.id != parent.id][: self.top]
            seen = {parent.id, *(m.id for m in inspirations)}
            inspirations += [m for m in everyone if m.id not in seen][: self.elites]
            selections.append(Selection((parent,), tuple(inspirations)))
        return selections


class ModelBandit:
    """UCB1 over models. A child's reward is how much it improved on its parent's objective (0 if not)."""

    def __init__(self, models: Mapping[str, Any], objective: str = "score", exploration: float = 1.0, seed: int | None = None):
        self.names = list(models)
        self.objective, self.exploration = objective, exploration
        self.pulls = {name: 0 for name in self.names}
        self.total = {name: 0.0 for name in self.names}
        self.rng = random.Random(seed)

    def fit(self, storage: Storage) -> "ModelBandit":
        """Reward every child already in the history, so a continued run keeps what it learned."""
        for child in history.evaluated(storage):
            try:
                parent = history.individual(storage, child.parents[0]) if child.parents else None
            except KeyError:
                parent = None
            self.reward(child, parent)
        return self

    def choose(self) -> str:
        untried = [name for name in self.names if not self.pulls[name]]
        if untried:
            return self.rng.choice(untried)
        n = sum(self.pulls.values())
        return max(
            self.names,
            key=lambda name: self.total[name] / self.pulls[name] + self.exploration * math.sqrt(2 * math.log(n) / self.pulls[name]),
        )

    def reward(self, child: Individual, parent: Individual | None) -> None:
        operator = str(child.info.get("operator", ""))
        name = operator.split("/", 1)[-1]
        if "/" not in operator or name not in self.pulls:
            return
        new, old = child.objective(self.objective), parent.objective(self.objective) if parent else None
        self.pulls[name] += 1
        self.total[name] += max(0.0, new - old) if new is not None and old is not None else 0.0


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
    storage: Storage,
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
    history.note(storage, "novelty-judge", reply.text[:200], reply.usage)
    return not reply.text.strip().upper().startswith("SAME")
