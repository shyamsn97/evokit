"""EvoRLM: evolution as one recursive language model (github.com/shyamsn97/evorlm), on rlmflow.

Every agent is an rlmflow agent running the same policy over a scored population of its own:
`population()` lists it best first, `candidate(id)` gives a candidate's files, `evaluate(candidate)`
scores one out of the agent's evaluations. The root's population is the archive. Each root-level
`await mutate(*ops)` is one generation: every op starts a fresh agent on the parents it names, with
evaluations and agent calls carved from its caller's, and that agent may mutate in turn while it
is above `max_depth` and holds agent calls. Each agent, the root included, hands up one candidate;
the host never trusts its score (it looks it up, or measures it out of what the agent has left, or
drops it) and admits it to its caller's population. What an agent does not spend goes back to its
caller. One mutate and one evaluate per turn, so every result is read before more is spent.

The run is an `Evolution`: an rlmflow Flow holding the populations and the tools, which every
harness offers through its REPL. Nothing is harness-specific: the policy is a `config.prompt`
section, an agent's role and budget are in its task, and the tools report what is left. Every
population is an archive in the experiment's storage: the root's is the experiment's archive, an
op's agent's is a PopulationArchive under populations/<agent id>/. Needs rlmflow 0.6 or later.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from rlmflow import AgentConfig, Flow, Graph, current_code, current_step, tool
from rlmflow.consumers import GraphCheckpointer
from rlmflow.engine.handoff import AWAITED_KEY
from rlmflow.engine.workspaces import LocalWorkspace
from rlmflow.graph.config import DEFAULT_MAX_ITERS
from rlmflow.graph.nodes import Node
from rlmflow.tools import waiting

from evokit.archive import Archive, improves
from evokit.archives import PopulationArchive
from evokit.experiment import Experiment
from evokit.files import Files
from evokit.records import Individual
from evokit.task import Task
from evokit.util import truncate

SOURCE_LIMIT = 20_000

RETURN = {
    "type": "object",
    "properties": {
        "candidate_id": {"type": "string", "description": "an unchanged candidate from your population"},
        "candidate": {"description": "or a new candidate you evaluated"},
        "title": {"type": "string"},
        "parent_ids": {"type": "array", "items": {"type": "string"}, "description": "for a new candidate, the candidates it came from"},
        "notes": {"type": "string", "description": "what you tried, what worked and what failed"},
    },
    "required": ["title"],
}

POLICY = """You are one agent of a recursive evolution policy. Every agent in the tree runs this \
same policy over a scored population of its own; an agent you start has your tools under the budget \
you give it, and hands back one measured candidate into your population. `population()` lists yours \
best first and `candidate(id)` returns one ({shape}). Scores come only from `evaluate(candidate)`, \
under the task's own limits: you may run code in your REPL to check that it works, but never report \
a score `evaluate` did not return. Evaluate at most once per turn, and read the result first.

1. Read `population()` before planning, and `candidate(id)` for exact values.
2. Improve the population: change candidates and evaluate them yourself, and, when `mutate` is \
available, spend part of your budget on recursive `await mutate(*ops)` calls.
3. Each op names parents from your population, a concrete instruction, and its budget: \
{{"parent_ids": [...], "instruction": "...", "evals": n, "agents": k}}. `agents` counts the op's own \
agent, so `agents` above 1 lets it mutate in turn.
4. End the block after `await mutate(...)`, and read its results on your next turn before spending \
more. At most one mutate call per turn.
5. Spend your evaluations: repair a failure, refine a success. Choose width versus recursive depth \
from the evidence.
6. Submit the strongest measured candidate in your population: an unchanged `candidate_id`, or a \
new `candidate` you evaluated with the `parent_ids` it came from. Your caller sees only that \
submission and its notes; everything else stays local.

Read before you act:

```repl
for row in population(topk=6):
    print(row["id"], row["score"], row["title"])
```

Launch ops (`mutate` is already bound; do not import it):

```repl
best = population(topk=1)[0]["id"]
ops = [{{"parent_ids": [best], "instruction": "Apply one task-relevant change and evaluate it.", "evals": 3, "agents": 2}}]
print(await mutate(*ops))
```

Finish with one measured candidate:

```repl
best = population(topk=1)[0]
submit({{"candidate_id": best["id"], "title": best["title"], "notes": "what I tried, what worked and what failed"}})
```"""

LONG_CELLS = """`await mutate(...)` returns only when every op's agent has finished, which can take \
an hour. If you run code through `rlmflow repl`, run that cell as \
`rlmflow repl --timeout 3600 <<'EOF' ... EOF` with your shell tool's timeout at its maximum, and \
wait for it rather than ending your turn."""

ROOT = """{task}

## Your role

Your population is the whole run's. Evolve it for {generations} generations: a generation is one \
`await mutate(...)` of at most {max_ops} ops, each given {eval_budget} evaluations unless you say \
otherwise. Read `population()` before and after every generation. Only once the generations are \
spent, submit the best measured candidate.

{budget}"""

CHILD = """{task}

## Your role

Your instruction, from the agent that started you: {instruction}

{budget}

## Your parents

Your population starts as these candidates, best first:

{parents}"""

NO_MUTATE = "`mutate` is not available to you: improve candidates yourself, and submit your best once your evaluations run out or stop helping."


@dataclass
class Scope:
    """One agent's population (the root's is the archive), what it has left to spend, and its
    place in the tree."""

    depth: int
    evals: int
    agents: int
    generation: int
    population: Archive
    instruction: str = ""
    parents: tuple[str, ...] = ()  # what the op named: a new candidate's lineage
    mutated: set[str] = field(default_factory=set)  # the turns that called mutate
    evaluated: set[str] = field(default_factory=set)  # and evaluate


class Scratch(LocalWorkspace):
    """Each agent works in its own empty directory under `root`, so no agent edits another's files."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def path(self, agent: Node) -> Path:
        directory = self.root / agent.id
        directory.mkdir(parents=True, exist_ok=True)
        return directory.resolve()



async def evorlm(
    task: Task,
    agent: AgentConfig | str,
    *,
    llm_clients: dict[str, Any] | None = None,
    dir: str | Path | None = None,
    archive: Archive | None = None,
    generations: int = 3,
    eval_budget: int = 3,
    population: int = 8,
    max_ops: int = 8,
    max_depth: int = 2,
    max_tokens: int | None = None,
    max_cost: float | None = None,
    guidance: str = "",
) -> Individual | None:
    """Run the root agent for `generations` mutate calls of at most `max_ops` ops; ops run as fresh
    agents at most `max_depth` deep. Every agent, root or op, runs the same policy: `agent`, an
    rlmflow agent config with any harness, or a model name for a REPL agent; `llm_clients` names
    clients a model name can't describe. Each agent works in its own directory, under agents/.

    The root holds generations * max_ops * eval_budget evaluations and generations * max_ops agent
    calls; an op gets `eval_budget` evaluations and one agent call unless it says otherwise.
    `population` caps every local population; `archive` defaults to PopulationArchive(population).
    `max_tokens` is the whole run's token budget; `max_cost` the experiment's dollar limit, which
    counts evaluations' costs. `guidance` is shown to every agent, for an outer loop that improves
    the policy between runs. The root gets at least 3 * generations + 4 turns, and an op at least
    2 * evals + 4. Each run's rlmflow graph is kept in the experiment, under rlm/."""
    archive = archive if archive is not None else PopulationArchive(population)
    experiment = task.create_experiment(dir, archive=archive, max_cost=max_cost, max_failures=None)
    if isinstance(agent, str):
        agent = AgentConfig(model=agent, max_iters=max(6, 2 * eval_budget + 4))
    shape = "a JSON value" if task.files is None else f"a dict from each editable file ({', '.join(task.editable)}) to its full text"
    config = replace(
        agent, name="evorlm", max_depth=0, output_schema=RETURN, max_budget=max_tokens,
        max_iters=max(agent.max_iters or DEFAULT_MAX_ITERS, 3 * generations + 4),
        prompt={"agents": "", "configs": "", **agent.prompt, "evolution": POLICY.format(shape=shape),
                "long cells": LONG_CELLS, "guidance": guidance},
    )
    flow = Evolution(
        experiment, config, llm_clients=llm_clients, cap=population, max_ops=max_ops, max_depth=max_depth,
        generations=generations, eval_budget=eval_budget,
        workspaces=Scratch(Path(dir) / "agents" if dir is not None else tempfile.mkdtemp(prefix="evorlm-")),
    )
    scope = Scope(0, generations * max_ops * eval_budget, generations * max_ops, flow.generation, archive)
    g = Graph()
    root = g.create_agent(ROOT.format(task=flow.instructions, generations=generations, max_ops=max_ops,
                                      eval_budget=eval_budget, budget=flow.budget(scope)))
    flow.scopes[root.id] = scope
    saved = GraphCheckpointer(Path(dir) / "rlm" / f"generation-{flow.generation}") if dir is not None else None
    reminded = set()
    try:
        while True:
            async for step in flow.run_streaming(g, root_config=config):
                if saved is not None:
                    saved.handle(step, g)
            try:
                await flow.admit(scope, await flow.handed_up(scope, g.result(root), g.path(root)))
            except (ValueError, RuntimeError) as exc:
                experiment.note("failed", f"the root's candidate was dropped: {exc}")
            if flow.generation in reminded or not flow.can_mutate(scope) or flow.budget_exhausted(g, root):
                break
            reminded.add(flow.generation)
            g.send(root, f"{flow.last - flow.generation} generations are left: run them before you submit.", sender="evorlm")
    finally:
        if saved is not None:
            saved.flush()
        await flow.aclose()
        experiment.note("agents", f"{len(g.agents())} agents", {**g.usage().as_dict(), "cost_usd": g.cost_usd()})
    experiment.note("stop", experiment.limits.reached() or f"{flow.generation - flow.first} generations")
    return experiment.best()


class Evolution(Flow):
    """The run: the experiment, every agent's scope, measured candidates by genome, and the tools."""

    def __init__(self, experiment: Experiment, config: AgentConfig, *, llm_clients: dict[str, Any] | None,
                 cap: int, max_ops: int, max_depth: int, generations: int, eval_budget: int, workspaces: Any = None):
        self.experiment, self.task, self.archive = experiment, experiment.task, experiment.archive
        self.cap, self.max_ops, self.max_depth, self.eval_budget = cap, max_ops, max_depth, eval_budget
        self.first = self.generation = len(experiment.notes("generation"))
        self.last = self.first + generations
        self.instructions = self.task.instructions or "Improve the candidate's score."
        self.scopes: dict[str, Scope] = {}
        self.measured = {key(m.genome): m for m in self.archive.members()}
        self.local: dict[str, Individual] = {}  # admitted below the root, by id
        super().__init__(llm_clients=llm_clients, configs=[config], tools=self.tools(), workspaces=workspaces)

    def budget(self, scope: Scope) -> str:
        """What the agent has left, and whether it can mutate."""
        budget = f"You have {scope.evals} evaluations and {scope.agents} agent calls left."
        return f"{budget} {NO_MUTATE}" if not self.can_mutate(scope) else budget

    def left(self, scope: Scope) -> dict:
        left = {"evals_left": scope.evals, "agents_left": scope.agents}
        return left | {"generations_left": self.last - self.generation} if scope.depth == 0 else left

    def scope(self) -> Scope:
        step = current_step()
        if step is None or step.agent.id not in self.scopes:
            raise RuntimeError("this tool runs only inside an EvoRLM agent")
        return self.scopes[step.agent.id]

    def find(self, scope: Scope, id: str) -> Individual:
        if (member := scope.population.get(id)) is None:
            raise ValueError(f"{id!r} is not in your population")
        return member

    def can_mutate(self, scope: Scope) -> bool:
        return (scope.depth < self.max_depth and scope.evals > 0 and scope.agents > 0
                and (scope.depth > 0 or self.generation < self.last) and not self.experiment.limits.reached())

    def tools(self) -> list:
        run = self

        @tool("Your population, best first: id, score, title, parent ids.", proxy=True)
        def population(topk: int | None = None) -> list[dict]:
            scope = run.scope()
            return [row(m, run.task.objective) for m in scope.population.top(len(scope.population))[:topk]]

        @tool("A candidate in your population, as `evaluate` takes it.", proxy=True)
        def candidate(id: str) -> Any:
            return value(run.find(run.scope(), id).genome)

        @tool("Score a candidate: one evaluation, unless it was scored before.", proxy=True, restore="result")
        async def evaluate(candidate: Any) -> dict:
            scope, turn = run.scope(), current_step().node.id
            if turn in scope.evaluated:
                return {"error": "one evaluate call per turn: read its result before evaluating again", "evals_left": scope.evals}
            try:
                genome = run.genome(candidate)
                scope.evaluated.add(turn)
                measured = await run.measure(scope, genome)
            except (ValueError, RuntimeError) as exc:
                return {"error": str(exc), "evals_left": scope.evals}
            evaluation = measured.evaluation
            return {"score": evaluation.objectives.get(run.task.objective), "objectives": dict(evaluation.objectives),
                    "feedback": truncate(evaluation.feedback, 2000), "evals_left": scope.evals}

        @tool("Run ops concurrently as fresh agents; returns what each handed back, and what you have left.",
              proxy=True, restore="result",
              when=lambda flow, node: node.agent_id in run.scopes and run.can_mutate(run.scopes[node.agent_id]))
        async def mutate(*ops: dict) -> dict:
            results = await run.mutate(list(ops))
            return {"ops": results, **run.left(run.scope())}

        return [population, candidate, evaluate, mutate]

    def genome(self, value: Any) -> Any:
        if self.task.files is None:
            return value
        if not isinstance(value, dict) or not all(isinstance(v, str) for v in value.values()):
            raise ValueError("a candidate is a dict from each editable file's path to its full text")
        if outside := [p for p in value if not self.task.files.is_genome(p)]:
            raise ValueError(f"{', '.join(outside)} are not editable")
        return Files.from_dict(value)

    async def measure(self, scope: Scope, genome: Any) -> Individual:
        """The genome measured: from the memo, or scored now (a draft) out of the scope's evaluations."""
        if (known := self.measured.get(key(genome))) is not None:
            return known
        if reason := self.experiment.limits.reached():
            raise RuntimeError(f"the run is over ({reason}): submit now")
        if scope.evals < 1:
            raise RuntimeError("no evaluations left: submit now")
        scope.evals -= 1
        individual = Individual.new(genome, info={"generation": scope.generation})
        evaluation = await self.experiment.evaluate(individual)
        individual = individual.with_evaluation(evaluation)
        self.experiment.note(
            "draft", ", ".join(f"{k}={v:.6g}" for k, v in evaluation.objectives.items()) or "failed", evaluation.usage,
            objectives=dict(evaluation.objectives), improved=improves(self.archive, individual, self.experiment.best()),
        )
        if evaluation.objectives:
            self.measured[key(genome)] = individual
        return individual

    async def mutate(self, ops: list) -> list[dict]:
        step, scope = current_step(), self.scope()
        g = step.graph
        if step.node.id in scope.mutated:
            raise RuntimeError("one mutate call per turn: read its results before planning more")
        if self.budget_exhausted(g, step.agent):
            raise RuntimeError("the run's token budget is spent: submit now")
        if not self.can_mutate(scope):
            raise RuntimeError("you cannot mutate: no generations, evaluations, agent calls or depth left, or the run is over")
        if not 1 <= len(ops) <= self.max_ops:
            raise ValueError(f"mutate takes 1 to {self.max_ops} ops")
        default = self.eval_budget if scope.depth == 0 else max(1, scope.evals // len(ops))
        plans = []
        for op in ops:
            if not isinstance(op, dict) or not op.get("instruction"):
                raise ValueError("every op is a dict with an instruction")
            parents = [self.find(scope, id) for id in op.get("parent_ids") or []]
            evals, agents = int(op.get("evals", default)), int(op.get("agents", 1))
            if evals < 1 or agents < 1:
                raise ValueError("every op needs at least one evaluation and one agent call (its own)")
            plans.append((op["instruction"], parents, evals, agents))
        evals, agents = sum(p[2] for p in plans), sum(p[3] for p in plans)
        if evals > scope.evals or agents > scope.agents:
            raise ValueError(f"these ops need {evals} evaluations and {agents} agent calls; you have {scope.evals} and {scope.agents}")
        scope.mutated.add(step.node.id)
        scope.evals, scope.agents = scope.evals - evals, scope.agents - agents
        generation = self.generation
        if scope.depth == 0:
            self.generation += 1
        launched = []
        for i, (instruction, parents, evals, agents) in enumerate(plans):
            child = Scope(scope.depth + 1, evals, agents - 1, generation, scope.population, instruction, tuple(p.id for p in parents))
            agent = self.spawn(step, self.child_task(child, parents), config=self.sized(evals), name=f"g{generation}_{len(scope.mutated)}_{i}")
            child.population = PopulationArchive(self.cap, name=f"populations/{agent.id}")
            child.population.attach(self.experiment.storage, self.task.objective, self.task.minimize)
            for parent in parents:
                child.population.add(parent)
            self.scopes[agent.id] = child
            launched.append((agent, child, evals, agents))
        if (code := current_code()) is not None:
            code.data.setdefault(AWAITED_KEY, []).extend(agent.id for agent, *_ in launched)

        async def finished(agent) -> None:
            with waiting(self, g, step.agent, agent):
                await self.queue.join(lambda: g.terminal(agent))

        await asyncio.gather(*(finished(agent) for agent, *_ in launched))
        results = []
        for agent, child, evals, agents in launched:
            returned = g.result(agent)
            notes = truncate(str(returned.get("notes", "")), 1000) if isinstance(returned, dict) else ""
            op = {"instruction": child.instruction, "parent_ids": list(child.parents), "notes": notes}
            try:
                individual = await self.handed_up(child, returned, g.path(agent))
                result = {
                    **row(individual, self.task.objective), **op,
                    "unchanged": individual.id in child.parents, "admitted": await self.admit(scope, individual),
                }
            except (ValueError, RuntimeError) as exc:
                result = {**op, "dropped": str(exc)}
            results.append(result | {"evals_used": evals - child.evals, "agents_used": agents - child.agents})
            scope.evals, scope.agents = scope.evals + child.evals, scope.agents + child.agents
        if scope.depth == 0:
            self.experiment.note("generation", f"generation {generation}: {len(ops)} ops", generation=generation, results=results)
        return results

    def sized(self, evals: int) -> str:
        """The name of the shared config with enough turns to spend `evals` evaluations."""
        base = self.config("evorlm")
        if (turns := 2 * evals + 4) <= base.max_iters:
            return "evorlm"
        name = f"evorlm-{turns}"
        self.configs.setdefault(name, replace(base, name=name, max_iters=turns))
        return name

    def child_task(self, child: Scope, parents: list[Individual]) -> str:
        shown = []
        for parent in parents:
            source = json.dumps(value(parent.genome), indent=1) if self.task.files is None else "\n\n".join(
                f"`{path}`:\n```\n{truncate(text, SOURCE_LIMIT)}\n```" for path, text in value(parent.genome).items()
            )
            shown.append(f"### {parent.id}: score {parent.objective(self.task.objective)}, {parent.name or 'untitled'}\n\n{source}")
        return CHILD.format(task=self.instructions, instruction=child.instruction, budget=self.budget(child),
                            parents="\n\n".join(shown) or "(none: start from the task)")

    async def handed_up(self, scope: Scope, result: Any, path: str) -> Individual:
        """The candidate an agent returned, measured by the host."""
        if not isinstance(result, dict):
            raise ValueError(f"returned no candidate: {truncate(str(result), 300)}")
        if result.get("candidate_id"):
            return self.find(scope, result["candidate_id"])
        measured = await self.measure(scope, self.genome(result.get("candidate")))
        if not measured.evaluation.objectives:
            raise ValueError(f"its candidate failed: {truncate(measured.evaluation.feedback, 300)}")
        named = tuple(id for id in result.get("parent_ids") or () if isinstance(id, str) and scope.population.get(id) is not None)
        return replace(
            measured, parents=named or scope.parents, name=result.get("title", ""), description=result.get("notes", ""),
            info={**measured.info, "operator": "evorlm", "agent": path},
        )

    async def admit(self, scope: Scope, individual: Individual) -> bool:
        """Into the scope's population; at the root, the archive, recorded, its parents traced back
        to candidates the experiment holds."""
        if scope.depth > 0:
            self.local[individual.id] = individual
            return scope.population.add(individual)
        if self.recorded(individual.id):
            return individual.id in self.archive
        individual = replace(individual, parents=self.lineage(individual.parents))
        self.experiment.record_created([individual])
        evaluation = replace(individual.evaluation, usage={})  # its draft note counted the cost
        return (await self.experiment._tell(individual, evaluation)).info["admitted"]

    def recorded(self, id: str) -> bool:
        try:
            self.experiment.individual(id)
            return True
        except KeyError:
            return False

    def lineage(self, ids: tuple[str, ...]) -> tuple[str, ...]:
        """`ids` with each local candidate replaced by its nearest ancestors the experiment holds."""
        traced: list[str] = []
        for id in ids:
            if self.recorded(id):
                traced.append(id)
            elif id in self.local:
                traced.extend(self.lineage(self.local[id].parents))
        return tuple(dict.fromkeys(traced))


def row(individual: Individual, objective: str) -> dict:
    return {"id": individual.id, "score": individual.objective(objective), "title": individual.name, "parent_ids": list(individual.parents)}


def key(genome: Any) -> str:
    return genome.digest if isinstance(genome, Files) else json.dumps(genome, sort_keys=True, default=str)


def value(genome: Any) -> Any:
    return {path: genome.text(path) for path in genome.paths()} if isinstance(genome, Files) else genome


__all__ = ["Evolution", "Scratch", "evorlm"]
