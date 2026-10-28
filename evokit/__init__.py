"""EvoKit: evolutionary search where the variation operator can be a coding agent.

Four things: a Task (what evolves and how it is scored), an Experiment (one search on it: its
history, archive and limits), an Agent (prompt it, send it messages), and recipes, plain
functions that take every component and run a published algorithm:

    task = evokit.Task.from_directory("examples/circle_packing")   # or Task(seed, evaluate)
    best = await evokit.recipes.avo(task, evokit.ClaudeCode(), dir="experiments/avo-1", max_cost=5)

A recipe is a loop over the experiment: pick individuals from the archive, make a child (with an
agent, or a variation function), and submit it:

    experiment = task.create_experiment(dir, archive=evokit.BestArchive(ties="accept"), max_cost=5)
    async with experiment.start(agent):              # the agent reads, scores and submits
        while not experiment.limits.reached():
            await agent.prompt("Improve the program, then submit it.")

    parent, *others = experiment.archive.sample(3)                               # or by hand:
    if child := await experiment.create(evokit.operators.diff(model, parent, others, task=task)):
        await experiment.submit(child)

Implementations live in folders named after their base: storages/, archives/, operators/
(variation functions); strategies.py has the sampling strategies. Optional packages: files/
(evolving files in a folder), agents/ (coding agents), recipes/, and interface/ (CLI, MCP, and
skill).
"""

from evokit import archives, operators, storages, strategies
from evokit.agents import Agent, ClaudeCode, CommandAgent, ReplAgent, ToolCall, Turn
from evokit.archive import Archive, improves
from evokit.archives import (
    BestArchive,
    IslandArchive,
    LatestArchive,
    NoveltyArchive,
    PopulationArchive,
    SimplerBestArchive,
    UnboundedArchive,
)
from evokit.errors import EnvironmentModified, EvoError, OperatorError, RunBusy, UsageError
from evokit.evaluate import Evaluate, as_evaluate, evaluator
from evokit.experiment import Experiment, Limits, StopRule, open_experiment
from evokit.records import Evaluation, Individual
from evokit.storage import Storage
from evokit.strategies import Strategy
from evokit.task import Task
from evokit import recipes  # noqa: E402  (recipes use everything above)

__version__ = "0.1.0"

__all__ = [
    "Agent",
    "Archive",
    "BestArchive",
    "ClaudeCode",
    "CommandAgent",
    "EnvironmentModified",
    "Evaluate",
    "Evaluation",
    "EvoError",
    "Experiment",
    "Individual",
    "IslandArchive",
    "LatestArchive",
    "Limits",
    "NoveltyArchive",
    "OperatorError",
    "PopulationArchive",
    "ReplAgent",
    "RunBusy",
    "SimplerBestArchive",
    "StopRule",
    "Storage",
    "Strategy",
    "Task",
    "ToolCall",
    "Turn",
    "UnboundedArchive",
    "UsageError",
    "archives",
    "as_evaluate",
    "evaluator",
    "improves",
    "open_experiment",
    "operators",
    "recipes",
    "storages",
    "strategies",
]
