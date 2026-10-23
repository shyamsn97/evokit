"""EvoKit: evolutionary search where the variation operator can be a coding agent.

The core is three functions over four base classes:

    asked = await evokit.ask(storage, archive, strategy, operator, n)     # select, create
    evaluated = await evokit.evaluate(storage, asked.individuals, eval_fn=eval_fn)
    admitted = evokit.tell(storage, archive, evaluated)                    # add to the archive

Storage (where everything is written), Archive (the population), Strategy (who becomes a
parent), Operator (how children are made). An eval function `eval_fn(individual, workdir)`
returns an Evaluation. `history` records what was created, evaluated and spent. A recipe is a
plain loop over ask, evaluate and tell. Implementations live in folders named after their
base: storages/, archives/, strategies/, operators/. Optional packages: files/ (evolving files
in a project directory), recipes/, and interface/ (CLI, MCP, and skill). `load_task(dir)`
reads a task folder (TASK.md, config.yaml, evaluate.py, the genome files) into those objects.
"""

from evokit import archives, history, operators, storages, strategies
from evokit.archive import Archive
from evokit.errors import EnvironmentModified, EvoError, OperatorError, RunBusy, UsageError
from evokit.interface.commands import command, tools
from evokit.loop import EvalFn, Limits, ask, call_eval_fn, evaluate, tell
from evokit.operator import Mutate, Operator
from evokit.records import Asked, Evaluation, Failure, Individual, Offspring, Selection
from evokit.storage import Storage
from evokit.strategy import Strategy
from evokit.task import Task, load_task

__version__ = "0.1.0"

__all__ = [
    "Archive",
    "Asked",
    "EnvironmentModified",
    "EvalFn",
    "EvoError",
    "Evaluation",
    "Failure",
    "Individual",
    "Limits",
    "Mutate",
    "Offspring",
    "Operator",
    "OperatorError",
    "RunBusy",
    "Selection",
    "Storage",
    "Strategy",
    "Task",
    "UsageError",
    "archives",
    "ask",
    "call_eval_fn",
    "command",
    "evaluate",
    "history",
    "load_task",
    "operators",
    "storages",
    "strategies",
    "tell",
    "tools",
]
