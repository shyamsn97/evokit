"""EvoKit: evolutionary search where the variation operator can be a coding agent.

The core is ask, an eval function, and tell, over four base classes:

    history = evokit.History(storage)                                        # the run's record
    asked = await evokit.ask(history, archive, strategy, operator, n)        # select, create
    evaluations = await asyncio.gather(*map(score, asked.individuals))       # score = @evokit.eval_fn
    admitted = await evokit.tell(history, archive, asked.individuals, evaluations)  # add to the archive

Storage (where everything is written), Archive (the population), Strategy (who becomes a
parent), Operator (how children are made). An agent that decides what is kept calls
`submit(history, archive, eval_fn, individual)`: scored and told in one step. An eval function `eval_fn(individual)` returns an
Evaluation; decorated with `evokit.eval_fn` it is async and turns crashes and timeouts into
failed evaluations. `History` records what was created, evaluated and spent. A recipe is a
plain loop over ask, evaluating and tell. Implementations live in folders named after their
base: storages/, archives/, strategies/, operators/. Optional packages: files/ (evolving files
in a project directory), agents/ (coding agents that edit them), recipes/, and interface/ (CLI, MCP, and skill). `load_task(dir)`
reads a task folder (TASK.md, config.yaml, evaluate.py, the genome files) into those objects.
"""

from evokit import archives, operators, storages, strategies
from evokit.archive import Archive
from evokit.errors import EnvironmentModified, EvoError, OperatorError, RunBusy, UsageError
from evokit.interface.commands import command, tools
from evokit.history import History
from evokit.loop import EvalFn, Limits, as_eval_fn, ask, eval_fn, submit, tell
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
    "History",
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
    "as_eval_fn",
    "ask",
    "command",
    "eval_fn",
    "load_task",
    "operators",
    "storages",
    "strategies",
    "submit",
    "tell",
    "tools",
]
