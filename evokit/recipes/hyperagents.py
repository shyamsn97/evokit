"""DGM-Hyperagents (Zhang et al., 2026): DGM, where the improvement procedure is part of the genome."""

from __future__ import annotations

import functools
from typing import Any

from evokit.files import Files
from evokit.operators.meta_agent import meta_agent_modify
from evokit.records import Individual
from evokit.recipes.dgm import dgm
from evokit.task import Task


async def hyperagents(task: Task, *, fixed_meta_agent: Files | None = None, **dgm_options: Any) -> Individual | None:
    """The genome holds task_agent.py and meta_agent.py; the parent's meta agent edits a copy of
    the whole repository, itself included (`meta_agent_modify`). The task's evaluate runs
    task_agent.py on the tasks and returns logs under `logs/` (Evaluation.logs), where the next
    meta agent reads them. `dgm_options` are dgm's (dir, archive, workers, limits, ...).

    The paper's ablations: `fixed_meta_agent=seed_repository` (the seed's meta agent edits every
    parent), and `open_ended=False`.
    """
    modify = functools.partial(meta_agent_modify, fixed_meta_agent=fixed_meta_agent, objective=task.objective)
    return await dgm(task, modify=modify, **dgm_options)
