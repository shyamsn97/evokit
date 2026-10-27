"""DGM-Hyperagents (Zhang et al., 2026): DGM, where the improvement procedure is part of the genome."""

from __future__ import annotations

from typing import Any

from evokit.archive import Archive
from evokit.agents import MetaAgentModify
from evokit.files import Files
from evokit.loop import EvalFn
from evokit.records import Individual
from evokit.recipes.dgm import dgm
from evokit.history import History


async def hyperagents(
    eval_fn: EvalFn,
    *,
    history: History,
    archive: Archive,
    fixed_meta_agent: Files | None = None,
    objective: str = "score",
    **dgm_options: Any,
) -> Individual | None:
    """The genome holds task_agent.py and meta_agent.py; the parent's meta agent edits a copy of
    the whole repository, itself included (MetaAgentModify). The eval_fn runs task_agent.py on
    the tasks and returns logs under `logs/` (Evaluation.logs), where the next meta agent reads them.

    The paper's ablations: `fixed_meta_agent=seed_repository` (the seed's meta agent edits every
    parent), and `open_ended=False` with a LatestArchive.
    """
    operator = MetaAgentModify(history.storage, fixed_meta_agent=fixed_meta_agent, objective=objective)
    return await dgm(eval_fn, history=history, archive=archive, operator=operator, objective=objective, **dgm_options)
