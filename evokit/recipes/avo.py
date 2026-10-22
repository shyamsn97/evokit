"""AVO (agentic variation operators): one coding-agent session for the whole run, redirected when it stalls."""

from __future__ import annotations

import evokit
from evokit import history
from evokit.archive import Archive
from evokit.files import CodingAgent, CodingAgentOperator, Project
from evokit.loop import EvalFn
from evokit.records import Individual
from evokit.storage import Storage
from evokit.strategies import Greedy

REDIRECT = (
    "The last attempts did not improve the score. Step back: look at what has been tried and why it "
    "failed, abandon the current line of attack, and try a substantially different approach."
)


async def avo(
    project: Project,
    eval_fn: EvalFn,
    agent: CodingAgent,
    *,
    storage: Storage,
    archive: Archive,
    instructions: str | None = None,
    max_cost: float = 5.0,
    max_evaluations: int | None = None,
    max_draft_evals: int = 10,
    redirect_after: int = 5,
    redirect: str = REDIRECT,
) -> Individual | None:
    """The agent scores its drafts with `eval_fn` before submitting; after `redirect_after`
    rejections in a row, the next prompt tells it to change direction.

    The caller passes a seeded archive, usually BestArchive(ties="accept"), and an eval_fn that
    fails incorrect programs, so only correct ones are kept.
    """
    strategy = Greedy()
    operator = CodingAgentOperator(
        agent, project, draft_evaluate=eval_fn, max_draft_evals=max_draft_evals, persistent=True, instructions=instructions
    )
    rejected_in_a_row = 0
    while history.usage(storage)["cost_usd"] < max_cost and (
        max_evaluations is None or history.counts(storage)["evaluated"] < max_evaluations
    ):
        asked = await evokit.ask(storage, archive, strategy, operator)
        if asked.exhausted:
            break
        evaluated = await evokit.evaluate(storage, asked.individuals, eval_fn=eval_fn)
        admitted = await evokit.tell(storage, archive, evaluated)
        rejected_in_a_row = 0 if admitted else rejected_in_a_row + len(evaluated) + len(asked.failures)
        if rejected_in_a_row >= redirect_after:
            operator.redirect(redirect)
            history.note(storage, "redirect", redirect)
            rejected_in_a_row = 0
    return archive.best()
