"""A persistent CodingAgentOperator: one conversation across children, scoring drafts, redirected between them."""

import asyncio

import pytest

import evokit as ek
from evokit.archives import BestArchive
from evokit.agents import CodingAgentOperator, CommandAgent
from evokit.files import Project, ScriptEvaluator
from evokit.storages import MemoryStorage


def setup(seed, tmp_path, fake_agent):
    project = Project(seed, genome=["main.py"], environment=tmp_path / "env")
    eval_fn = ScriptEvaluator(project, "python evaluate.py")  # exits 1 when X > 3
    operator = CodingAgentOperator(
        CommandAgent([*fake_agent, "{session}"]), project, eval_fn, ek.History(MemoryStorage()), max_draft_evals=2, persistent=True
    )
    parent = ek.Individual.new(project.seed_genome()).with_evaluation(ek.Evaluation({"score": -2.0}))
    return operator, parent


def test_persistent_session_drafts_and_redirect(seed, tmp_path, fake_agent):
    operator, parent = setup(seed, tmp_path, fake_agent)
    archive = BestArchive()

    first = asyncio.run(operator.create(ek.Selection((parent,)), archive))
    (child,) = first.children
    assert child.genome.text("main.py") == "X = 1.5\n" and child.parents == (parent.id,)
    assert "draft={'score': -1.5}" in child.name and "resumed=;" not in child.name
    assert first.usage["cost_usd"] == pytest.approx(0.01) and operator.history.usage()["draft_evals"] == 1
    assert operator.session == "sess-1"

    operator.redirect("Step back and try something else.")
    (second,) = asyncio.run(operator.create(ek.Selection((child,)), archive)).children
    assert "redirect=True" in second.name and "resumed=sess-1" in second.name
    assert second.info["redirect"] == "Step back and try something else."

    (third,) = asyncio.run(operator.create(ek.Selection((second,)), archive)).children
    assert "redirect=False" in third.name and "redirect" not in third.info

    resumed = CodingAgentOperator(operator.agent, operator.project, persistent=True, session="sess-1")
    (fourth,) = asyncio.run(resumed.create(ek.Selection((third,)), archive)).children
    assert "resumed=sess-1" in fourth.name


def test_persistent_session_serves_one_worker(seed, tmp_path, fake_agent):
    operator, parent = setup(seed, tmp_path, fake_agent)

    async def two_at_once():
        await asyncio.gather(*(operator.create(ek.Selection((parent,)), BestArchive()) for _ in range(2)))

    with pytest.raises(ek.UsageError, match="two workers"):
        asyncio.run(two_at_once())
