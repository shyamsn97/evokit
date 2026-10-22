"""The pieces AVO is made of: a persistent CodingAgentOperator that scores drafts and can be redirected."""

import asyncio

import pytest

import evokit as ek
from evokit.archives import BestArchive
from evokit.files import CodingAgentOperator, CommandAgent, Project, ScriptEvaluator


def setup(seed, tmp_path, fake_agent):
    project = Project(seed, genome=["main.py"], environment=tmp_path / "env")
    eval_fn = ScriptEvaluator(project, "python evaluate.py")  # exits 1 when X > 3
    operator = CodingAgentOperator(
        CommandAgent([*fake_agent, "{session}"]), project, draft_evaluate=eval_fn, persistent=True, max_draft_evals=2
    )
    parent = ek.Individual.new(project.seed_genome()).with_evaluation(ek.Evaluation({"score": -2.0}))
    return operator, parent


def test_persistent_session_drafts_and_redirect(seed, tmp_path, fake_agent):
    operator, parent = setup(seed, tmp_path, fake_agent)
    archive = BestArchive()

    first = asyncio.run(operator.create(ek.Selection((parent,)), archive))
    (child,) = first.children
    assert child.genome.text("main.py") == "X = 1.5\n" and child.parents == (parent.id,)
    assert "draft={'score': -1.5}" in child.info["summary"] and "resumed=;" not in child.info["summary"]
    assert first.usage["draft_evals"] == 1 and first.usage["cost_usd"] == pytest.approx(0.01)
    assert operator.session == "sess-1"

    operator.redirect("Step back and try something else.")
    (second,) = asyncio.run(operator.create(ek.Selection((child,)), archive)).children
    assert "redirect=True" in second.info["summary"] and "resumed=sess-1" in second.info["summary"]
    assert second.info["redirect"] == "Step back and try something else."

    (third,) = asyncio.run(operator.create(ek.Selection((second,)), archive)).children
    assert "redirect=False" in third.info["summary"] and "redirect" not in third.info

    resumed = CodingAgentOperator(operator.agent, operator.project, persistent=True, session="sess-1")
    (fourth,) = asyncio.run(resumed.create(ek.Selection((third,)), archive)).children
    assert "resumed=sess-1" in fourth.info["summary"]


def test_persistent_session_serves_one_worker(seed, tmp_path, fake_agent):
    operator, parent = setup(seed, tmp_path, fake_agent)

    async def two_at_once():
        await asyncio.gather(*(operator.create(ek.Selection((parent,)), BestArchive()) for _ in range(2)))

    with pytest.raises(ek.UsageError, match="two workers"):
        asyncio.run(two_at_once())
